"""HTTP API channel for programmatic bot-to-bot communication."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


class APIChannel(BaseChannel):
    """
    Lightweight HTTP API channel for programmatic bot-to-bot integration.

    Listens on POST /message and pushes the request into the message bus.
    Supports synchronous request-response: if the caller sets ``wait=true``,
    the endpoint holds the connection until the agent publishes a reply
    (resolved via an asyncio.Future keyed by chat_id).

    Auth: optional Bearer token (set ``api_key`` in config).
    IP filtering: optional ``allowed_ips`` list (empty = allow all).
    """

    name: str = "api"

    def __init__(self, config: Any, bus: MessageBus):
        super().__init__(config, bus)
        self._runner: Any | None = None
        # Pending sync-response futures, keyed by chat_id
        self._pending: dict[str, asyncio.Future[str]] = {}

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the aiohttp HTTP server."""
        try:
            from aiohttp import web
        except ImportError as exc:
            raise RuntimeError(
                "aiohttp is required for the API channel. "
                "Install it with: pip install aiohttp"
            ) from exc

        self._running = True
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/message", self._handle_request)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.config.port)
        await site.start()

        logger.info("API channel listening on 0.0.0.0:{}", self.config.port)

        # Keep coroutine alive until stop() sets _running=False
        while self._running:
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        """Shut down the HTTP server."""
        self._running = False
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    async def send(self, msg: OutboundMessage) -> None:
        """
        Deliver an outbound message back to a waiting caller.

        Progress messages are ignored (the caller only receives the final
        agent response). If no Future is pending for msg.chat_id the
        message is silently dropped (fire-and-forget callers don't wait).
        """
        if msg.metadata.get("_progress"):
            return  # Skip intermediate progress/tool-hint messages

        future = self._pending.get(msg.chat_id)
        if future and not future.done():
            future.set_result(msg.content)

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: Any) -> Any:
        """GET /health — no auth required, used by master for liveness checks."""
        from aiohttp import web
        return web.json_response({"status": "ok", "channel": "api", "port": self.config.port})

    async def _handle_request(self, request: Any) -> Any:
        """Handle POST /message requests."""
        from aiohttp import web

        # --- Bearer token auth ---
        if self.config.api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {self.config.api_key}":
                logger.warning("API channel: invalid or missing API key from {}", request.remote)
                return web.Response(status=401, text="Unauthorized")

        # --- IP allowlist ---
        if self.config.allowed_ips:
            peer_ip = request.remote
            if peer_ip not in self.config.allowed_ips:
                logger.warning("API channel: IP {} not in allowed_ips", peer_ip)
                return web.Response(status=403, text="Forbidden")

        # --- Parse body ---
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="Invalid JSON body")

        content: str = body.get("content", "")
        if not content:
            return web.Response(status=400, text="Missing required field: content")

        sender_id: str = str(body.get("sender_id", "api"))
        chat_id: str = str(body.get("chat_id", str(uuid.uuid4())))
        wait: bool = bool(body.get("wait", False))
        timeout: float = float(body.get("timeout", 30.0))

        # --- Register sync-response future BEFORE publishing ---
        if wait:
            loop = asyncio.get_event_loop()
            future: asyncio.Future[str] = loop.create_future()
            self._pending[chat_id] = future

        await self._handle_message(sender_id, chat_id, content)

        if not wait:
            return web.json_response({"ok": True, "chat_id": chat_id})

        # --- Wait for agent response ---
        try:
            response_text = await asyncio.wait_for(future, timeout=timeout)
            return web.json_response({"response": response_text, "chat_id": chat_id})
        except asyncio.TimeoutError:
            logger.warning("API channel: response timeout for chat_id={}", chat_id)
            return web.json_response(
                {"error": "timeout", "chat_id": chat_id}, status=504
            )
        finally:
            self._pending.pop(chat_id, None)
