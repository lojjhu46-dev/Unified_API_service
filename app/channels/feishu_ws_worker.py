"""Feishu long-connection event worker."""

import asyncio
import json
import threading
from concurrent.futures import Future

from app.config import settings
from app.main import handle_feishu_event_body
from app.observability.logging import get_logger

logger = get_logger(__name__)


def _sdk_event_to_body(lark, data) -> dict:
    payload = lark.JSON.marshal(data)
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported Feishu SDK event payload: {type(payload).__name__}")


class AsyncLoopRunner:
    """Run SDK callback coroutines on a long-lived event loop."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="feishu-ws-event-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


def build_event_handler(lark, loop_runner: AsyncLoopRunner | None = None, card_response_builder=None):
    runner = loop_runner or AsyncLoopRunner()

    def on_message(data) -> None:
        body = _sdk_event_to_body(lark, data)
        future = runner.submit(handle_feishu_event_body(body, source="ws"))
        future.add_done_callback(_log_callback_error)

    def on_card_action(data):
        body = _sdk_event_to_body(lark, data)
        result = runner.submit(handle_feishu_event_body(body, source="ws")).result(timeout=10)
        if card_response_builder is None:
            try:
                from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTriggerResponse
                return P2CardActionTriggerResponse(result)
            except Exception:
                return result
        return card_response_builder(result)

    return (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(on_message)
        .register_p2_card_action_trigger(on_card_action)
        .build()
    )


def _log_callback_error(future: Future) -> None:
    try:
        future.result()
    except Exception as e:
        logger.error("Feishu long-connection event handling failed: %s", e, exc_info=True)


def run_worker() -> None:
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are required for Feishu long connection")

    import lark_oapi as lark

    event_handler = build_event_handler(lark)
    client = lark.ws.Client(
        settings.feishu_app_id,
        settings.feishu_app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )
    logger.info("Feishu long-connection worker starting")
    client.start()


if __name__ == "__main__":
    run_worker()
