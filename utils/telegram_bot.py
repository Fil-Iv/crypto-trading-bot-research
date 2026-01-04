import os
import threading
import time
import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return None

# Support both package-relative and top-level imports.
try:
    from .notifier import send_telegram  # type: ignore
except Exception:
    from notifier import send_telegram  # type: ignore


class TelegramCommandBot:
    """Minimal Telegram long‑polling bot for /pause and /resume commands."""

    def __init__(self) -> None:
        load_dotenv()
        self.enabled = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._offset: int | None = None
        self._paused = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start polling Telegram for commands in a background thread."""
        if not self.enabled or not self.token or not self.chat_id:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        while True:
            try:
                params = {"timeout": 60}
                if self._offset is not None:
                    params["offset"] = self._offset
                resp = requests.get(url, params=params, timeout=65).json()
                for upd in resp.get("result", []):
                    self._offset = upd.get("update_id", 0) + 1
                    text = upd.get("message", {}).get("text", "")
                    if text == "/pause":
                        self._paused = True
                        send_telegram("⏸️ Bot paused")
                    elif text == "/resume":
                        self._paused = False
                        send_telegram("▶️ Bot resumed")
            except Exception:
                time.sleep(5)

    def is_paused(self) -> bool:
        return self._paused


TELEGRAM_BOT = TelegramCommandBot()
