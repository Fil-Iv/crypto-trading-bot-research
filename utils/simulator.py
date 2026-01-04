# utils/simulator.py
from __future__ import annotations
import time
try:
    from .logger import log
    from .notifier import send_telegram
except Exception:
    def log(x): print(x)
    def send_telegram(x): print(x)

class PaperSimulator:
    """
    Simple paper‑trading simulator that repeatedly calls the bot's trading
    method at a fixed interval. The simulator will prefer a ``trade`` method
    if present, falling back to ``step`` for backward compatibility. Each
    iteration logs the result of the trading call.

    Parameters
    ----------
    bot_cls : type
        The bot class to instantiate for simulation.
    interval_sec : int, optional
        Number of seconds to wait between iterations, by default 15.
    """
    def __init__(self, bot_cls, interval_sec: int = 15) -> None:
        self.bot = bot_cls()
        self.interval = int(interval_sec)

    def run_forever(self) -> None:
        """Continuously run the bot's trading loop at the configured interval."""
        try:
            send_telegram("🧪 Paper-trade simulator started.")
        except Exception:
            pass
        while True:
            try:
                # Prefer the new trade() method; fallback to step() if present
                if hasattr(self.bot, 'trade'):
                    result = self.bot.trade()
                elif hasattr(self.bot, 'step'):
                    result = self.bot.step()
                else:
                    result = None
                log(f"[SIM] {result}")
            except Exception as e:
                log(f"[SIM] loop error: {e}")
            time.sleep(self.interval)
