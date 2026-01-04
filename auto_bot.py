"""
Entry point for the fully automated trading agent.
"""

import os
import time
from datetime import datetime, timezone
import sys
import pathlib
import threading

# --- package path bootstrap (run as script) ---
if __package__ in (None, "",):
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))

# Import logger and notifier with fallbacks
try:
    from utils.logger import log  # type: ignore
except Exception:
    # Fallback to main_bot.log if utils.logger is unavailable
    try:
        from main_bot import log  # type: ignore
    except Exception:
        def log(msg: str) -> None:  # type: ignore
            print(msg)

try:
    from utils.notifier import send_telegram  # type: ignore
except Exception:
    try:
        from main_bot import send_telegram  # type: ignore
    except Exception:
        def send_telegram(msg: str) -> None:  # type: ignore
            log(msg)

# Telegram command bot
try:
    from utils.telegram_bot import TELEGRAM_BOT  # type: ignore
except Exception:
    class _DummyBot:
        def start(self) -> None:
            pass
        def is_paused(self) -> bool:
            return False
    TELEGRAM_BOT = _DummyBot()  # type: ignore

# dotenv (optional)
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

from parameter_optimizer import optimise_parameters, update_best_params
from main_bot import SmartCryptoBot


def _async_call(fn, *args, **kwargs) -> threading.Thread:
    """Run a function in a daemon thread and return the thread."""
    t = threading.Thread(target=lambda: fn(*args, **kwargs), daemon=True)
    t.start()
    return t


def run_autobot():
    """Run the fully automated bot with periodic optimisation and training (async)."""
    os.environ.setdefault("DRY_RUN", "0")
    load_dotenv()

    try:
        send_telegram("🚀 Bot started")
    except Exception:
        pass

    TELEGRAM_BOT.start()

    # --- optional initial optimisation (async), only if enabled ---
    try:
        if int(os.getenv("OPT_SAMPLES", "0")) > 0:
            _async_call(lambda: update_best_params(optimise_parameters().get("threshold")))
            log("🎯 Стартирах начална оптимизация (async).")
        else:
            log("[opt] Пропускам първоначалната оптимизация (OPT_SAMPLES=0).")
    except Exception as e:
        log(f"[WARN] Началната оптимизация не стартира: {e}")

    # --- bot instance ---
    bot = SmartCryptoBot()
    # Perform an initial synchronous training so that first predictions are meaningful.
    try:
        bot.train_models()
        log("🧠 Началното обучение завърши.")
    except Exception as e:
        # Fallback to async training if synchronous training fails.
        log(f"[WARN] Началното обучение не успя: {e}")
        _async_call(bot.train_models)

    last_optimisation = datetime.now(timezone.utc)

    try:
        while True:
            now = datetime.now(timezone.utc)

            if TELEGRAM_BOT.is_paused():
                time.sleep(5)
                continue

            # на всеки 6 часа: обучение + (по желание) оптимизация — асинхронно
            if (now - last_optimisation).total_seconds() >= 6 * 3600:
                try:
                    _async_call(bot.train_models)
                    if int(os.getenv("OPT_SAMPLES", "0")) > 0:
                        _async_call(lambda: update_best_params(optimise_parameters().get("threshold")))
                except Exception as e:
                    log(f"[WARN] Ре-оптимизация/тренировка не стартира: {e}")
                last_optimisation = now

            try:
                # ✅ Ново: on_tick() преди trade()
                if hasattr(bot, "trade_manager") and hasattr(bot.trade_manager, "on_tick"):
                    try:
                        if hasattr(bot, "trade_manager") and hasattr(bot.trade_manager, "on_tick"):
                            bot.trade_manager.on_tick(bot.exchange)


                    except Exception as e:
                        log(f"[on_tick error] {e}")

                # 🧠 Основна стъпка на търговия
                bot.trade()
            except Exception as e:
                log(f"[cycle] error: {e}")
                try:
                    send_telegram(f"[Error] {e}")
                except Exception:
                    pass

            # shorten the sleep interval to 60 seconds to check positions more frequently
            time.sleep(60)

    except KeyboardInterrupt:
        log("Interrupted by user")
    finally:
        try:
            send_telegram("🛑 Bot stopped")
        except Exception:
            pass


if __name__ == "__main__":
    run_autobot()
