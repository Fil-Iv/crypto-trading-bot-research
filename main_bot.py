from __future__ import annotations
import os
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
try:
    # Support SVM models if sklearn is available
    from sklearn.svm import SVC  # type: ignore
except Exception:
    SVC = None  # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

"""
This module defines a SmartCryptoBot that performs data collection, feature
engineering, model training and trading logic. The original version relied on
a number of external utility modules (located under a `utils` package) for
logging, notifications, exchange access, indicator calculations, model
selection, feature configuration and adaptive parameters. In many deployment
environments these modules may not be present. To improve portability and
extend functionality, the imports below are wrapped in try/except blocks
providing fallbacks.  Additional indicators (MACD, RSI, Bollinger Bands,
Stochastic Oscillator) are computed locally, and a simple model selection
routine based on cross‑validation is provided.  A dummy exchange and capital
manager are also defined when the corresponding modules are unavailable.
"""

# Import optional utilities; provide fallbacks if they are unavailable.
try:
    from utils.logger import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        """Fallback logger that prints to stdout."""
        print(msg)

try:
    from utils.notifier import send_telegram  # type: ignore
except Exception:
    def send_telegram(msg: str) -> None:
        """Fallback notifier that logs the message."""
        log(msg)

try:
    from utils.exchange_factory import get_exchange, DRY_RUN  # type: ignore
except Exception:
    # Define a simple dummy exchange with the minimal API used in this module.
    import numpy as _np
    class DummyExchange:
        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 20000):
            """Return synthetic OHLCV data for demonstration purposes."""
            import pandas as _pd
            now = datetime.now()
            # Create equally spaced timestamps going backwards
            dates = _pd.date_range(end=now, periods=limit, freq='15min')
            # Generate synthetic open, high, low, close, volume data
            data = _np.random.rand(limit, 5)
            df = _pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"], index=dates)
            # Reset index to match ccxt return shape (timestamp first)
            return df.reset_index().values.tolist()
        def fetch_ticker(self, symbol: str):
            return {"last": float(_np.random.rand())}
        def create_market_buy_order(self, symbol: str, amount: float) -> None:
            log(f"Dummy buy order: {symbol} amount={amount:.4f}")
    def get_exchange() -> DummyExchange:
        return DummyExchange()
    DRY_RUN = True  # When using the dummy exchange, always dry run

try:
    # We will use our own indicator computation; ignore external one if missing
    from utils.add_indicators import add_indicators  # type: ignore
except Exception:
    add_indicators = None  # Placeholder; our own indicator functions are defined below

try:
    from utils.model_selector import evaluate_models  # type: ignore
except Exception:
    evaluate_models = None  # We'll define a simple evaluation routine later

try:
    from utils.get_top_pairs import get_top_usdc_pairs  # type: ignore
except Exception:
    def get_top_usdc_pairs(exchange: Any, limit: int = 10) -> list[str]:
        """Fallback that returns a default list of popular USDC pairs."""
        return ["BTC/USDC", "ETH/USDC", "SOL/USDC", "ADA/USDC", "BNB/USDC"][:limit]

try:
    from utils.feature_config import get_feature_columns  # type: ignore
except Exception:
    def get_feature_columns() -> list[str]:
        """Return a list of feature names to be used for model training.

        This fallback includes OHLCV data and several technical indicators.
        """
        # Base OHLCV and indicator features. We include fractal peaks/troughs
        # and rolling 5‑day range features to provide both local and
        # medium‑term context for price action. If you change the
        # ``ROLLING_DAYS`` environment variable, adjust the feature names
        # accordingly (e.g. high_10d, low_10d, pos_in_10d_range).
        return [
            # Base OHLCV columns
            "open", "high", "low", "close", "volume",
            # Standard indicators
            "macd", "macd_signal", "macd_hist",
            "rsi_14", "bollinger_high", "bollinger_low",
            "stoch_k", "stoch_d",
            # Higher timeframe indicators
            "MACD_1h", "RSI_4h",
            # Fractal peaks and troughs
            "fractal_up", "fractal_down",
            # Rolling range features (e.g. high_5d, low_5d, pos_in_5d_range)
            # The actual day count is derived from ROLLING_DAYS environment variable
            "high_5d", "low_5d", "pos_in_5d_range",
            # New technical indicators for trend strength and momentum
            "adx", "cci", "roc", "atr"
        ]

try:
    from utils.adaptive_strategy import adaptive_parameters  # type: ignore
except Exception:
    def adaptive_parameters() -> dict[str, float]:
        """Fallback adaptive parameters returning default threshold, TP and SL.

        These values can be overridden via environment variables.
        """
        thr = float(os.getenv("THRESHOLD", "0.55"))
        tp = float(os.getenv("TP", "0.02"))
        sl = float(os.getenv("SL", "0.01"))
        aggressiveness = float(os.getenv("AGGRESSIVENESS", "1.0"))
        return {"threshold": thr, "tp": tp, "sl": sl, "aggressiveness": aggressiveness}

try:
    from utils.metrics_server import start_metrics_server  # type: ignore
except Exception:
    def start_metrics_server(*args, **kwargs) -> None:
        """No-op if metrics server is unavailable."""
        return None

try:
    from utils.trade_manager import execute_trade, monitor_positions, send_periodic_report, reconcile_open_positions  # type: ignore
except Exception:
    # Define no-op placeholders for trading functions if absent
    def execute_trade(*args, **kwargs) -> None:
        log("execute_trade called (no-op)")
    def monitor_positions(*args, **kwargs) -> None:
        log("monitor_positions called (no-op)")
    def send_periodic_report(*args, **kwargs) -> None:
        log("send_periodic_report called (no-op)")
    def reconcile_open_positions(*args, **kwargs) -> None:
        log("reconcile_open_positions called (no-op)")

# Try to import the parameter optimizer for periodic tuning of threshold/TP/SL.
# If unavailable, the bot will skip auto‑optimization.
try:
    from parameter_optimizer import search_params  # type: ignore
except Exception:
    search_params = None  # type: ignore

try:
    from utils.capital_manager import CapitalManager  # type: ignore
except Exception:
    class CapitalManager:
        """Simple capital manager that allocates a fixed amount for each trade."""
        def __init__(self, default_capital: float = 100.0) -> None:
            self.default_capital = default_capital
        def size_quote(self, exchange: Any, symbol: str, probability: float, tp: float, sl: float, aggressiveness: float) -> float:
            # Allocate more capital for higher confidence but cap at a max
            base = self.default_capital * aggressiveness
            alloc = base * probability
            return float(max(0.0, alloc))

from model_with_context import ModelWithContext

# -----------------------------------------------------------------------------
# Indicator computation utilities
#
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a set of technical indicators on the given OHLCV DataFrame.

    The function operates in place and returns the modified DataFrame with
    additional columns. It computes:
      * MACD (12/26 EMA) and signal line (9 EMA) plus histogram
      * 14‑period Relative Strength Index (RSI)
      * 20‑period Bollinger Bands (upper and lower)
      * 14‑period Stochastic Oscillator (K and D)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns 'open', 'high', 'low',
        'close', and 'volume'. The index is assumed to be timestamps.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with new indicator columns added.
    """
    if df.empty:
        return df

    # MACD: difference between 12‑period EMA and 26‑period EMA
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = macd_hist

    # RSI (14 periods)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df['rsi_14'] = rsi.fillna(50.0)  # fill missing with neutral value

    # Bollinger Bands (20 periods, 2 std)
    rolling_mean = df['close'].rolling(window=20, min_periods=20).mean()
    rolling_std = df['close'].rolling(window=20, min_periods=20).std()
    df['bollinger_high'] = rolling_mean + 2 * rolling_std
    df['bollinger_low'] = rolling_mean - 2 * rolling_std

    # Stochastic Oscillator (%K and %D)
    low_min = df['low'].rolling(window=14, min_periods=14).min()
    high_max = df['high'].rolling(window=14, min_periods=14).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, pd.NA)
    stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
    df['stoch_k'] = stoch_k.fillna(method='bfill').fillna(50.0)
    df['stoch_d'] = stoch_d.fillna(method='bfill').fillna(50.0)

    # ---------------------------------------------------------------------
    # Exponential Moving Average (EMA)
    #
    # To identify the prevailing trend, compute an exponential moving
    # average over the closing prices.  Trend‑following systems often
    # restrict trades to those in the direction of a long‑term moving
    # average.  For example, QuantifiedStrategies notes that moving
    # averages and ADX filters help avoid overtrading in sideways
    # markets【583346410250045†L140-L146】.  The period for the EMA is
    # configurable via the EMA_PERIOD environment variable.  When
    # EMA_FILTER_ENABLED=1, the bot will only trade when the price is
    # above (for long) this EMA.
    try:
        ema_period = int(os.getenv("EMA_PERIOD", "50"))
    except Exception:
        ema_period = 50
    if ema_period < 1:
        ema_period = 50
    try:
        df[f"ema_{ema_period}"] = df["close"].ewm(span=ema_period, adjust=False).mean()
    except Exception:
        df[f"ema_{ema_period}"] = pd.Series([np.nan] * len(df), index=df.index)

    # ---------------------------------------------------------------------
    # Peak and trough detection (Fractal indicator)
    #
    # In addition to the standard indicators above, we compute simple
    # fractal highs and lows to identify potential peaks and troughs in
    # price action. A fractal up (peak) occurs when a high is greater than
    # the highs of the two preceding and two succeeding candles. A fractal
    # down (trough) occurs when a low is lower than the lows of the two
    # preceding and two succeeding candles. These indicators help the
    # predictive model learn about local swing points.  Missing values at
    # the start/end of the series are filled with zeros.
    try:
        n = len(df)
        fractal_up = np.zeros(n)
        fractal_down = np.zeros(n)
        highs = df['high'].values
        lows = df['low'].values
        for i in range(2, n - 2):
            # detect local maximum
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
                fractal_up[i] = 1
            # detect local minimum
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
                fractal_down[i] = 1
        df['fractal_up'] = fractal_up
        df['fractal_down'] = fractal_down
    except Exception:
        # In case of any error (e.g. insufficient length), fall back to zeros
        df['fractal_up'] = 0.0
        df['fractal_down'] = 0.0

    # ---------------------------------------------------------------------
    # Rolling high/low range over multiple days
    #
    # To provide a broader context for price action beyond the very local
    # 5‑bar fractal pattern, compute the highest high and lowest low over
    # the past N days (default 5). This helps the model understand where
    # the current close sits within the recent trading range and can be
    # used to gauge overbought/oversold conditions. The number of days for
    # the rolling window is configurable via the ``ROLLING_DAYS`` environment
    # variable. We estimate the number of bars per day by measuring the
    # median time difference between consecutive timestamps. When timestamps
    # are missing from the DataFrame, we fall back to assuming 96 bars per
    # day (for a 15‑minute timeframe) which covers most cryptocurrency
    # exchanges.
    try:
        # Determine the number of days for the rolling window
        rolling_days = int(os.getenv("ROLLING_DAYS", "5"))
        # Estimate bar duration in seconds using timestamp differences
        bar_seconds = None
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
            ts_diff = ts.diff().dropna().median()
            if pd.notnull(ts_diff):
                # Convert timedelta to seconds
                bar_seconds = ts_diff.total_seconds()
        # Compute number of bars in the rolling window
        if bar_seconds and bar_seconds > 0:
            window_bars = max(int((rolling_days * 24 * 3600) / bar_seconds), 1)
        else:
            # Fallback: assume 15‑minute bars (96 per day)
            window_bars = rolling_days * 96
        # Compute rolling high and low
        high_roll = df['high'].rolling(window=window_bars, min_periods=window_bars).max()
        low_roll = df['low'].rolling(window=window_bars, min_periods=window_bars).min()
        df[f'high_{rolling_days}d'] = high_roll
        df[f'low_{rolling_days}d'] = low_roll
        # Position within the rolling range (0 = near low, 1 = near high)
        range_diff = high_roll - low_roll
        # Avoid division by zero
        range_diff = range_diff.replace(0, pd.NA)
        pos_in_range = (df['close'] - low_roll) / range_diff
        df[f'pos_in_{rolling_days}d_range'] = pos_in_range.fillna(0.5)
    except Exception:
        # On error, provide dummy columns filled with NaN or neutral values
        df['high_5d'] = df['high']
        df['low_5d'] = df['low']
        df['pos_in_5d_range'] = 0.5

    # ---------------------------------------------------------------------
    # Additional indicators: ADX, CCI, ROC, ATR
    #
    # To better capture trend strength, momentum and volatility, we compute
    # several classic technical indicators. The Average Directional Index
    # (ADX) measures the strength of a trend, the Commodity Channel Index
    # (CCI) identifies cyclical movements, the Rate of Change (ROC) captures
    # momentum, and the Average True Range (ATR) gauges volatility. These
    # indicators are computed via the `ta` library when available. If the
    # library is not present or an error occurs, we fall back to neutral
    # values so that models remain well-defined.
    try:
        import ta  # type: ignore
        # ADX using 14-period default
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        # CCI (Commodity Channel Index) with 20-period window and standard constant
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20, constant=0.015)
        # ROC (Rate of Change) with 12-period window
        df['roc'] = ta.momentum.roc(df['close'], window=12)
        # ATR (Average True Range) over 14 periods
        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr_indicator.average_true_range()
        # Fill missing values: use zeros for trend/momentum measures and median for ATR
        df['adx'] = df['adx'].fillna(0.0)
        df['cci'] = df['cci'].fillna(0.0)
        df['roc'] = df['roc'].fillna(0.0)
        if df['atr'].isna().all():
            # if ATR could not be computed, set a constant (1.0) to avoid division by zero
            df['atr'] = 1.0
        else:
            df['atr'] = df['atr'].fillna(df['atr'].median())
    except Exception:
        # Fallback when `ta` is unavailable or errors occur
        df['adx'] = 0.0
        df['cci'] = 0.0
        df['roc'] = 0.0
        df['atr'] = 1.0

    return df

# -----------------------------------------------------------------------------
# Model evaluation fallback
#
# The original implementation may import a model selection routine from
# `utils.model_selector.evaluate_models`. If that import fails (i.e. when
# `evaluate_models` is ``None``), we provide a simple cross‑validation based
# routine here. This function fetches recent OHLCV data for a few top symbols,
# computes the same set of indicators used during training and then fits a
# selection of candidate models to predict the direction of the next price
# movement. It evaluates each candidate using ROC AUC on a hold‑out set and
# returns the name of the model with the highest average score. The set of
# candidate models can be overridden via the environment variable
# ``MODEL_CANDIDATES`` (comma‑separated list). Valid options include
# ``lstm``, ``gbm`` (Gradient Boosting), ``rf`` (Random Forest), ``baseline``
# (logistic regression) and ``svm`` (support vector machine).

if evaluate_models is None:
    def evaluate_models() -> dict[str, str]:  # type: ignore
        """Select the best model type via a lightweight cross‑validation.

        Returns
        -------
        dict
            A dictionary containing the selected model name under the key
            "model".
        """
        try:
            ex = get_exchange()
        except Exception:
            # No exchange available; default to gradient boosting
            return {"model": "gbm"}

        # Determine candidate models: read from environment or use defaults
        cand_env = os.getenv("MODEL_CANDIDATES") or os.getenv("MODEL_TYPES")
        if cand_env:
            candidates = [c.strip() for c in cand_env.split(',') if c.strip()]
        else:
            candidates = ["gbm", "rf", "baseline"]
            # include LSTM only when tensorflow is available
            if tf is not None:
                candidates.append("lstm")
            if SVC is not None:
                candidates.append("svm")

        # Evaluate on a few symbols to avoid overfitting to a single asset
        try:
            syms = get_top_usdc_pairs(ex, limit=3)
        except Exception:
            syms = []
        if not syms:
            syms = ["BTC/USDC"]

        # Initialise score accumulator
        scores: dict[str, List[float]] = {c: [] for c in candidates}

        # For each symbol collect data and evaluate
        for sym in syms:
            try:
                # fetch ~2000 candles on 15 minute timeframe
                ohlcv = ex.fetch_ohlcv(sym, timeframe="15m", limit=2000)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = compute_indicators(df)
                # Only use features present in this timeframe
                feat_cols = [c for c in get_feature_columns() if c in df.columns]
                if not feat_cols:
                    continue
                features = df[feat_cols].dropna().values
                # require at least 61 rows (30 for lookback + 31 for target window)
                if len(features) < 61:
                    continue
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(features)
                # Build simple supervised dataset: predict direction of next bar
                X, y = [], []
                for i in range(30, len(scaled) - 1):
                    # flatten 30‑bar window for sklearn models
                    X.append(scaled[i - 30:i].reshape(-1))
                    y.append(1 if scaled[i + 1][0] > scaled[i][0] else 0)
                X = np.array(X)
                y = np.array(y)
                if len(X) == 0:
                    continue
                # sample at most 1000 examples to speed up
                if len(X) > 1000:
                    idx = np.linspace(0, len(X) - 1, 1000, dtype=int)
                    X = X[idx]
                    y = y[idx]
                # split 70/30
                n = len(X)
                split = int(n * 0.7)
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]
                # For each candidate model train and evaluate
                for m in candidates:
                    m_low = m.lower()
                    try:
                        if m_low == "lstm":
                            # We skip expensive LSTM evaluation; assume a neutral score
                            auc = 0.5
                        else:
                            # instantiate model
                            if m_low in ("gbm", "gradient_boosting"):
                                model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
                            elif m_low in ("rf", "random_forest"):
                                model = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
                            elif m_low in ("baseline", "dense"):
                                # logistic regression baseline
                                from sklearn.linear_model import LogisticRegression
                                model = LogisticRegression(max_iter=200)
                            elif m_low == "svm" and SVC is not None:
                                model = SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42)
                            else:
                                # fallback to gradient boosting
                                model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42)
                            model.fit(X_tr, y_tr)
                            proba = model.predict_proba(X_te)[:, 1]
                            if len(set(y_te)) > 1:
                                auc = float(roc_auc_score(y_te, proba))
                            else:
                                auc = 0.5
                    except Exception:
                        auc = 0.5
                    scores[m].append(auc)
            except Exception as e:
                log(f"[evaluate_models] error for {sym}: {e}")
                continue

        # compute average score for each candidate and pick the best
        best = None
        best_score = -1.0
        for m, vals in scores.items():
            if vals:
                avg = float(np.mean(vals))
            else:
                avg = 0.0
            if avg > best_score:
                best_score = avg
                best = m
        if not best:
            best = candidates[0]
        return {"model": best}

PARAMS_PATH = "best_params.json"

def build_model(model_type: str, input_shape: tuple[int, int]) -> Any:
    """
    Връща малък модел според model_type: Keras или sklearn.
    """
    if tf is None:
        class DummyModel:
            def predict(self, X, verbose=0):
                return np.full((len(X), 1), 0.5)
            def fit(self, X, y, epochs=1, batch_size=32, verbose=0, validation_split=0.0):
                return self
            def save(self, path):
                pass
        return DummyModel()

    model_type = (model_type or "baseline").lower()

    class SklearnModelWrapper:
        def __init__(self, cls_model: Any):
            self.model = cls_model
        def fit(self, X, y, epochs: int = 1, batch_size: int = 32,
                verbose: int = 0, validation_split: float = 0.0):
            self.model.fit(X, y)
            return self
        def predict(self, X, verbose: int = 0):
            try:
                proba = self.model.predict_proba(X)[:, 1]
            except Exception:
                proba = self.model.predict(X)
            return proba.reshape(-1, 1)
        def save(self, path: str):
            import joblib
            try:
                joblib.dump(self.model, path)
            except Exception:
                pass

    if model_type == "lstm":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    if model_type in ("gbm", "gradient_boosting"):
        gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        return SklearnModelWrapper(gbm)

    if model_type in ("rf", "random_forest"):
        # RandomForest tends to be biased when classes are imbalanced.  Pass
        # class_weight='balanced' so that the minority class is weighted more
        # heavily during training.  Without this the model often predicts
        # probabilities close to the dominant class proportion which degrades
        # trading performance.
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
        )
        return SklearnModelWrapper(rf)

    # Support support‑vector machines when requested
    if model_type == "svm":
        if SVC is not None:
            # Provide a class weight for imbalanced data sets.  SVC can accept a
            # dict for class_weight or the string 'balanced'.  Here we use
            # 'balanced' to automatically adjust weights inversely proportional
            # to class frequencies in the input data.
            svm_model = SVC(
                probability=True,
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=42,
                class_weight='balanced',
            )
            return SklearnModelWrapper(svm_model)
        else:
            # Fallback: if SVC is unavailable, use gradient boosting
            gbm = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )
            return SklearnModelWrapper(gbm)

    # Explicitly support logistic regression models when requested.  This option
    # provides a simple linear model that can serve as a baseline.  The
    # LogisticRegression is instantiated with class_weight='balanced' to
    # handle imbalanced target classes and max_iter to ensure convergence.
    if model_type in ("lr", "logistic_regression", "logistic"):
        from sklearn.linear_model import LogisticRegression
        lr_model = LogisticRegression(max_iter=500, class_weight='balanced')
        return SklearnModelWrapper(lr_model)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

class SmartCryptoBot:
    """
    Основен клас за автоматичен крипто бот. Управлява модели, прогнози и изпълнение на сделки.
    """
    def __init__(self):
        load_dotenv()
        self.exchange = get_exchange()
        if DRY_RUN:
            log("ℹ️ DRY_RUN=True – private endpoints се пропускат.")
        # инициализираме CapitalManager
        self.capital_manager = CapitalManager()
        # трейд мениджър (optional): attempt to import an existing TradeManager instance
        try:
            from utils.trade_manager import _tm_instance  # type: ignore
            self.trade_manager = _tm_instance
        except Exception:
            # Fallback to None if trade_manager is unavailable
            self.trade_manager = None
        # символи
        self.symbols = get_top_usdc_pairs(
            self.exchange,
            limit=int(os.getenv("OPT_SYMBOLS", "12"))
        )

        # Exclude pairs where the base currency is a fiat or stablecoin. This prevents
        # trading on pairs like EUR/USDC or other stablecoin combinations, which
        # typically offer minimal volatility and poor profit potential. Stable bases
        # can be customised via the STABLE_BASES environment variable (comma‑separated list).
        try:
            stable_env = os.getenv("STABLE_BASES") or "EUR,USD,USDT,USDC,FDUSD,TUSD,USDP,BUSD,DAI,EURT"
            self.stable_bases = {s.strip().upper() for s in stable_env.split(",") if s.strip()}
        except Exception:
            self.stable_bases = {"EUR", "USD", "USDT", "USDC", "FDUSD", "TUSD", "USDP", "BUSD", "DAI", "EURT"}
        # Filter symbols list to remove stable pairs
        try:
            self.symbols = [s for s in self.symbols if s.split("/")[0].upper() not in self.stable_bases]
        except Exception:
            pass
        # кешове за моделите
        self.model_map: Dict[str, Any] = {}
        self.scaler_map: Dict[str, MinMaxScaler] = {}
        self.calibrator_map: Dict[str, Any] = {}
        self.positions = self.load_positions()
        now = datetime.now(timezone.utc)
        self.last_train = now
        self.last_model_selection = now

        selection = evaluate_models()
        adapt = adaptive_parameters()
        self.selected_model_name = selection["model"]
        self.threshold = float(adapt["threshold"])
        self.tp = float(adapt["tp"])
        self.sl = float(adapt["sl"])
        self.aggressiveness = float(adapt["aggressiveness"])

        # Minimum volatility threshold for trading. Symbols whose recent price range
        # relative to price is below this threshold will be skipped. This helps
        # avoid trading on very stable pairs (e.g. fiat or stablecoins) that have
        # minimal movement and therefore poor profit potential. Configure via
        # environment variable MIN_VOLATILITY (e.g. 0.005 = 0.5%).
        try:
            self.min_volatility = float(os.getenv("MIN_VOLATILITY", "0.005"))
        except Exception:
            self.min_volatility = 0.005

        try:
            self.opt_thr_min = float(os.getenv("OPT_THR_MIN", "0.40"))
        except Exception:
            self.opt_thr_min = 0.40
        try:
            self.opt_thr_max = float(os.getenv("OPT_THR_MAX", "0.70"))
        except Exception:
            self.opt_thr_max = 0.70
        if not (0.0 < self.opt_thr_min < 1.0) or not (0.0 < self.opt_thr_max < 1.0) or self.opt_thr_min >= self.opt_thr_max:
            self.opt_thr_min, self.opt_thr_max = 0.40, 0.70

        try:
            self.ema_alpha = float(os.getenv("EMA_ALPHA", "0.30"))
            if not (0.0 < self.ema_alpha < 1.0):
                raise ValueError
        except Exception:
            self.ema_alpha = 0.30
        self.dust_threshold = float(os.getenv("DUST_THRESHOLD", "1.0"))

        # --- risk management parameters ---
        # Limit the proportion of available quote currency (e.g. USDC) to risk per
        # trade.  If not specified via the RISK_PCT environment variable, fall
        # back to 2 %.  Values outside (0,1) are clamped to the default.  This
        # setting will be used later when determining the capital allocated to
        # each new trade.  It acts as a hard cap on exposure per position and
        # aims to prevent outsized losses when stop‑losses are triggered.
        # Risk percentage per trade.  The default fallback is increased to 5 % to
        # allow larger position sizing when the user does not override
        # RISK_PCT via the environment.  Users should adjust this value in
        # their .env file according to their own risk tolerance.
        try:
            _rp = float(os.getenv("RISK_PCT", "0.05"))
            if 0.0 < _rp < 1.0:
                self.risk_pct = _rp
            else:
                raise ValueError
        except Exception:
            self.risk_pct = 0.05

        # Trailing stop configuration. If TRAILING_STOP is set (1/true), the bot will
        # update the stop‑loss for each open position as the price moves up, keeping a
        # constant distance of TRAIL_SL_PCT below the highest recorded price. This helps
        # to lock in profits without exiting on minor pullbacks. If not set, the
        # original fixed SL will be used.
        ts_env = os.getenv("TRAILING_STOP", "0").strip().lower()
        self.use_trailing_stop = ts_env in ("1", "true", "yes", "on")
        try:
            self.trail_sl_pct = float(os.getenv("TRAIL_SL_PCT", str(self.sl)))
            if self.trail_sl_pct <= 0.0 or self.trail_sl_pct >= 1.0:
                # Default to the existing SL percentage if provided percentage is invalid
                self.trail_sl_pct = self.sl
        except Exception:
            self.trail_sl_pct = self.sl

        # Logistic scaling factor for position sizing.  This governs the
        # steepness of the sigmoid used to convert the gap between the
        # confidence and threshold into a scaling factor.  A higher value
        # produces a more step‑like curve, resulting in larger position sizes
        # for modest increases in confidence above the threshold.  Configure
        # via LOGISTIC_FACTOR environment variable.  Default is 10.
        try:
            lf = float(os.getenv("LOGISTIC_FACTOR", "10"))
            if lf > 0:
                self.logistic_factor = lf
            else:
                raise ValueError
        except Exception:
            self.logistic_factor = 10.0

        # --- EMA trend filter settings ---
        # When EMA_FILTER_ENABLED=1, the bot will only open long trades when
        # the most recent closing price is above the exponential moving average
        # of period EMA_PERIOD.  This helps filter out trades during sideways
        # or bearish markets, as recommended by trend‑filtering articles
        # that emphasise moving averages and ADX to avoid overtrading【583346410250045†L140-L146】.
        ema_filter_env = os.getenv("EMA_FILTER_ENABLED", "0").strip().lower()
        self.ema_filter_enabled = ema_filter_env in ("1", "true", "yes", "on")
        try:
            ep = int(os.getenv("EMA_PERIOD", "50"))
            if ep >= 1:
                self.ema_period = ep
            else:
                raise ValueError
        except Exception:
            self.ema_period = 50

        # Auto‑optimization of threshold/TP/SL parameters.  When enabled,
        # the bot will periodically call parameter_optimizer.search_params() to
        # recompute these settings based on recent historical performance.  To
        # enable, set AUTO_OPTIMIZE=1 and optionally set
        # OPT_INTERVAL_HOURS to the desired number of hours between runs.
        auto_opt_env = os.getenv("AUTO_OPTIMIZE", "0").strip().lower()
        self.auto_optimize = auto_opt_env in ("1", "true", "yes", "on") and search_params is not None
        try:
            self.opt_interval_hours = float(os.getenv("OPT_INTERVAL_HOURS", "12"))
            if self.opt_interval_hours <= 0:
                raise ValueError
        except Exception:
            self.opt_interval_hours = 12.0
        # Track the last optimisation time in UTC
        self.last_opt_time = datetime.now(timezone.utc)

        # Pyramiding configuration.  When pyramiding is enabled, the bot will
        # add to a winning position as the market moves in its favour.  Each
        # additional buy is triggered when the price exceeds the last add price
        # by a percentage defined by PYRAMID_STEP_PCT.  The number of
        # additional buys per position is capped by PYRAMID_MAX_ADDS.  The
        # default step is 2 % and the maximum number of adds is 2.
        pyr_env = os.getenv("PYRAMID_ENABLED", "0").strip().lower()
        self.pyramid_enabled = pyr_env in ("1", "true", "yes", "on")
        try:
            step = float(os.getenv("PYRAMID_STEP_PCT", "0.02"))
            if step > 0.0:
                self.pyramid_step_pct = step
            else:
                raise ValueError
        except Exception:
            self.pyramid_step_pct = 0.02
        try:
            adds = int(os.getenv("PYRAMID_MAX_ADDS", "2"))
            if adds >= 0:
                self.pyramid_max_adds = adds
            else:
                raise ValueError
        except Exception:
            self.pyramid_max_adds = 2

        # --- minimum size factor for capital allocation ---
        # When scaling the trade size based on how much the confidence exceeds the threshold,
        # very small gaps can lead to near‑zero allocations.  To avoid skipping trades due
        # to excessively small position sizes, introduce a minimum size factor that
        # guarantees at least a fraction of the computed capital is allocated.  This value
        # can be configured via the MIN_SIZE_FACTOR environment variable (0–1).  If not set
        # or invalid, default to 0.5 (50 %).
        try:
            _msf = float(os.getenv("MIN_SIZE_FACTOR", "0.5"))
            if 0.0 < _msf <= 1.0:
                self.min_size_factor = _msf
            else:
                raise ValueError
        except Exception:
            self.min_size_factor = 0.5

        # Ensure the trailing stop distance is not too tight.  To reduce the
        # likelihood of premature stop‑outs from normal price noise, require
        # the trailing stop percentage to be at least 1.5× the fixed SL.  If
        # configured value is lower, raise it to this minimum.  This rule
        # applies only when trailing stops are enabled.
        try:
            if self.use_trailing_stop and (self.trail_sl_pct < (self.sl * 1.5)):
                self.trail_sl_pct = self.sl * 1.5
        except Exception:
            pass

        # If trailing stop is enabled, ensure any pre-existing open positions have
        # a max_price field for correct trailing stop behaviour. For positions
        # opened before this feature existed, initialise max_price to the entry price.
        try:
            if self.use_trailing_stop and isinstance(self.positions, dict):
                updated_positions = False
                for _sym, _pos in list(self.positions.items()):
                    if isinstance(_pos, dict) and _pos.get("status") == "open":
                        # Only add max_price if it doesn't already exist
                        if _pos.get("max_price") is None:
                            entry_price = 0.0
                            try:
                                entry_price = float(_pos.get("entry", 0.0))
                            except Exception:
                                pass
                            _pos["max_price"] = entry_price
                            updated_positions = True
                if updated_positions:
                    # Persist updates so that future runs maintain the max_price
                    self.save_positions()
        except Exception:
            # If anything goes wrong during the trailing stop initialisation,
            # silently ignore the error. This prevents the bot from
            # crashing on startup due to unexpected position formats.
            pass

        # Minimum trade sizing parameters.  Many exchanges enforce minimum trade
        # sizes for both the base asset and the notional value of an order.  To
        # avoid repeated order errors when buying or selling small amounts, the
        # bot exposes environment variables MIN_BASE_AMOUNT and MIN_NOTIONAL.
        # MIN_BASE_AMOUNT specifies the minimum quantity of the base asset (e.g.
        # 0.001 BNB).  MIN_NOTIONAL specifies the minimum quote currency value
        # (e.g. 10 USDC).  If these variables are not set, they default to
        # typical values or fall back to MIN_QUOTE_ALLOC for the notional.
        try:
            self.min_base_amount = float(os.getenv("MIN_BASE_AMOUNT", "0.001"))
        except Exception:
            self.min_base_amount = 0.001
        try:
            self.min_notional = float(os.getenv("MIN_NOTIONAL", os.getenv("MIN_QUOTE_ALLOC", "10")))
        except Exception:
            # fallback to 10 if parsing fails
            self.min_notional = 10.0

        # Dynamic threshold configuration.  These parameters control how the bot
        # adapts its entry threshold based on recent model predictions.  The
        # DYNAMIC_THRESHOLD_WINDOW specifies how many recent predictions to
        # consider, and DYNAMIC_THRESHOLD_PERCENTILE determines which percentile
        # to use.  The bot maintains a history of predictions in pred_history.
        try:
            self.dynamic_threshold_window = int(os.getenv("DYNAMIC_THRESHOLD_WINDOW", "200"))
            if self.dynamic_threshold_window < 1:
                raise ValueError
        except Exception:
            self.dynamic_threshold_window = 200
        try:
            self.dynamic_threshold_percentile = float(os.getenv("DYNAMIC_THRESHOLD_PERCENTILE", "70"))
            if not (0.0 < self.dynamic_threshold_percentile < 100.0):
                raise ValueError
        except Exception:
            self.dynamic_threshold_percentile = 70.0
        # Prediction history for dynamic threshold
        self.pred_history: List[float] = []

        # ADX threshold: minimum trend strength required for trading.  If the
        # latest ADX value on the 10‑minute timeframe is below this threshold,
        # the bot will skip the trade.  Default 20 indicates a weak trend.
        try:
            self.adx_threshold = float(os.getenv("ADX_THRESHOLD", "20"))
        except Exception:
            self.adx_threshold = 20.0

        # ATR stop/loss multipliers.  Stop loss and take profit distances are
        # derived from the Average True Range (ATR) times these multipliers.
        try:
            self.atr_sl_mult = float(os.getenv("ATR_SL_MULT", "2.0"))
        except Exception:
            self.atr_sl_mult = 2.0
        try:
            self.atr_tp_mult = float(os.getenv("ATR_TP_MULT", "3.0"))
        except Exception:
            self.atr_tp_mult = 3.0

        thr_env = os.getenv("THRESHOLD")
        if thr_env:
            try:
                thr_env_val = float(thr_env)
                if 0.0 < thr_env_val < 1.0:
                    self.threshold = thr_env_val
            except Exception:
                pass

        self._load_best_params_if_any()

        # Automatically recalculate adaptive parameters at startup.  This ensures
        # TP/SL/threshold values remain current with the latest market volatility
        # rather than relying solely on defaults at program launch.
        try:
            self._auto_update_parameters()
        except Exception:
            pass

        try:
            send_telegram(f"🤖 Ботът стартира. Модел: {self.selected_model_name.upper()}")
        except Exception:
            pass

        self.config = self._load_config()
        self.sentiment_weight: float = float(self.config.get("sentiment_weight", 0.05))
        model_types_env = os.getenv("MODEL_TYPES")
        if model_types_env:
            model_list = [m.strip() for m in model_types_env.split(",") if m.strip()]
            self.model_types = model_list or [self.selected_model_name]
            if model_list:
                self.selected_model_name = model_list[0]
        else:
            self.model_types = [self.selected_model_name]

        # Keep track of last snapshot of open positions to avoid spamming identical updates
        self._last_open_positions_snapshot: Optional[tuple] = None

        # --- Cooldown configuration ---
        # To prevent the bot from immediately re‑entering a position after it has
        # been closed, introduce a cooldown period in minutes.  When a position
        # hits its stop‑loss or take‑profit, the current timestamp is recorded
        # for that symbol.  The trade() method will skip opening a new
        # position for a symbol until this cooldown has elapsed.  The default
        # is 30 minutes but can be overridden via the COOLDOWN_MINUTES
        # environment variable.
        try:
            self.cooldown_minutes = float(os.getenv("COOLDOWN_MINUTES", "30"))
            if self.cooldown_minutes < 0:
                raise ValueError
        except Exception:
            self.cooldown_minutes = 30.0
        # Maintain a mapping from symbol to the UTC timestamp of the last time
        # it was closed.  When the cooldown has not yet expired, the bot will
        # not open a new position for that symbol.  Timestamps are stored in
        # seconds since epoch (time.time()).
        self.last_closed_time: Dict[str, float] = {}

        # Sequence length for input windows.  Traditionally the bot looked back
        # over the last 30 bars to build its features.  To allow users to
        # experiment with different context lengths without modifying the code,
        # read the SEQ_LEN environment variable (integer).  A larger value
        # increases the amount of historical context the model sees.  If not
        # provided or invalid, fall back to 30.
        try:
            self.seq_len = int(os.getenv("SEQ_LEN", "30"))
            if self.seq_len < 5:
                raise ValueError
        except Exception:
            self.seq_len = 30

        # Initialise trailing stop data for existing positions if required.
        # Older positions may not have 'max_price'. We set it to the entry price to
        # ensure trailing stops work for positions opened before enabling this feature.
        if self.use_trailing_stop:
            updated = False
            for sym, pos in self.positions.items():
                if not pos:
                    continue
                if pos.get("status") == "open" and "max_price" not in pos:
                    pos["max_price"] = float(pos.get("entry", 0.0))
                    updated = True
            if updated:
                # When trailing stop is enabled, ensure initial stop is not tighter than
                # the trailing distance. This prevents older positions from being
                # stopped out prematurely due to a low fixed SL configured previously.
                for sym, pos in self.positions.items():
                    if not pos or pos.get("status") != "open":
                        continue
                    max_price = float(pos.get("max_price", pos.get("entry", 0.0)))
                    current_sl = float(pos.get("sl", 0.0))
                    trail_based_sl = max_price * (1.0 - self.trail_sl_pct)
                    if trail_based_sl > current_sl:
                        pos["sl"] = trail_based_sl
                self.save_positions()

        # Immediately check any loaded positions against the latest market
        # prices.  If a position's current price is already below its stop‑loss,
        # mark it as closed to prevent carrying unrealised losses.
        try:
            self.close_outdated_positions()
        except Exception:
            pass

    def load_positions(self):
        try:
            with open("positions.json", "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_positions(self):
        try:
            with open("positions.json", "w") as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            log(f"[save_positions] Error: {e}")

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _get_news_sentiment(self) -> float:
        """Compute sentiment score from recent headlines.

        Whenever possible, use a lightweight NLP library (TextBlob) to derive
        sentiment polarity from each headline.  The polarity ranges from
        –1.0 (very negative) to +1.0 (very positive).  The final score is the
        average polarity of all headlines.  If TextBlob is not available,
        fall back to counting predefined positive/negative keywords.  If no
        headlines are present, return 0 (neutral sentiment).
        """
        try:
            headlines = self.config.get("latest_news") or []
            if not isinstance(headlines, list) or not headlines:
                return 0.0
            # Try to use TextBlob for sentiment analysis
            try:
                from textblob import TextBlob  # type: ignore
                sentiments = []
                for h in headlines:
                    try:
                        blob = TextBlob(h)
                        sentiments.append(blob.sentiment.polarity)
                    except Exception:
                        sentiments.append(0.0)
                if sentiments:
                    return float(sum(sentiments) / len(sentiments))
            except Exception:
                pass
            # Fallback: simple keyword counting
            pos_words = ("up", "bull", "surge", "rally", "positive", "beat")
            neg_words = ("down", "bear", "drop", "selloff", "negative", "miss")
            pos = sum(any(w in h.lower() for w in pos_words) for h in headlines)
            neg = sum(any(w in h.lower() for w in neg_words) for h in headlines)
            total = max(1, len(headlines))
            return float((pos - neg) / total)
        except Exception:
            return 0.0

    def _load_best_params_if_any(self) -> None:
        try:
            path = PARAMS_PATH
            if not os.path.isfile(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
            thr = float(params.get("threshold", 0))
            tp = float(params.get("tp", 0))
            sl = float(params.get("sl", 0))
            if 0.0 < thr < 1.0:
                self.threshold = thr
            if 0.0 < tp < 1.0:
                self.tp = tp
            if 0.0 < sl < 1.0:
                self.sl = sl
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Automatic parameter update
    #
    # On startup and periodically during training/trading, the bot should
    # refresh its threshold, take‑profit, stop‑loss and aggressiveness
    # settings based on current market volatility.  This method wraps
    # adaptive_parameters() and applies the returned values to the instance.
    def _auto_update_parameters(self) -> None:
        try:
            params = adaptive_parameters()
            # Only assign values that are sensible; fall back to existing ones
            thr = float(params.get("threshold", self.threshold))
            tp  = float(params.get("tp", self.tp))
            sl  = float(params.get("sl", self.sl))
            ag  = float(params.get("aggressiveness", self.aggressiveness))
            if 0.0 < thr < 1.0:
                self.threshold = thr
            if 0.0 < tp < 1.0:
                self.tp = tp
            if 0.0 < sl < 1.0:
                self.sl = sl
            if ag > 0.0:
                self.aggressiveness = ag
        except Exception:
            # If any error occurs, retain existing parameters
            pass

    # -----------------------------------------------------------------
    # Check and close outdated positions
    #
    # Iterate through the current positions and fetch the latest market
    # price for each open symbol.  If the price is below the stop‑loss, mark
    # the position as closed.  This prevents holding on to losing trades
    # indefinitely when the bot is restarted.
    def close_outdated_positions(self) -> None:
        if not isinstance(self.positions, dict):
            return
        updated = False
        for sym, pos in list(self.positions.items()):
            if not isinstance(pos, dict):
                continue
            if pos.get("status") != "open":
                continue
            try:
                ticker = self.exchange.fetch_ticker(sym)
                current_price = float(ticker.get("last", ticker.get("price", 0.0)))
            except Exception:
                # If fetching price fails, skip checking this symbol
                continue
            try:
                sl = float(pos.get("sl", 0.0))
            except Exception:
                sl = 0.0
            # If current price is below stop‑loss, close the position
            if sl > 0.0 and current_price <= sl:
                pos["status"] = "closed"
                updated = True
                # Enforce cooldown after a close to prevent immediate re-entry loops
                try:
                    self.last_closed_time[sym] = time.time()
                except Exception:
                    pass
                try:
                    log(f"[close_outdated_positions] Закриване на {sym}: цена {current_price} ≤ SL {sl}")
                except Exception:
                    pass
        if updated:
            self.save_positions()

    def fetch_data(self, symbol: str, limit: int = 12000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch OHLCV data for multiple timeframes and compute indicators.

        Parameters
        ----------
        symbol : str
            Trading pair to fetch (e.g. "BTC/USDC").
        limit : int, optional
            Maximum number of candles to retrieve per timeframe, by default 12000.

        Returns
        -------
        tuple of DataFrames
            DataFrames for 15m, 1h and 4h timeframes, each with indicators added.
        """
        def fetch_and_process(tfname: str) -> pd.DataFrame:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=tfname, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df.index.name = None
                # Remove any residual timestamp column if present
                if "timestamp" in df.columns:
                    del df["timestamp"]
                # Compute indicators: prefer external add_indicators if available; fall back to local compute_indicators
                if callable(add_indicators):
                    try:
                        return add_indicators(df.copy())
                    except Exception:
                        # fall through to local computation
                        pass
                return compute_indicators(df.copy())
            except Exception as e:
                log(f"Error fetching {tfname} data for {symbol}: {e}")
                return pd.DataFrame()

        return (
            fetch_and_process("15m"),
            fetch_and_process("1h"),
            fetch_and_process("4h")
        )

    def prepare_data(self, df_10m, df_1h, df_4h):
        try:
            # drop duplicated timestamp columns and normalise index names
            for df in [df_10m, df_1h, df_4h]:
                if "timestamp" in df.columns and df.index.name == "timestamp":
                    del df["timestamp"]
                    df.index.name = None
                elif df.index.name == "timestamp":
                    df.index.name = None
            df = df_10m.copy().dropna()
            # align 1h and 4h indicators using forward fill to avoid forward‑looking bias
            df["MACD_1h"] = df_1h["macd"].reindex(df.index, method="ffill")
            df["RSI_4h"]  = df_4h["rsi_14"].reindex(df.index, method="ffill")
            features = df[get_feature_columns()].dropna().values
            # require at least seq_len+1 rows to form one sample
            if len(features) < (self.seq_len + 1):
                return None, None, None
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(features)
            # lookahead window and minimum return from environment
            lookahead = int(os.getenv("LOOKAHEAD_BARS", "4"))
            target_ret = float(os.getenv("TARGET_RET_PCT", "0"))
            X, y = [], []
            for i in range(self.seq_len, len(scaled) - lookahead):
                X.append(scaled[i - self.seq_len:i])
                future_close = scaled[i + lookahead][0]
                current_close = scaled[i][0]
                if target_ret > 0.0:
                    change = (future_close - current_close) / max(abs(current_close), 1e-9)
                    label = 1 if change >= target_ret else 0
                else:
                    label = 1 if future_close > current_close else 0
                y.append(label)
            return np.array(X), np.array(y), scaler
        except Exception as e:
            log(f"Error in prepare_data: {e}")
            return None, None, None

    def train_models(self):
        threshold_candidates: List[float] = []
        for symbol in self.symbols:
            try:
                df_10m, df_1h, df_4h = self.fetch_data(symbol)
                X, y, scaler = self.prepare_data(df_10m, df_1h, df_4h)
                if X is None or len(X) == 0:
                    continue
                models_for_symbol: Dict[str, Any] = {}
                ensemble_raw_preds: Optional[np.ndarray] = None
                for mtype in self.model_types:
                    try:
                        # Build the model for this type.  Use the configured sequence length
                        # for the time dimension instead of a hard‑coded 30.  The build_model
                        # function returns either a scikit‑learn model (for gbm, rf, lr, etc.)
                        # or a tf.keras.Model for LSTM/baseline/svm variants.
                        mdl = build_model(model_type=mtype,
                                          input_shape=(self.seq_len, len(get_feature_columns())))

                        # Determine how to shape the input depending on whether this is a Keras model.
                        # Keras models expect a 3D array of shape (samples, timesteps, features),
                        # whereas scikit‑learn models expect a 2D matrix (samples, flattened features).
                        if tf is not None and isinstance(mdl, tf.keras.Model):
                            # Keep the 3D tensor shape for keras models: (n_samples, seq_len, n_features)
                            X_train = X
                            raw_input = X
                        else:
                            # Flatten the timesteps and features into a single dimension for sklearn
                            # models: (n_samples, seq_len * n_features)
                            X_train = X.reshape((X.shape[0], -1))
                            raw_input = X_train

                        # Train the model.  For sklearn models the extra args are ignored.
                        mdl.fit(X_train, y, epochs=5, batch_size=32, verbose=0, validation_split=0.1)
                        models_for_symbol[mtype] = mdl

                        # Collect raw predictions from this model for potential calibration.
                        try:
                            preds = mdl.predict(raw_input, verbose=0).reshape(-1)
                        except Exception:
                            preds = mdl.predict(raw_input).reshape(-1)
                        if ensemble_raw_preds is None:
                            ensemble_raw_preds = preds
                        else:
                            ensemble_raw_preds += preds
                    except Exception as e:
                        log(f"[train_models] error training {mtype} for {symbol}: {e}")
                        continue
                if models_for_symbol:
                    self.model_map[symbol] = models_for_symbol
                    self.scaler_map[symbol] = scaler
                    try:
                        import joblib
                        for mtype, mdl in models_for_symbol.items():
                            if tf and isinstance(mdl, tf.keras.Model):
                                mdl.save(f"models/{symbol.replace('/', '_')}_{mtype}.keras")
                            else:
                                joblib.dump(mdl, f"models/{symbol.replace('/', '_')}_{mtype}.pkl")
                        joblib.dump(scaler, f"models/{symbol.replace('/', '_')}_scaler.pkl")
                    except Exception:
                        pass
                    # After saving the models and scaler, build a calibration model if possible.
                    if ensemble_raw_preds is not None:
                        avg_raw = ensemble_raw_preds / float(len(models_for_symbol))
                        if len(set(y)) > 1:
                            try:
                                # Use class_weight='balanced' to improve calibration when
                                # classes are imbalanced.  Without this, the calibrator
                                # tends to bias towards the majority class resulting in
                                # thresholds near 0.5 regardless of actual performance.
                                cal = LogisticRegression(class_weight='balanced')
                                cal.fit(avg_raw.reshape(-1, 1), y)
                                self.calibrator_map[symbol] = cal
                                coef = float(cal.coef_[0][0])
                                intercept = float(cal.intercept_[0])
                                if abs(coef) > 1e-6:
                                    thr = -intercept / coef
                                    if 0.0 < thr < 1.0:
                                        threshold_candidates.append(float(thr))
                            except Exception as e:
                                log(f"[train_models] calibrator error for {symbol}: {e}")
            except Exception as e:
                log(f"Train error {symbol}: {e}")
        if threshold_candidates:
            try:
                median_thr = float(np.median(threshold_candidates))
                if 0.0 < median_thr < 1.0:
                    lo, hi = self.opt_thr_min, self.opt_thr_max
                    median_thr = max(lo, min(hi, median_thr))
                    self.threshold = float(
                        self.ema_alpha * median_thr + (1.0 - self.ema_alpha) * self.threshold
                    )
                    log(f"[train_models] recalibrated threshold to {self.threshold:.4f}")
                else:
                    log(f"[train_models] invalid median_thr={median_thr:.4f}, keep={self.threshold:.4f}")
            except Exception as e:
                log(f"[train_models] threshold recalibration error: {e}")

        # After training, refresh adaptive parameters to keep TP/SL and threshold up to date.
        try:
            self._auto_update_parameters()
        except Exception:
            pass

    def _optimize_params_if_needed(self) -> None:
        """
        Periodically optimise trading parameters (threshold, TP, SL) using
        parameter_optimizer.search_params().  The optimisation runs only
        when AUTO_OPTIMIZE is enabled and the required module is available.
        It compares the current UTC time with the last optimisation time
        and triggers a new search when more than opt_interval_hours have
        elapsed.  Updated parameters are applied in place on the bot
        instance.  Any exceptions during optimisation are logged and do
        not interrupt trading.
        """
        if not self.auto_optimize:
            return
        # Ensure search_params is available
        if search_params is None:
            return
        now = datetime.now(timezone.utc)
        try:
            elapsed = (now - self.last_opt_time).total_seconds() / 3600.0
        except Exception:
            # On first run or error, trigger optimisation immediately
            elapsed = self.opt_interval_hours + 1.0
        if elapsed < self.opt_interval_hours:
            return
        try:
            log(f"[auto-opt] Running parameter optimisation after {elapsed:.2f}h")
            params = search_params()
            if isinstance(params, dict):
                thr = float(params.get("threshold", self.threshold))
                tp = float(params.get("tp", self.tp))
                sl = float(params.get("sl", self.sl))
                # Clamp threshold to configured bounds
                if 0.0 < thr < 1.0:
                    self.threshold = thr
                if tp > 0.0:
                    self.tp = tp
                if sl > 0.0:
                    self.sl = sl
                log(f"[auto-opt] Updated params: threshold={self.threshold:.4f}, tp={self.tp:.4f}, sl={self.sl:.4f}")
        except Exception as e:
            log(f"[auto-opt] optimisation error: {e}")
        # Reset timer regardless of success to avoid rapid reattempts
        self.last_opt_time = now

    def _pyramid_positions(self) -> None:
        """
        Check open positions and add to them when the price has moved
        sufficiently in the trade's favour.  Pyramiding is controlled by
        PYRAMID_ENABLED, PYRAMID_STEP_PCT and PYRAMID_MAX_ADDS.  Each time
        the current price exceeds the last add price by the configured
        step percentage and the maximum number of adds has not been
        reached, a new market buy order is placed.  The size of each
        additional buy defaults to the base position size recorded when
        the position was opened.  TP and SL levels are adjusted
        proportionally relative to the new entry price.
        """
        if not self.pyramid_enabled:
            return
        for symbol, pos in list(self.positions.items()):
            # Only manage open positions
            if not isinstance(pos, dict) or pos.get("status") != "open":
                continue
            # Respect maximum number of adds
            adds_done = int(pos.get("pyramid_adds", 0))
            if adds_done >= self.pyramid_max_adds:
                continue
            try:
                last_add_price = float(pos.get("last_add_price", pos.get("entry", 0.0)))
            except Exception:
                last_add_price = float(pos.get("entry", 0.0))
            # Calculate target price for next add
            step_pct = self.pyramid_step_pct
            target_price = last_add_price * (1.0 + step_pct)
            # Fetch current market price
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float((ticker.get("last") or ticker.get("close") or 0.0))
            except Exception:
                price = 0.0
            if price <= 0.0:
                continue
            if price < target_price:
                continue
            # Determine new capital size based on the base position.  Use
            # pyramid_base if available, otherwise fall back to current
            # relative_position.
            try:
                base_cap = float(pos.get("pyramid_base", pos.get("relative_position", 0.0)))
            except Exception:
                base_cap = float(pos.get("relative_position", 0.0))
            if base_cap <= 0.0:
                continue
            # Enforce risk cap for this additional trade
            try:
                equity = self.capital_manager.get_available_capital(self.exchange)
                risk_cap = equity * float(self.risk_pct)
                if base_cap > risk_cap:
                    base_cap = risk_cap
            except Exception:
                pass
            # Determine buy price including spread and fee
            try:
                fee_rate = float(os.getenv("TAKER_FEE", "0.0"))
            except Exception:
                fee_rate = 0.0
            try:
                spread_pct = float(os.getenv("SPREAD_PCT", "0.0"))
            except Exception:
                spread_pct = 0.0
            buy_price = price * (1.0 + spread_pct + fee_rate)
            # Compute amount to buy
            try:
                amount = base_cap / buy_price
            except Exception:
                amount = 0.0
            # Ensure minimum trade size
            if amount < self.min_base_amount or base_cap < self.min_notional:
                continue
            # Place market buy order
            try:
                self.exchange.create_market_buy_order(symbol, amount)
            except Exception as e:
                log(f"[pyramid] buy error for {symbol}: {e}")
                continue
            # Track the base amount so closing sells the full position.
            try:
                pos["amount"] = float(pos.get("amount", 0.0)) + float(amount)
            except Exception:
                pass
            # Update capital manager exposure
            try:
                self.capital_manager.update_after_trade(symbol, base_cap)
            except Exception:
                pass
            # Increase the position's relative position
            try:
                pos["relative_position"] = float(pos.get("relative_position", 0.0)) + base_cap
            except Exception:
                pass
            # Increment add counter and update last_add_price
            pos["pyramid_adds"] = adds_done + 1
            pos["last_add_price"] = price
            # Recalculate TP and SL relative to new price using original percentages
            try:
                entry = float(pos.get("entry", 0.0))
                tp_old = float(pos.get("tp", 0.0))
                sl_old = float(pos.get("sl", 0.0))
                # avoid division by zero
                tp_pct = (tp_old - entry) / entry if entry > 0 else 0.0
                sl_pct = (entry - sl_old) / entry if entry > 0 else 0.0
                # apply same percentages to current price
                pos["tp"] = price * (1.0 + tp_pct)
                pos["sl"] = price * (1.0 - sl_pct)
            except Exception:
                pass
            log(f"[pyramid] added to {symbol} at {price:.4f}; adds={pos['pyramid_adds']} total cap={pos['relative_position']:.2f}")
    def predict(self, symbol, df_10m, df_1h, df_4h):
        models = self.model_map.get(symbol)
        scaler = self.scaler_map.get(symbol)
        if models is None or scaler is None:
            return 0.5
        try:
            df = df_10m.copy().dropna()
            # Align higher timeframe indicators using forward fill rather than "nearest"
            # to avoid peeking into future candles.  Using nearest or backfill can
            # inadvertently leak future information when the higher timeframe
            # resamples to a later timestamp.  With forward fill, the most
            # recently available value is used until a new bar arrives.
            df["MACD_1h"] = df_1h["macd"].reindex(df.index, method="ffill")
            df["RSI_4h"] = df_4h["rsi_14"].reindex(df.index, method="ffill")
            features = df[get_feature_columns()].dropna().values
            # require at least seq_len+1 rows; otherwise return neutral probability
            if len(features) < (self.seq_len + 1):
                return 0.5
            scaled = scaler.transform(features)
            preds: List[float] = []
            for mtype, mdl in models.items():
                try:
                    # Determine input shape based on the actual model type (Keras vs sklearn).
                    if tf is not None and isinstance(mdl, tf.keras.Model):
                        # For Keras models we need a 3D tensor: (1, timesteps, features)
                        X_input = np.expand_dims(scaled[-self.seq_len:], axis=0)
                        p = float(mdl.predict(X_input, verbose=0)[0][0])
                    else:
                        # For sklearn models we pass a flattened 2D array
                        X_input = scaled[-self.seq_len:].reshape(1, -1)
                        # Some sklearn models return a 1D array; others return 2D
                        raw_pred = mdl.predict(X_input)
                        # Use proba if available; otherwise raw prediction
                        if hasattr(mdl, "predict_proba"):
                            # Use probability for class 1
                            p = float(mdl.predict_proba(X_input)[0][1])
                        else:
                            p = float(raw_pred[0])
                    preds.append(p)
                except Exception:
                    try:
                        if tf is not None and isinstance(mdl, tf.keras.Model):
                            X_input = np.expand_dims(scaled[-self.seq_len:], axis=0)
                            p = float(mdl.predict(X_input)[0][0])
                        else:
                            X_input = scaled[-self.seq_len:].reshape(1, -1)
                            if hasattr(mdl, "predict_proba"):
                                p = float(mdl.predict_proba(X_input)[0][1])
                            else:
                                p = float(mdl.predict(X_input)[0])
                        preds.append(p)
                    except Exception as e:
                        log(f"[predict] model {mtype} error for {symbol}: {e}")
                        continue
            if not preds:
                return 0.5
            avg_raw = float(sum(preds) / len(preds))
            calibrator = self.calibrator_map.get(symbol)
            if calibrator is not None:
                try:
                    return float(calibrator.predict_proba(np.array([[avg_raw]]))[0][1])
                except Exception:
                    return avg_raw
            return avg_raw
        except Exception as e:
            log(f"Prediction error for {symbol}: {e}")
            return 0.5

    def trade(self):
        """
        Scan for trade signals, open new positions subject to risk management rules,
        and maintain existing positions.  This implementation introduces several
        enhancements over the original version:

        * Uses a logistic function to scale position size based on how far the
          model confidence is above a dynamic threshold.  This results in
          smoother size adjustments compared to linear scaling and avoids tiny
          allocations on marginal signals.
        * Incorporates transaction costs and spread when determining the buy
          price so that position sizing reflects the effective cost.
        * Ensures each new position meets minimum notional and base asset
          requirements before placing an order.
        * Applies a cooldown period after closing a position during which
          re‑entering the same symbol is disallowed.  This prevents the bot
          from rapidly entering and exiting in choppy markets.
        * Logs the current unrealised PnL at the end of each cycle.
        """

        # Update adaptive TP/SL/threshold before trading
        try:
            self._auto_update_parameters()
        except Exception:
            pass
        # Run periodic parameter optimisation if enabled.  This may update
        # self.threshold, self.tp and self.sl based on recent backtesting.
        try:
            self._optimize_params_if_needed()
        except Exception:
            pass

        # Sync unrecorded positions and update local positions from manager
        try:
            self.sync_unrecorded_positions()
        except Exception as e:
            log(f"[sync_unrecorded] error: {e}")
        try:
            self._sync_positions_from_manager()
        except Exception as e:
            log(f"[sync positions error] {e}")

        # -----------------------------------------------------------------
        # Throttle entry scans (opening new positions) so we don't open trades
        # every minute. Position management (TP/SL, trailing, etc.) still runs
        # every cycle.
        # -----------------------------------------------------------------
        do_entries: bool = True
        try:
            interval_min = 0.0
            try:
                interval_min = float(os.getenv("ENTRY_INTERVAL_MINUTES", "0"))
            except Exception:
                interval_min = 0.0
            if interval_min <= 0.0:
                try:
                    interval_min = float(getattr(self, "config", {}).get("trading_interval_minutes", 15))
                except Exception:
                    interval_min = 15.0
            now_ts = datetime.now(timezone.utc)
            last_scan = getattr(self, "_last_entry_scan", None)
            if last_scan is not None and interval_min > 0.0:
                try:
                    elapsed_min = (now_ts - last_scan).total_seconds() / 60.0
                except Exception:
                    elapsed_min = 0.0
                if elapsed_min < interval_min:
                    do_entries = False
                    log(f"[entries] skip scan: {elapsed_min:.1f}m/{interval_min:.0f}m")
            if do_entries:
                self._last_entry_scan = now_ts
        except Exception:
            do_entries = True

        predictions: Dict[str, tuple[pd.DataFrame, float]] = {}
        new_trades: List[str] = []

        # Generate predictions for each symbol and filter by volatility and trend
        if do_entries:
            for symbol in self.symbols:
                try:
                    df_10m, df_1h, df_4h = self.fetch_data(symbol)
                    # compute volatility over the lookback window
                    try:
                        max_high = float(df_10m["high"].max())
                        min_low = float(df_10m["low"].min())
                        last_close = float(df_10m["close"].iloc[-1])
                        volatility = (max_high - min_low) / last_close if last_close > 0 else 0.0
                    except Exception:
                        volatility = 0.0
                    if volatility < self.min_volatility:
                        log(f"[skip] {symbol}: volatility={volatility:.4f} < min_volatility={self.min_volatility:.4f}")
                        continue
                    # Apply optional EMA trend filter: only trade long when price > EMA
                    if self.ema_filter_enabled:
                        ema_col = f"ema_{self.ema_period}"
                        try:
                            last_ema = float(df_10m.get(ema_col, pd.Series([0.0])).iloc[-1])
                        except Exception:
                            last_ema = 0.0
                        try:
                            price_now = float(df_10m["close"].iloc[-1])
                        except Exception:
                            price_now = 0.0
                        # If the price is below the EMA, skip this symbol.  This helps
                        # avoid entering trades in bearish or sideways markets.
                        if last_ema > 0.0 and price_now < last_ema:
                            log(f"[skip] {symbol}: price {price_now:.4f} < EMA({self.ema_period}) {last_ema:.4f}")
                            continue
                    conf = self.predict(symbol, df_10m, df_1h, df_4h)
                    predictions[symbol] = (df_10m, conf)
                except Exception as e:
                    log(f"Prediction error: {e}")

        # Optionally log top N predictions for debugging
        try:
            for sym, (_df_top, p_top) in sorted(predictions.items(), key=lambda x: x[1][1], reverse=True)[:5]:
                log(f"{sym} prediction: {p_top:.4f}")
        except Exception:
            pass

        # Compute a dynamic base threshold using a sliding window of past predictions
        try:
            for _, (_, c_hist) in predictions.items():
                try:
                    self.pred_history.append(float(c_hist))
                except Exception:
                    self.pred_history.append(0.0)
            if len(self.pred_history) > self.dynamic_threshold_window:
                self.pred_history = self.pred_history[-self.dynamic_threshold_window:]
            if len(self.pred_history) >= self.dynamic_threshold_window:
                dyn_base = float(np.percentile(self.pred_history, self.dynamic_threshold_percentile))
                dyn_base = max(self.opt_thr_min, min(dyn_base, self.opt_thr_max))
            else:
                dyn_base = self.threshold
        except Exception:
            dyn_base = self.threshold
        sentiment = self._get_news_sentiment()
        adjusted_threshold = dyn_base * (1.0 - self.sentiment_weight * sentiment)

        # Enforce maximum number of open positions
        try:
            open_count = sum(1 for p in self.positions.values() if isinstance(p, dict) and p.get("status") == "open")
            max_positions_cfg = getattr(self.capital_manager.cfg, 'max_positions', 5)
            if open_count >= max_positions_cfg:
                log(f"[skip] maximum open positions reached: {open_count}/{max_positions_cfg}")
                predictions = {}
        except Exception:
            pass

        for symbol, (df, conf) in predictions.items():
            # Skip if there is already an open position for this symbol
            pos = self.positions.get(symbol)
            if pos and pos.get("status") != "closed":
                continue
            # Skip if within cooldown period after closing a position
            try:
                last_closed = self.last_closed_time.get(symbol)
                if last_closed is not None:
                    elapsed = (time.time() - last_closed) / 60.0
                    if elapsed < self.cooldown_minutes:
                        log(f"[skip] {symbol}: cooldown {elapsed:.1f}m/{self.cooldown_minutes}m")
                        continue
            except Exception:
                pass
            # Trend filter using ADX with optional bypass margin
            try:
                last_adx = float(df.get('adx', pd.Series([0])).iloc[-1])
            except Exception:
                last_adx = 0.0
            try:
                bypass_margin = float(os.getenv("ADX_BYPASS_MARGIN", "0.05"))
            except Exception:
                bypass_margin = 0.05
            if last_adx < self.adx_threshold and conf < (adjusted_threshold + bypass_margin):
                log(f"[skip] {symbol}: adx={last_adx:.2f} < adx_threshold={self.adx_threshold:.2f}")
                continue
            if conf < adjusted_threshold:
                log(f"[skip] {symbol}: conf={conf:.4f} < threshold={adjusted_threshold:.4f}")
                continue
            # Compute base capital allocation
            try:
                cap = self.capital_manager.size_quote(
                    exchange=self.exchange,
                    symbol=symbol,
                    probability=conf,
                    tp=self.tp,
                    sl=self.sl,
                    aggressiveness=self.aggressiveness,
                )
            except Exception as e:
                log(f"[capital] error sizing {symbol}: {e}")
                cap = 0.0
            # Apply logistic scaling to cap based on how far the confidence is above the threshold
            try:
                gap = float(conf - adjusted_threshold)
                # logistic function yields values between 0 and 1; steepness controlled by LOGISTIC_FACTOR
                size_factor = 1.0 / (1.0 + math.exp(-self.logistic_factor * gap))
                size_factor = max(self.min_size_factor, min(size_factor, 1.0))
                cap *= size_factor
            except Exception:
                pass
            # Further scale position size based on recent volatility (ATR).  Higher
            # volatility results in smaller position sizes.  We normalise the
            # ATR relative to the latest close to obtain a dimensionless
            # vol_norm.  Dividing by (1 + vol_norm) reduces cap when vol_norm
            # is large and has little effect when vol_norm is small.
            try:
                atr_val = 0.0
                if isinstance(df, pd.DataFrame):
                    if "atr" in df.columns:
                        atr_val = float(df["atr"].iloc[-1])
                    elif "atr_14" in df.columns:
                        atr_val = float(df["atr_14"].iloc[-1])
                    elif "atr_28" in df.columns:
                        atr_val = float(df["atr_28"].iloc[-1])
                last_close = float(df['close'].iloc[-1])
                if atr_val > 0.0 and last_close > 0.0:
                    vol_norm = atr_val / last_close
                    cap = cap / (1.0 + vol_norm)
            except Exception:
                pass
            # Enforce minimum notional allocation
            try:
                env_min_alloc = float(os.getenv("MIN_QUOTE_ALLOC", "10"))
            except Exception:
                env_min_alloc = 0.0
            min_alloc = max(env_min_alloc, 5.0)
            # Cap by risk percentage of available capital
            try:
                equity = self.capital_manager.get_available_capital(self.exchange)
                if equity > 0.0:
                    risk_cap = equity * float(self.risk_pct)
                    if cap > risk_cap:
                        cap = risk_cap
            except Exception:
                pass
            # If after all caps the position is smaller than MIN_QUOTE_ALLOC, skip (don't force it).
            if cap < min_alloc:
                log(f"[skip] {symbol}: cap={cap:.2f} < min_alloc={min_alloc:.2f}")
                continue
            if cap <= 0:
                log(f"[skip] {symbol}: cap={cap:.2f} below minimum")
                continue
            # Determine trade size accounting for fees/spread and place the order.
            # NOTE: We record the base amount and use entry_price for TP/SL and later closing.
            atr_value = 0.0
            fill_price = 0.0
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                raw_price = float((ticker.get("last") or ticker.get("close") or 0.0))
                if raw_price <= 0.0:
                    raise ValueError("invalid price")
                fee_rate = float(os.getenv("TAKER_FEE", "0.0"))
                spread_pct = float(os.getenv("SPREAD_PCT", "0.0"))

                # ATR (support both compute_indicators and add_indicators naming)
                try:
                    if isinstance(df, pd.DataFrame):
                        if "atr" in df.columns:
                            atr_value = float(df["atr"].iloc[-1])
                        elif "atr_14" in df.columns:
                            atr_value = float(df["atr_14"].iloc[-1])
                        elif "atr_28" in df.columns:
                            atr_value = float(df["atr_28"].iloc[-1])
                except Exception:
                    atr_value = 0.0

                # Cost-aware filter: skip trades where expected edge doesn't cover fees/spread.
                try:
                    min_expectancy = float(os.getenv("MIN_EXPECTANCY_PCT", "0.0005"))
                except Exception:
                    min_expectancy = 0.0005
                roundtrip_cost = 2.0 * (fee_rate + spread_pct)
                if atr_value > 0.0 and raw_price > 0.0:
                    tp_est_pct = (atr_value * self.atr_tp_mult) / raw_price
                    sl_est_pct = (atr_value * self.atr_sl_mult) / raw_price
                else:
                    tp_est_pct = float(self.tp)
                    sl_est_pct = float(self.sl)
                expectancy = float(conf) * tp_est_pct - (1.0 - float(conf)) * sl_est_pct
                if expectancy < (roundtrip_cost + min_expectancy):
                    log(
                        f"[skip] {symbol}: low net expectancy={expectancy:.4f} < "
                        f"cost+buffer={(roundtrip_cost + min_expectancy):.4f} "
                        f"(conf={conf:.3f}, tp≈{tp_est_pct:.4f}, sl≈{sl_est_pct:.4f})"
                    )
                    continue

                buy_price = raw_price * (1.0 + spread_pct + fee_rate)
                amount = cap / buy_price
                notional = cap
                if amount < self.min_base_amount or notional < self.min_notional:
                    log(f"[skip] {symbol}: amount={amount:.6f}, cap={notional:.2f} below minimum thresholds")
                    continue

                order = self.exchange.create_market_buy_order(symbol, amount)
                fill_price = raw_price
                try:
                    if isinstance(order, dict):
                        fill_price = float(order.get("average") or order.get("price") or raw_price)
                except Exception:
                    fill_price = raw_price
            except Exception as e:
                log(f"[direct trade error] {e}")
                continue

            entry_price = float(fill_price) if fill_price > 0.0 else float(raw_price)

            # Compute TP and SL using ATR or fixed percentages, anchored to entry_price.
            if atr_value > 0.0:
                tp_price = entry_price + (atr_value * self.atr_tp_mult)
                sl_price = entry_price - (atr_value * self.atr_sl_mult)
            else:
                tp_price = entry_price * (1.0 + self.tp)
                sl_price = entry_price * (1.0 - self.sl)
            if self.use_trailing_stop:
                trail_sl = entry_price * (1.0 - self.trail_sl_pct)
                if sl_price < trail_sl:
                    sl_price = trail_sl
            if sl_price < 0.0:
                sl_price = 0.0

            # Record new position (store base amount so closing sells the full position).
            self.positions[symbol] = {
                "entry": entry_price,
                "tp": tp_price,
                "sl": sl_price,
                "relative_position": cap,
                "amount": float(amount),
                "sentiment": "neutral",
                "status": "open",
            }
            # Initialise pyramiding metadata for this position.
            self.positions[symbol]["pyramid_adds"] = 0
            self.positions[symbol]["last_add_price"] = entry_price
            self.positions[symbol]["pyramid_base"] = cap
            if self.use_trailing_stop:
                self.positions[symbol]["max_price"] = entry_price
            try:
                self.capital_manager.update_after_trade(symbol, cap)
            except Exception:
                pass
            self.save_positions()
            log(f"[direct trade] opened {symbol} for {cap:.2f} USDC at price {entry_price:.4f}")
            # Prepare summary for telegram
            summary = (
                f"Сделка: {symbol}\n"
                f"Вход: {entry_price:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}\n"
                f"Увереност: {conf*100:.1f}% | Позиция: {cap:.1f}\n"
                f"Настроение: neutral"
            )
            new_trades.append(summary)

        # Evaluate and close positions if TP or SL hit
        try:
            self.check_positions()
        except Exception as e:
            log(f"[check_positions error] {e}")
        try:
            monitor_positions(self.exchange, self.positions, None)
        except Exception as e:
            log(f"[monitor_positions error] {e}")
        try:
            reconcile_open_positions(self.exchange, self.positions)
        except Exception as e:
            log(f"[reconcile_positions error] {e}")
        # Attempt to pyramid into winning positions if configured
        try:
            self._pyramid_positions()
        except Exception as e:
            log(f"[pyramid error] {e}")
        if new_trades:
            try:
                send_telegram("📈 Нови сделки:\n\n" + "\n\n".join(new_trades))
            except Exception:
                pass

        # Log current unrealised PnL
        try:
            self.report_pnl()
        except Exception:
            pass

        # Persist positions and optionally send update on open positions
        try:
            self.save_positions()
        except Exception:
            pass
        try:
            open_positions: List[str] = []
            snapshot_list: List[str] = []
            for sym, p in self.positions.items():
                if p.get("status") == "open":
                    entry_val = float(p.get('entry', 0.0))
                    tp_val = float(p.get('tp', 0.0))
                    sl_val = float(p.get('sl', 0.0))
                    rel_cap = float(p.get('relative_position', 0.0))
                    open_positions.append(
                        f"{sym}: entry={entry_val:.4f}, tp={tp_val:.4f}, sl={sl_val:.4f}, cap={rel_cap:.2f}"
                    )
                    snapshot_list.append(f"{sym}|{entry_val:.8f}|{tp_val:.8f}|{sl_val:.8f}|{rel_cap:.8f}")
            if open_positions:
                snapshot_tuple = tuple(sorted(snapshot_list))
                if self._last_open_positions_snapshot is None or snapshot_tuple != self._last_open_positions_snapshot:
                    send_telegram("🔔 Отворени позиции:\n" + "\n".join(open_positions))
                    self._last_open_positions_snapshot = snapshot_tuple
            else:
                self._last_open_positions_snapshot = None
        except Exception:
            pass

    def check_positions(self) -> None:
        """
        Iterate over open positions stored in positions.json and close them when current price hits
        the take‑profit (TP) or stop‑loss (SL) thresholds. This method uses the exchange directly
        to place market sell orders, updates the capital manager accordingly, marks the position
        as closed, and sends a Telegram notification summarising closed trades.
        """
        closed_summaries: List[str] = []
        # iterate over a copy of keys to allow modification during iteration
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos or pos.get("status") != "open":
                continue
            try:
                # fetch ticker and derive a robust price using bid/ask/last/close
                ticker = self.exchange.fetch_ticker(symbol)
                last_price = float(ticker.get("last") or 0.0)
                bid_price  = float(ticker.get("bid") or 0.0)
                ask_price  = float(ticker.get("ask") or 0.0)
                close_price= float(ticker.get("close") or 0.0)
                if bid_price > 0.0 and ask_price > 0.0:
                    price = (bid_price + ask_price) / 2.0
                elif last_price > 0.0:
                    price = last_price
                elif close_price > 0.0:
                    price = close_price
                else:
                    price = 0.0
                if price <= 0.0:
                    continue
                tp_price = float(pos.get("tp", 0.0))
                sl_price = float(pos.get("sl", 0.0))
                entry_quote = float(pos.get("relative_position", 0.0))

                # Update trailing stop if enabled. Maintain a per‑position max_price and
                # adjust SL upwards based on the highest observed price. Do not allow
                # SL to decrease. This applies to both newly opened and legacy positions.
                if self.use_trailing_stop:
                    try:
                        max_price = float(pos.get("max_price", 0.0))
                    except Exception:
                        max_price = 0.0
                    # If current price exceeds historical max_price, update it
                    if price > max_price:
                        max_price = price
                        pos["max_price"] = max_price
                    # Calculate a new trailing SL as a percentage below max_price
                    new_sl = max_price * (1.0 - self.trail_sl_pct)
                    # Only update SL if trailing stop moves it higher
                    if new_sl > sl_price:
                        pos["sl"] = new_sl
                        sl_price = new_sl
                    # Note: we intentionally do not call save_positions() here for every
                    # update to reduce I/O. Positions are saved when a position is
                    # closed or when sync_unrecorded_positions() is called.
                # Determine if TP or SL has been hit
                hit_tp = tp_price > 0.0 and price >= tp_price
                hit_sl = sl_price > 0.0 and price <= sl_price
                if hit_tp or hit_sl:
                    # Determine the base amount to close.
                    # IMPORTANT: closing should be based on the *position size* (base qty),
                    # not on entry_quote/current_price. Using current_price sells only a
                    # fraction on profitable trades and leaves leftover "dust" balances.
                    desired_amount = 0.0
                    try:
                        desired_amount = float(pos.get("amount", 0.0))
                    except Exception:
                        desired_amount = 0.0
                    if desired_amount <= 0.0:
                        try:
                            entry_px = float(pos.get("entry", 0.0))
                        except Exception:
                            entry_px = 0.0
                        if entry_px > 0.0:
                            desired_amount = entry_quote / entry_px
                    # fetch current free balance for the base asset
                    try:
                        base = symbol.split('/')[0]
                        bal = self.exchange.fetch_balance() or {}
                        free_bal = float((bal.get('free') or {}).get(base, 0.0))
                    except Exception:
                        free_bal = 0.0
                    if free_bal <= 0.0 or desired_amount <= 0.0:
                        # no free balance or no desired amount; keep position open for next attempt
                        continue
                    # if free balance value is below dust threshold, mark position as closed without selling
                    dust_value = free_bal * price
                    try:
                        threshold = self.dust_threshold
                    except AttributeError:
                        threshold = 0.0
                    if dust_value < threshold:
                        # update capital exposure using the original quote amount
                        try:
                            self.capital_manager.update_after_trade(symbol, -entry_quote)
                        except Exception:
                            pass
                        # mark position as closed due to dust
                        pos["status"] = "closed"
                        # record the last closed time for cooldown
                        self.last_closed_time[symbol] = time.time()
                        reason = "dust"
                        summary = (
                            f"Затворена позиция ({reason}): {symbol}\n"
                            f"Изход: {price:.4f} | Вход: {pos.get('entry', 0):.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}\n"
                            f"Резултат: прах"
                        )
                        closed_summaries.append(summary)
                        # save updated positions to disk
                        self.save_positions()
                        continue
                    target_amount = min(desired_amount, free_bal)
                    # Before attempting to sell, check if the amount and notional meet minimum thresholds
                    # to avoid order failures due to exchange filters.  If either the base amount or
                    # the notional value is too small, close the position directly and classify
                    # it as dust.
                    try:
                        target_notional = target_amount * price
                    except Exception:
                        target_notional = 0.0
                    # If the base amount or notional is too small, close the position to avoid
                    # repeated order failures.  Treat it as dust and record the time of closure.
                    if target_amount < self.min_base_amount or target_notional < self.min_notional:
                        # update capital exposure using the original quote amount
                        try:
                            self.capital_manager.update_after_trade(symbol, -entry_quote)
                        except Exception:
                            pass
                        pos["status"] = "closed"
                        # record last closed time for cooldown
                        self.last_closed_time[symbol] = time.time()
                        reason = "dust"
                        summary = (
                            f"Затворена позиция ({reason}): {symbol}\n"
                            f"Изход: {price:.4f} | Вход: {pos.get('entry', 0):.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}\n"
                            f"Резултат: прах"
                        )
                        closed_summaries.append(summary)
                        # save updated positions to disk
                        self.save_positions()
                        continue
                    sold = False
                    # attempt to sell, reducing the amount slightly if insufficient balance error occurs
                    for _ in range(3):
                        if target_amount <= 0.0:
                            break
                        try:
                            self.exchange.create_market_sell_order(symbol, target_amount)
                            sold = True
                            break
                        except Exception as e:
                            # reduce amount by 1 % and retry
                            log(f"[sell error] {e} — retrying with smaller amount")
                            target_amount *= 0.99
                    if not sold:
                        # could not sell after retries; mark as closed to avoid endless retry.
                        try:
                            # update capital exposure using the original quote amount
                            self.capital_manager.update_after_trade(symbol, -entry_quote)
                        except Exception:
                            pass
                        pos["status"] = "closed"
                        # record last closed time for cooldown
                        self.last_closed_time[symbol] = time.time()
                        reason = "error"
                        summary = (
                            f"Затворена позиция ({reason}): {symbol}\n"
                            f"Изход: {price:.4f} | Вход: {pos.get('entry', 0):.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}\n"
                            f"Резултат: грешка"
                        )
                        closed_summaries.append(summary)
                        # save updated positions to disk
                        self.save_positions()
                        continue
                    # update capital exposure using the original quote amount
                    try:
                        self.capital_manager.update_after_trade(symbol, -entry_quote)
                    except Exception:
                        pass
                    # mark position as closed
                    pos["status"] = "closed"
                    # record last closed time for cooldown
                    self.last_closed_time[symbol] = time.time()
                    reason = "TP" if hit_tp else "SL"
                    summary = (
                        f"Затворена позиция ({reason}): {symbol}\n"
                        f"Изход: {price:.4f} | Вход: {pos.get('entry', 0):.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}\n"
                        f"Резултат: {'печалба' if hit_tp else 'загуба'}"
                    )
                    closed_summaries.append(summary)
                    # save updated positions to disk
                    self.save_positions()
            except Exception as e:
                log(f"[check_positions error] {e}")
        if closed_summaries:
            try:
                send_telegram("📉 Затворени позиции:\n\n" + "\n\n".join(closed_summaries))
            except Exception:
                pass

    def sync_unrecorded_positions(self) -> None:
        """
        Scan account balances and record any free base assets that are not already
        tracked in ``self.positions``. If a free balance exists for a symbol, a new
        position is recorded using the current market price and configured TP/SL
        percentages. To avoid repeatedly opening positions for leftover ``dust``
        (very small balances), the free amount is only considered when its value
        (free_amt * price) exceeds the configured ``dust_threshold``. This prevents
        tiny residual balances from being treated as open positions.
        """
        try:
            bal = self.exchange.fetch_balance() or {}
        except Exception as e:
            log(f"[sync_unrecorded] balance error: {e}")
            return
        free: Dict[str, float] = {}
        try:
            free = bal.get("free") or {}
        except Exception:
            free = {}
        # Dust threshold below which we ignore balances
        try:
            dust_threshold = self.dust_threshold
        except AttributeError:
            dust_threshold = 0.0

        updated = False
        # --- Handle manual sells ---
        # For every open position, if there is no corresponding free balance or its
        # value is below the dust threshold, mark the position as closed. This
        # captures positions that have been closed manually outside the bot.
        for symbol, pos in list(self.positions.items()):
            if not pos or pos.get("status") != "open":
                continue
            try:
                base = symbol.split("/")[0]
            except Exception:
                continue
            free_amt = 0.0
            try:
                free_amt = float(free.get(base, 0.0))
            except Exception:
                pass
            # Fetch market price once per symbol
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0.0)
            except Exception:
                price = 0.0
            dust_value = free_amt * price
            if free_amt <= 0.0 or dust_value < dust_threshold:
                # consider as manually sold; close position
                pos["status"] = "closed"
                updated = True
                # Enforce cooldown after a close to prevent immediate re-entry loops
                try:
                    self.last_closed_time[symbol] = time.time()
                except Exception:
                    pass
                try:
                    entry_quote = float(pos.get("relative_position", 0.0))
                    self.capital_manager.update_after_trade(symbol, -entry_quote)
                except Exception:
                    pass
                try:
                    summary = (
                        f"Затворена позиция (manual sell): {symbol}\n"
                        f"Изход: {price:.4f} | Вход: {pos.get('entry', 0):.4f} | TP: {pos.get('tp', 0):.4f} | SL: {pos.get('sl', 0):.4f}\n"
                        f"Резултат: ръчно затворена"
                    )
                    send_telegram(summary)
                except Exception:
                    pass

        # --- Handle manual buys ---
        for symbol in self.symbols:
            try:
                base, quote = symbol.split('/')
            except Exception:
                continue
            try:
                free_amt = float(free.get(base, 0.0))
            except Exception:
                free_amt = 0.0
            if free_amt <= 0.0:
                continue
            # skip if we already track this symbol with an open position
            if symbol in self.positions and self.positions[symbol].get("status") == "open":
                continue
            # fetch current market price
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0.0)
            except Exception:
                price = 0.0
            if price <= 0.0:
                continue
            # compute quote value of free amount
            quote_val = free_amt * price
            # ignore extremely small values to avoid recording dust as new positions
            if quote_val < dust_threshold:
                continue
            # compute TP and SL
            tp_price = price * (1 + self.tp)
            sl_price = price * (1 - self.sl)
            if self.use_trailing_stop:
                # ensure initial SL is not smaller than trailing distance
                trail_sl = price * (1 - self.trail_sl_pct)
                sl_price = max(sl_price, trail_sl)
            # record the position
            self.positions[symbol] = {
                "entry": price,
                "tp": tp_price,
                "sl": sl_price,
                "relative_position": quote_val,
                "amount": free_amt,
                "sentiment": "neutral",
                "status": "open",
            }
            if self.use_trailing_stop:
                self.positions[symbol]["max_price"] = price
            updated = True
            try:
                send_telegram(f"🪙 Добавена ръчна позиция {symbol}: amount={free_amt:.6f} price={price:.4f}")
            except Exception:
                pass
        if updated:
            self.save_positions()

    def report_pnl(self) -> None:
        """
        Compute and log the unrealised profit/loss of all open positions.

        For each open position the bot calculates the difference between
        the current market price and the entry price and multiplies it by
        the position's base amount (derived from the relative_position in
        quote currency divided by the entry price).  The sum of these
        values across all open positions yields the current unrealised
        PnL.  If price data is unavailable the entry price is used as a
        fallback.  The currency for reporting is taken from the capital
        manager configuration if available.
        """
        total_pnl: float = 0.0
        for sym, pos in self.positions.items():
            if not pos or pos.get("status") != "open":
                continue
            try:
                entry = float(pos.get("entry", 0.0))
                if entry <= 0.0:
                    continue
                quote_alloc = float(pos.get("relative_position", 0.0))
                # Prefer stored base amount if available (more accurate, especially with pyramiding)
                try:
                    base_amount = float(pos.get("amount", 0.0))
                except Exception:
                    base_amount = 0.0
                if base_amount <= 0.0:
                    base_amount = quote_alloc / entry if entry > 0 else 0.0
                ticker = self.exchange.fetch_ticker(sym)
                price = float(ticker.get("last", entry))
                total_pnl += (price - entry) * base_amount
            except Exception:
                continue
        try:
            ccy = self.capital_manager.cfg.quote_ccy  # type: ignore[attr-defined]
        except Exception:
            ccy = ""
        log(f"[PnL] Current unrealised PnL: {total_pnl:.2f} {ccy}")

    def run_cycle(self):
        """
        Изпълнява една итерация: прогнозира, търгува, проверява позиции.
        """
        now = datetime.now(timezone.utc)
        if (now - self.last_train).total_seconds() > 6 * 3600:
            selection = evaluate_models()
            adapt = adaptive_parameters()
            self.selected_model_name = selection["model"]
            self.threshold = float(adapt["threshold"])
            self.tp = float(adapt["tp"])
            self.sl = float(adapt["sl"])
            self.aggressiveness = float(adapt["aggressiveness"])
            thr_env = os.getenv("THRESHOLD")
            if thr_env:
                try:
                    v = float(thr_env)
                    if 0.0 < v < 1.0:
                        self.threshold = v
                except Exception:
                    pass
            try:
                self.opt_thr_min = float(os.getenv("OPT_THR_MIN", str(self.opt_thr_min)))
                self.opt_thr_max = float(os.getenv("OPT_THR_MAX", str(self.opt_thr_max)))
                if self.opt_thr_min >= self.opt_thr_max:
                    raise ValueError
            except Exception:
                self.opt_thr_min, self.opt_thr_max = 0.40, 0.70
            try:
                self.ema_alpha = float(os.getenv("EMA_ALPHA", str(self.ema_alpha)))
                if not (0.0 < self.ema_alpha < 1.0):
                    raise ValueError
            except Exception:
                self.ema_alpha = 0.30
            self._load_best_params_if_any()
            try:
                send_telegram(f"🤖 Нов модел избран: {self.selected_model_name}")
            except Exception:
                pass
            self.train_models()
            try:
                send_telegram("🧠 Моделите са обновени.")
            except Exception:
                pass
            self.last_train = now
            self.last_model_selection = now
        self.trade()
        send_periodic_report()

    def _sync_positions_from_manager(self):
        """Fallback метод за синхронизация, когато trade_manager липсва."""
        return
