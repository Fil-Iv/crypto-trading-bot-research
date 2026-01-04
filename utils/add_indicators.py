from __future__ import annotations
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of indicators to OHLCV DataFrame."""

    if df.empty:
        return df

    # 🔧 FIX: махаме двойно "timestamp", ако съществува като колона и индекс
    if "timestamp" in df.columns and df.index.name == "timestamp":
        del df["timestamp"]
        df.index.name = None
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.index.name = None
    elif df.index.name == "timestamp":
        df.index.name = None

    # --- Основни индикатори ---
    for p in (9, 21, 50):
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    # Допълнителни EMA за по-дълги трендове
    for p in (100, 150):
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    for p in (20, 200):
        df[f"sma_{p}"] = df["close"].rolling(window=p).mean()

    # Средна за 50 периода (полезна за по-краткосрочен тренд)
    df["sma_50"] = df["close"].rolling(window=50).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # RSI за кратък и дълъг хоризонт (7 и 28 периода)
    for p in (7, 28):
        g = delta.where(delta > 0, 0).rolling(window=p).mean()
        l = -delta.where(delta < 0, 0).rolling(window=p).mean()
        r = g / l.replace(0, 1)
        df[f"rsi_{p}"] = 100 - (100 / (1 + r))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df["atr_14"] = tr.rolling(window=14).mean()

    # ATR за по-дълъг период (28 периода)
    df["atr_28"] = tr.rolling(window=28).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Отношение между кратка и дълга EMA като допълнителен трендов индикатор
    # (използва вече изчислени ema_9 и ema_21)
    try:
        df["ema_ratio"] = df["ema_9"] / df["ema_21"].replace(0, 1)
    except Exception:
        df["ema_ratio"] = np.nan

    # --- ML фийчъри за prepare_data() ---
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    # Допълнителни краткосрочни промени в цената. Те помагат на модела да усети
    # "моментум" на по-къси интервали (3 и 7 периода). Ако цената се покачва
    # последователно, тези стойности ще са положителни и могат да сигнализират
    # за начален импулс.
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_7"] = df["close"].pct_change(7)
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["mom_10"] = df["close"] - df["close"].shift(10)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_diff"] = df["ema_9"] - df["ema_21"]

    # ---------------------------------------------------------------------
    # Допълнителни признаци за диапазон на цената
    #
    # В някои стратегии е полезно да знаем къде се намира текущата цена спрямо
    # най-високата и най-ниската стойност за последните N периода. Това показва
    # дали сме близо до локален връх или дъно. Включваме 10‑дневен и 50‑дневен
    # диапазон, както и процентната дистанция от тези екстреми.
    high_10 = df["high"].rolling(window=10).max()
    low_10 = df["low"].rolling(window=10).min()
    df["high_10d"] = high_10
    df["low_10d"] = low_10
    # Позиция в 10‑дневния диапазон (0 = на дъното, 1 = на върха)
    df["pos_in_10d_range"] = (df["close"] - low_10) / (high_10 - low_10 + 1e-9)
    # Процентно отстояние от 10‑дневния връх/дъно
    df["pct_from_high_10"] = (high_10 - df["close"]) / (high_10 + 1e-9)
    df["pct_from_low_10"] = (df["close"] - low_10) / (low_10 + 1e-9)
    # 50‑дневен диапазон за по-дълъг контекст
    high_50 = df["high"].rolling(window=50).max()
    low_50 = df["low"].rolling(window=50).min()
    df["pct_from_high_50"] = (high_50 - df["close"]) / (high_50 + 1e-9)
    df["pct_from_low_50"] = (df["close"] - low_50) / (low_50 + 1e-9)

    # Lag фийчъри
    for i in [1, 2, 3]:
        df[f"ret_1_lag{i}"] = df["ret_1"].shift(i)
        df[f"rsi_14_lag{i}"] = df["rsi_14"].shift(i)
        df[f"ema_diff_lag{i}"] = df["ema_diff"].shift(i)

    # Volume change
    df["volume_change"] = df["volume"].pct_change()

    # On-Balance Volume (OBV) – акумулира обема според посоката на движението
    try:
        # Използваме np.sign за да определим посоката на промяната
        direction = np.sign(df["close"].diff()).fillna(0)
        df["obv"] = (direction * df["volume"]).cumsum()
    except Exception:
        df["obv"] = 0.0

    # Price change (1h, 4h)
    df["price_change_1h"] = df["close"].pct_change(periods=1)
    df["price_change_4h"] = df["close"].pct_change(periods=4)

    # Bollinger Bands
    df["bollinger_upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["bollinger_lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    df["bollinger_width"] = df["bollinger_upper"] - df["bollinger_lower"]

    # Volatility ratio
    df["volatility_ratio"] = df["atr_14"] / df["close"]

    # Williams %R
    highest_high = df["high"].rolling(window=14).max()
    lowest_low = df["low"].rolling(window=14).min()
    df["willr"] = -100 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-9)

    # Stochastic Oscillator
    df["stoch_k"] = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low + 1e-9))
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ---------------------------------------------------------------------
    # ADX (Average Directional Index)
    #
    # Compute trend strength via the ta library. If unavailable or an error occurs,
    # fall back to a nominal value so that the bot doesn't skip all trades.
    try:
        import ta  # type: ignore
        adx = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx.fillna(0.0)
        # If ADX is entirely zeros (e.g. due to missing variation), set to nominal minimum.
        try:
            if (df["adx"] == 0).all():
                df["adx"] = 15.0
        except Exception:
            pass
    except Exception:
        df["adx"] = 15.0

    return df
