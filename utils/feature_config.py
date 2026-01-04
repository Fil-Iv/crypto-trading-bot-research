from __future__ import annotations
from typing import List

# Единствен източник на истина за фийчърите.
# Трябва да съвпадат с колоните, които създаваме в add_indicators.add_indicators().

_BASE_FEATURES: List[str] = [
    "close",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_9",
    "ema_21",
    "ema_20",
    "ema_50",
    "sma_20",
    "sma_200",
    "ema_diff",
    "ema_diff_lag1",
    "ema_diff_lag2",
    "ema_diff_lag3",
    "rsi_14_lag1",
    "rsi_14_lag2",
    "rsi_14_lag3",
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_1_lag1",
    "ret_1_lag2",
    "ret_1_lag3",
    "mom_10",
    "vol_20",
    "atr_14",
    "volume_change",
    "price_change_1h",
    "price_change_4h",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_width",
    "volatility_ratio",
    "willr",
    "stoch_k",
    "stoch_d"
    ,
    # Допълнителни EMA
    "ema_100",
    "ema_150",
    # Допълнителен SMA
    "sma_50",
    # Допълнителни RSI
    "rsi_7",
    "rsi_28",
    # Допълнителен ATR за по-дълъг период
    "atr_28",
    # On-Balance Volume
    "obv",
    # Отношение на EMA 9 и EMA 21
    "ema_ratio"

    # --- Нови краткосрочни признаци ---
    ,"ret_3"
    ,"ret_7"
    # --- Диапазонни признаци ---
    ,"high_10d"
    ,"low_10d"
    ,"pos_in_10d_range"
    ,"pct_from_high_10"
    ,"pct_from_low_10"
    ,"pct_from_high_50"
    ,"pct_from_low_50"
]

def get_feature_columns() -> List[str]:
    # Връщаме копие, за да не се модифицира оригиналният списък
    return list(_BASE_FEATURES)
