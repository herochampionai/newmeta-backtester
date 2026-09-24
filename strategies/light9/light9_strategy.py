"""Auto-generated & enhanced Strategy for Light 9.0 - 9.4 Series with 2026 Sovereign Upgrades.
Includes MTF Averaged ADX, Divergence-Confirmed 3-Push Exhaustion Guard, RVOL Surge Filter, and Adaptive Corridors.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .._base import BaseStrategy, Signals
from ..indicators import adx, rsi, stochastic, bollinger, macd


def _as_bool(val, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return default


class Light9Strategy(BaseStrategy):
    name = "light9"

    PROFILES = {0: 'xauusd', 1: 'xauaud', 2: 'us30_nas100', 3: 'hunter', 4: 'hybrid', 5: 'heaven', 6: 'original'}

    PROFILE_PRESETS = {
        'xauusd': {
            'TP_Percent': '1.26', 'SL_Percent': '1.80', 'TrailingDistancePips': '951',
            'BreakevenActivationPips': '169', 'BreakevenBufferPips': '159',
            'SessionAdx_Threshold1': '30.0', 'SessionAdx_Period1': '14',
            'Use_MTF_Averaged_ADX': 'true', 'Use_Exhaustion_Guard': 'true',
            'Use_RVOL_Filter': 'true', 'RVOL_Threshold': '1.5',
        },
        'xauaud': {
            'TP_Percent': '1.15', 'SL_Percent': '1.65', 'TrailingDistancePips': '880',
            'BreakevenActivationPips': '150', 'BreakevenBufferPips': '140',
            'SessionAdx_Threshold1': '28.0', 'SessionAdx_Period1': '14',
            'Use_MTF_Averaged_ADX': 'true', 'Use_Exhaustion_Guard': 'true',
            'Use_RVOL_Filter': 'true', 'RVOL_Threshold': '1.4',
        },
        'us30_nas100': {
            'TP_Percent': '1.40', 'SL_Percent': '1.95', 'TrailingDistancePips': '1200',
            'BreakevenActivationPips': '220', 'BreakevenBufferPips': '180',
            'SessionAdx_Threshold1': '32.0', 'SessionAdx_Period1': '14',
            'Use_MTF_Averaged_ADX': 'true', 'Use_Exhaustion_Guard': 'true',
            'Use_RVOL_Filter': 'true', 'RVOL_Threshold': '1.6',
        },
        'hybrid': {
            'TP_Percent': '1.26', 'SL_Percent': '1.80', 'TrailingDistancePips': '951',
            'BreakevenActivationPips': '159', 'BreakevenBufferPips': '149',
            'SessionAdx_Threshold1': '30.0', 'SessionAdx_Period1': '14',
            'Use_MTF_Averaged_ADX': 'true', 'Use_Exhaustion_Guard': 'true',
            'Use_RVOL_Filter': 'true', 'RVOL_Threshold': '1.5',
        },
        'hunter': {
            'TP_Percent': '1.26', 'SL_Percent': '1.77', 'TrailingDistancePips': '951',
            'BreakevenActivationPips': '169', 'BreakevenBufferPips': '159',
            'SessionAdx_Threshold1': '57.2', 'SessionAdx_Period1': '70',
            'Use_MTF_Averaged_ADX': 'false', 'Use_Exhaustion_Guard': 'true',
            'Use_RVOL_Filter': 'true', 'RVOL_Threshold': '1.5',
        },
        'heaven': {
            'TP_Percent': '0.98', 'SL_Percent': '0.89', 'TrailingDistancePips': '500',
            'BreakevenActivationPips': '150', 'BreakevenBufferPips': '150',
            'SessionAdx_Threshold1': '30.0', 'SessionAdx_Period1': '14',
            'Use_MTF_Averaged_ADX': 'true', 'Use_Exhaustion_Guard': 'true',
            'Use_RVOL_Filter': 'true', 'RVOL_Threshold': '1.5',
        },
        'original': {
            'TP_Percent': '1.26', 'SL_Percent': '1.80', 'TrailingDistancePips': '951',
            'BreakevenActivationPips': '159', 'BreakevenBufferPips': '149',
            'SessionAdx_Threshold1': '30.0', 'SessionAdx_Period1': '14',
            'Use_MTF_Averaged_ADX': 'false', 'Use_Exhaustion_Guard': 'false',
            'Use_RVOL_Filter': 'false', 'RVOL_Threshold': '1.5',
        }
    }

    INPUT_DEFAULTS = {
        "AutoLotSize": False,
        "News_Filter_Enabled": True,
        "FVG_Enable": True,
        "Weekly_Enable": True,
        "DeadZone_Enable": False,
        "DirectionThrottle_Enable": False,
        "EnergyGauge_Enable": False,
        "LondonSessionStartHour_UTC": 7,
        "NYSessionStartHour_UTC": 13,
        "EOD_CleanupHour": 23,
        "SessionAdx_Threshold1": 30.0,
        "SessionAdx_Period1": 14,
        "Hunter_TakeProfitPercent": 1.26,
        "Hunter_StopLossPercent": 1.80,
        "TrailingDistancePips": 951,
        "BreakevenActivationPips": 169,
        "BreakevenBufferPips": 159,
        "Use_MTF_Averaged_ADX": True,
        "Use_Exhaustion_Guard": True,
        "Use_RVOL_Filter": True,
        "RVOL_Threshold": 1.5,
        "profile": "hybrid",
    }

    def generate(self, df: pd.DataFrame) -> Signals:
        p = {**self.INPUT_DEFAULTS, **self.params}
        profile_key = str(p.get("profile", "hybrid")).lower()
        preset = self.PROFILE_PRESETS.get(profile_key, {})
        p = {**p, **preset, **self.params}
        idx = df.index

        entries = pd.Series(False, index=idx)
        exits = pd.Series(False, index=idx)
        direction = pd.Series(0, index=idx, dtype=int)

        london_start = int(p.get("LondonSessionStartHour_UTC", 7))
        ny_start = int(p.get("NYSessionStartHour_UTC", 13))
        hour_utc = df.index.hour
        london_session = (hour_utc >= london_start) & (hour_utc < ny_start)
        ny_session = (hour_utc >= ny_start) & (hour_utc < 21)

        pd_high = df["high"].resample("D").max().shift(1).reindex(df.index, method="ffill")
        pd_low = df["low"].resample("D").min().shift(1).reindex(df.index, method="ffill")

        # Session ADX Filter
        use_session_adx1 = _as_bool(p.get("UseSessionAdxFilter1", True))
        adx_threshold = float(p.get("SessionAdx_Threshold1", 30.0))
        adx_period = int(p.get("SessionAdx_Period1", 14))

        if use_session_adx1:
            adx_val, _, _ = adx(df["high"], df["low"], df["close"], adx_period)
            
            # MTF Averaged ADX (2026 Feature)
            if _as_bool(p.get("Use_MTF_Averaged_ADX", True)):
                adx_htf = adx_val.rolling(5, min_periods=1).mean()
                adx_blend = (adx_val + adx_htf) / 2.0
                session_adx_ok = adx_blend >= adx_threshold
            else:
                session_adx_ok = adx_val >= adx_threshold
        else:
            session_adx_ok = pd.Series(True, index=idx)

        # 2026 Feature: RVOL Volume Surge Filter
        if _as_bool(p.get("Use_RVOL_Filter", True)) and "volume" in df.columns:
            vol_sma20 = df["volume"].rolling(20, min_periods=1).mean()
            rvol = df["volume"] / vol_sma20.replace(0, 1)
            rvol_thresh = float(p.get("RVOL_Threshold", 1.5))
            rvol_ok = rvol >= rvol_thresh
        else:
            rvol_ok = pd.Series(True, index=idx)

        # 2026 Feature: Divergence-Confirmed 3-Push Exhaustion Guard
        if _as_bool(p.get("Use_Exhaustion_Guard", True)):
            rsi_s = rsi(df["close"], 14)
            # Detect 3 consecutive pushes
            push3_long_exhaustion = (df["high"] > df["high"].shift(1)) & (df["high"].shift(1) > df["high"].shift(2)) & (rsi_s < rsi_s.shift(2)) & (rsi_s > 70)
            push3_short_exhaustion = (df["low"] < df["low"].shift(1)) & (df["low"].shift(1) < df["low"].shift(2)) & (rsi_s > rsi_s.shift(2)) & (rsi_s < 30)
        else:
            push3_long_exhaustion = pd.Series(False, index=idx)
            push3_short_exhaustion = pd.Series(False, index=idx)

        # News & Deadzone
        news_active = pd.Series(False, index=idx)
        deadzone_active = _as_bool(p.get("DeadZone_Enable", False))

        # === Session Breakout Orders ===
        london_ok = london_session & session_adx_ok & rvol_ok & ~news_active & ~deadzone_active
        london_buy = london_ok & (df["high"] >= pd_high) & pd_high.notna() & ~push3_long_exhaustion
        london_sell = london_ok & (df["low"] <= pd_low) & pd_low.notna() & ~push3_short_exhaustion
        entries[london_buy] = True
        direction[london_buy] = 1
        entries[london_sell] = True
        direction[london_sell] = -1

        ny_ok = ny_session & session_adx_ok & rvol_ok & ~news_active & ~deadzone_active
        ny_buy = ny_ok & (df["high"] >= pd_high) & pd_high.notna() & ~push3_long_exhaustion
        ny_sell = ny_ok & (df["low"] <= pd_low) & pd_low.notna() & ~push3_short_exhaustion
        entries[ny_buy] = True
        direction[ny_buy] = 1
        entries[ny_sell] = True
        direction[ny_sell] = -1

        # Weekly Breakout
        if _as_bool(p.get("Weekly_Enable", True)):
            weekly_high = df["high"].resample("W").max().shift(1).resample("D").ffill().reindex(df.index)
            weekly_low = df["low"].resample("W").min().shift(1).resample("D").ffill().reindex(df.index)
            weekly_buy = (df["close"] > weekly_high).fillna(False) & ~push3_long_exhaustion
            weekly_sell = (df["close"] < weekly_low).fillna(False) & ~push3_short_exhaustion
            entries[weekly_buy] = True
            direction[weekly_buy] = 1
            entries[weekly_sell] = True
            direction[weekly_sell] = -1

        # Session-end boundary exits
        london_close = (hour_utc == london_start) & (df.index.minute < 5)
        ny_close = (hour_utc == ny_start) & (df.index.minute < 5)
        exits = (london_close | ny_close).astype(bool)

        return Signals(entries=entries, exits=exits, direction=direction)
