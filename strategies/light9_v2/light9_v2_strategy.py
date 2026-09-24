"""Auto-generated from MQL5 EA: LIGHT Primary - Professional Edition             | Ultimate Trading System                     |"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .._base import BaseStrategy, Signals
from ..indicators import adx, rsi, stochastic


class Light9V2Strategy(BaseStrategy):
    name = "light9v2"

    PROFILES = {0: 'hunter', 1: 'hybrid', 2: 'heaven', 3: 'original'}

    # Profile presets extracted from ApplyPreset() — maps profile name to effective params
    PROFILE_PRESETS = {'hunter': {'MagicNumber': '6501', 'TP_Percent': '1.26', 'SL_Percent': '1.77', 'BreakevenActivationPips': '169', 'BreakevenBufferPips': '159', 'TrailingDistancePips': '951', 'Trail_UseOnePipStep': 'false', 'MinTrailStepPips': '78', 'Phase2_ActivationPips': '0', 'Phase2_TrailingDistancePips': '0', 'Phase2_MinTrailStepPips': '0', 'LondonDelayMinutes': '60', 'NYDelayMinutes': '120', 'UseSessionAdxFilter1': 'true', 'SessionAdx_Timeframe1': '5', 'SessionAdx_Period1': '70', 'SessionAdx_Threshold1': '57.2', 'UseSessionAdxFilter2': 'true', 'SessionAdx_Timeframe2': '2', 'SessionAdx_Period2': '30', 'SessionAdx_Threshold2': '40.0', 'UseDailyAdxFilter1': 'false', 'DailyAdx_Timeframe1': '15', 'DailyAdx_Period1': '14', 'DailyAdx_Threshold1': '23.0', 'UseDailyAdxFilter2': 'false', 'DailyAdx_Timeframe2': '15', 'DailyAdx_Period2': '14', 'DailyAdx_Threshold2': '20.0', 'Heaven_EnableApproachFilter': 'false', 'Heaven_Enable3StageTrailing': 'false'}, 'hybrid': {'MagicNumber': '6502', 'TP_Percent': '1.26', 'SL_Percent': '1.8', 'BreakevenActivationPips': '159', 'BreakevenBufferPips': '149', 'TrailingDistancePips': '951', 'Stage2_TrailPips': '637', 'Stage3_TrailPips': '314', 'Trail_UseOnePipStep': 'false', 'MinTrailStepPips': '78', 'Phase2_ActivationPips': '0', 'Phase2_TrailingDistancePips': '0', 'Phase2_MinTrailStepPips': '0', 'LondonDelayMinutes': '60', 'NYDelayMinutes': '120', 'UseSessionAdxFilter1': 'true', 'SessionAdx_Timeframe1': '1', 'SessionAdx_Period1': '14', 'SessionAdx_Threshold1': '30.0', 'UseSessionAdxFilter2': 'true', 'SessionAdx_Timeframe2': '5', 'SessionAdx_Period2': '14', 'SessionAdx_Threshold2': '25.0', 'UseDailyAdxFilter1': 'false', 'DailyAdx_Timeframe1': '15', 'DailyAdx_Period1': '14', 'DailyAdx_Threshold1': '23.0', 'UseDailyAdxFilter2': 'true', 'DailyAdx_Timeframe2': '15', 'DailyAdx_Period2': '14', 'DailyAdx_Threshold2': '25.0', 'Heaven_EnableApproachFilter': 'false', 'Heaven_Enable3StageTrailing': 'false'}, 'heaven': {'MagicNumber': '7603', 'TP_Percent': '0.98', 'SL_Percent': '0.89', 'Heaven_EnableApproachFilter': 'true', 'Heaven_ApproachDistancePips': '20', 'Heaven_CancelIfMultipleFail': 'true', 'Heaven_MinFiltersToPass': '2', 'Heaven_UseApproachAdxFilter1': 'true', 'Heaven_UseApproachAdxFilter2': 'true', 'Heaven_ApproachAdxThreshold1': '50.0', 'Heaven_ApproachAdxThreshold2': '45.0', 'Heaven_UseApproachVolumeFilter': 'true', 'Heaven_VolumeMultiplierRequired': '2.5', 'Heaven_VolumeLookbackPeriod': '20', 'Heaven_CheckVolumeCluster': 'true', 'Heaven_VolumeClusterBars': '5', 'Heaven_UseApproachOBVFilter': 'true', 'Heaven_OBV_MA_Period': '14', 'Heaven_OBV_DivergenceThreshold': '5.0', 'Heaven_OBV_MustTrendWithBreak': 'true', 'Heaven_UseApproachFVGFilter': 'true', 'Heaven_Enable3StageTrailing': 'true', 'Heaven_Stage1_ActivationPct': '33.0', 'Heaven_Stage2_ActivationPct': '66.0', 'Heaven_Stage3_ActivationPct': '100.0', 'Heaven_Stage1_TrailDistance': '500', 'Heaven_Stage2_TrailDistance': '350', 'Heaven_Stage3_TrailDistance': '175', 'Heaven_BreakevenBufferPips': '150', 'BreakevenActivationPips': 'Heaven_Stage1_TrailDistance_Eff', 'BreakevenBufferPips': 'Heaven_BreakevenBufferPips_Eff', 'TrailingDistancePips': 'Heaven_Stage1_TrailDistance_Eff', 'Stage2_TrailPips': 'Heaven_Stage2_TrailDistance_Eff', 'Stage3_TrailPips': 'Heaven_Stage3_TrailDistance_Eff', 'Trail_UseOnePipStep': 'false', 'MinTrailStepPips': '190', 'TP1_Percent': '0.5', 'Phase2_ActivationPips': '100', 'Phase2_TrailingDistancePips': '260', 'Phase2_MinTrailStepPips': '22', 'LondonDelayMinutes': '60', 'NYDelayMinutes': '120', 'UseSessionAdxFilter1': 'true', 'SessionAdx_Timeframe1': '1', 'SessionAdx_Period1': '14', 'SessionAdx_Threshold1': '30.0', 'UseSessionAdxFilter2': 'false', 'SessionAdx_Timeframe2': '15', 'SessionAdx_Period2': '14', 'SessionAdx_Threshold2': '20.0', 'UseDailyAdxFilter1': 'false', 'DailyAdx_Timeframe1': '15', 'DailyAdx_Period1': '14', 'DailyAdx_Threshold1': '23.0', 'UseDailyAdxFilter2': 'true', 'DailyAdx_Timeframe2': '15', 'DailyAdx_Period2': '14', 'DailyAdx_Threshold2': '25.0'}, 'original': {'MagicNumber': '6504', 'BreakevenActivationPips': '159', 'BreakevenBufferPips': '149', 'TrailingDistancePips': '951', 'Stage2_TrailPips': '637', 'Stage3_TrailPips': '314', 'Trail_UseOnePipStep': 'false', 'MinTrailStepPips': '78', 'Phase2_ActivationPips': '0', 'Phase2_TrailingDistancePips': '0', 'Phase2_MinTrailStepPips': '0', 'LondonDelayMinutes': '60', 'NYDelayMinutes': '120', 'UseSessionAdxFilter1': 'true', 'SessionAdx_Timeframe1': 'PERIOD_M1', 'SessionAdx_Period1': '14', 'UseSessionAdxFilter2': 'true', 'SessionAdx_Timeframe2': 'PERIOD_M5', 'SessionAdx_Period2': '14', 'UseDailyAdxFilter1': 'false', 'DailyAdx_Timeframe1': 'PERIOD_M15', 'DailyAdx_Period1': '14', 'UseDailyAdxFilter2': 'true', 'DailyAdx_Timeframe2': 'PERIOD_M15', 'DailyAdx_Period2': '14', 'Heaven_EnableApproachFilter': 'false', 'Heaven_Enable3StageTrailing': 'false'}}

    # Input parameters extracted from MQL5 inputs
    INPUT_DEFAULTS = {
        "AutoLotSize": 'false',
        "ProfitVelocityEngine_Enable": 'true',
        "Hunter_MagicNumber": '6501',
        "News_Filter_Enabled": 'true',
        "FVG_Enable": 'true',
        "Weekly_Enable": 'true',
        "DeadZone_Enable": 'true',
        "DirectionThrottle_Enable": 'true',
        "EnergyGauge_Enable": 'true',
        "NY_LiquidityGrab_Enable": 'true',
        "Smartboard_StochGate_Enable": 'true',
        "Hunter_TakeProfitPercent": '1.26',
        "Cons_TakeProfitPercent": '1.26',
        "Use_ATR_SlowGuard": 'false',
        "Heaven_TakeProfitPercent": '0.98',
        "Original_TakeProfitPercent": '1.26',
        "Heaven_EnableApproachFilter": 'true',
        "Heaven_UseApproachAdxFilter1": 'true',
        "Heaven_UseApproachVolumeFilter": 'true',
        "Heaven_UseApproachOBVFilter": 'true',
        "Heaven_UseApproachFVGFilter": 'true',
        "Heaven_Enable3StageTrailing": 'true',
        "SkipWeekendsUTC": 'true',
        "LondonSessionStartHour_UTC": '7',
        "EOD_CleanupHour": '23',
        "DynamicTP_Enable": 'true',
        "CancelOppositeOnFill": 'false',
        "DebugMode": 'true',
        "FixedLots": '0.01',
        "RiskPercent": '1.0',
        "MaxFixedLots": '5.0',
        "SleepOnWeekends": 'true',
        "RiskProfile": '2',
        "ProfitVelocityEngine_Mode": '0',
        "ProfitVelocityEngine_RefreshMinutes": '5',
        "ProfitVelocityEngine_EMA_Alpha": '0.35',
        "ProfitVelocityEngine_AutoDefensiveSpeed": '0.0',
        "ProfitVelocityEngine_AutoAggressiveSpeed": '0.25',
        "ProfitVelocityEngine_MinTotalLots": '0.01',
        "ProfitVelocityEngine_DefensiveRoomMult": '0.9',
        "ProfitVelocityEngine_AggressiveRoomMult": '1.15',
        "ProfitVelocityEngine_DefensiveTrailMult": '0.85',
        "ProfitVelocityEngine_AggressiveTrailMult": '1.25',
        "ProfitVelocityEngine_DefensiveTPMult": '0.95',
        "ProfitVelocityEngine_AggressiveTPMult": '1.1',
        "ProfitVelocityEngine_DefensiveMoveMult": '0.85',
        "ProfitVelocityEngine_AggressiveMoveMult": '1.15',
        "Hybrid_MagicNumber": '6502',
        "Heaven_MagicNumber": '7603',
        "Original_MagicNumber": '6504',
        "Weekly_MagicNumber": '1520',
        "News_Pause_Before_Min": '240',
        "News_Pause_After_Min": '180',
        "News_Delete_Pending": 'true',
        "News_AllowSLAdjustment": 'true',
        "FVG_Timeframe": '5',
        "FVG_LookbackBars": '14',
        "FVG_Mode": '0',
        "ShowFVG_Boxes": 'true',
        "ShowFilledFVG": 'false',
        "BullishFVG_Color": '32768',
        "BearishFVG_Color": '255',
        "FilledFVG_Color": '8421504',
        "FVG_ForceDailyPlacement": 'false',
        "Weekly_OffsetPips": '3',
        "Weekly_LotMultiplier": '2.0',
        "Weekly_TakeProfitPercent": '1.5',
        "Weekly_StopLossPercent": '2.0',
        "DeadZone_StartHour_UTC": '22',
        "DeadZone_EndHour_UTC": '24',
        "DirectionThrottle_TF": '5',
        "DirectionThrottle_Bars": '2',
        "DirectionThrottle_WickBodyRatio": '2.0',
        "DirectionThrottle_DoubleTopLookback": '8',
        "DirectionThrottle_DoubleBottomLookback": '8',
        "DirectionThrottle_TolerancePips": '25.0',
        "EnergyGauge_TF": '5',
        "EnergyGauge_MinScore": '4',
        "EnergyGauge_ForceStraddleOCO": 'true',
        "EnergyGauge_RVOL_Lookback": '20',
        "EnergyGauge_RVOL_MinRatio": '1.2',
        "EnergyGauge_ChaikinFast": '3',
        "EnergyGauge_ChaikinSlow": '10',
        "EnergyGauge_MomentumPeriod": '10',
        "EnergyGauge_ATRPeriod": '14',
        "EnergyGauge_MAPeriod": '20',
        "AsiaSession_StartHour_UTC": '0',
        "AsiaSession_EndHour_UTC": '7',
        "LiquidityGrab_SweepPips": '5',
        "Smartboard_ADXGate_Enable": 'false',
        "Smartboard_VolDeltaGate_Enable": 'false',
        "Smartboard_MACDGate_Enable": 'false',
        "Smartboard_ApplyToDailyOrders": 'true',
        "Smartboard_ApplyToSessionOrders": 'false',
        "Smartboard_ApplyToWeeklyOrders": 'false',
        "Smartboard_Stoch_Overbought": '80',
        "Smartboard_Stoch_Oversold": '20.0',
        "Smartboard_ADX_Min": '21.0',
        "Hunter_StopLossPercent": '1.77',
        "Hunter_BreakevenActivationPips": '169',
        "Hunter_BreakevenBufferPips": '159',
        "Hunter_TrailingDistancePips": '951',
        "Hunter_Trail_UseOnePipStep": 'false',
        "Hunter_MinTrailStepPips": '78',
        "Hunter_LondonDelayMinutes": '60',
        "Hunter_NYDelayMinutes": '120',
        "Hunter_UseSessionAdxFilter1": 'true',
        "Hunter_SessionAdx_Timeframe1": '5',
        "Hunter_SessionAdx_Period1": '70',
        "Hunter_SessionAdx_Threshold1": '57.2',
        "Hunter_UseSessionAdxFilter2": 'true',
        "Hunter_SessionAdx_Timeframe2": '2',
        "Hunter_SessionAdx_Period2": '30',
        "Hunter_SessionAdx_Threshold2": '40.0',
        "Hunter_UseDailyAdxFilter1": 'false',
        "Hunter_DailyAdx_Timeframe1": '15',
        "Hunter_DailyAdx_Period1": '14',
        "Hunter_DailyAdx_Threshold1": '23.0',
        "Hunter_UseDailyAdxFilter2": 'false',
        "Hunter_DailyAdx_Timeframe2": '15',
        "Hunter_DailyAdx_Period2": '14',
        "Hunter_DailyAdx_Threshold2": '20.0',
        "Cons_StopLossPercent": '1.8',
        "Cons_Stage1_ActivationPips": '159',
        "Cons_Stage1_BufferPips": '149',
        "Cons_Stage1_TrailPips": '951',
        "Cons_Stage2_TrailPips": '637',
        "Cons_Stage3_TrailPips": '314',
        "Cons_MinTrailStepPips": '78',
        "Cons_LondonDelayMinutes": '60',
        "Cons_NYDelayMinutes": '120',
        "Cons_UseSessionAdxFilter1": 'true',
        "Cons_SessionAdx_Timeframe1": '1',
        "Cons_SessionAdx_Period1": '14',
        "Cons_SessionAdx_Threshold1": '30.0',
        "Cons_UseSessionAdxFilter2": 'true',
        "Cons_SessionAdx_Timeframe2": '5',
        "Cons_SessionAdx_Period2": '14',
        "Cons_SessionAdx_Threshold2": '25.0',
        "Cons_UseDailyAdxFilter1": 'false',
        "Cons_DailyAdx_Timeframe1": '15',
        "Cons_DailyAdx_Period1": '14',
        "Cons_DailyAdx_Threshold1": '23.0',
        "Cons_UseDailyAdxFilter2": 'true',
        "Cons_DailyAdx_Timeframe2": '15',
        "Cons_DailyAdx_Period2": '14',
        "Cons_DailyAdx_Threshold2": '25.0',
        "ATRSlow_Timeframe": '15',
        "ATRSlow_Period": '21',
        "ATRSlow_Multiplier": '2.50',
        "ATRSlow_MinProfitPipsActivate": '120',
        "ATRSlow_UpdateOnNewBarOnly": 'true',
        "Heaven_StopLossPercent": '0.89',
        "Heaven_TP1_Percent": '0.5',
        "Heaven_PostTP1_ActivationPips": '100',
        "Heaven_PostTP1_TrailDistancePips": '260',
        "Heaven_PostTP1_MinStepPips": '22',
        "Heaven_LondonDelayMinutes": '60',
        "Heaven_NYDelayMinutes": '120',
        "Heaven_UseSessionAdxFilter1": 'true',
        "Heaven_SessionAdx_Timeframe1": '1',
        "Heaven_SessionAdx_Period1": '14',
        "Heaven_SessionAdx_Threshold1": '30.0',
        "Heaven_UseSessionAdxFilter2": 'false',
        "Heaven_SessionAdx_Timeframe2": '15',
        "Heaven_SessionAdx_Period2": '14',
        "Heaven_SessionAdx_Threshold2": '20.0',
        "Heaven_UseDailyAdxFilter1": 'false',
        "Heaven_DailyAdx_Timeframe1": '15',
        "Heaven_DailyAdx_Period1": '14',
        "Heaven_DailyAdx_Threshold1": '23.0',
        "Heaven_UseDailyAdxFilter2": 'true',
        "Heaven_DailyAdx_Timeframe2": '15',
        "Heaven_DailyAdx_Period2": '14',
        "Heaven_DailyAdx_Threshold2": '25.0',
        "Original_StopLossPercent": '1.80',
        "Original_Stage1_ActivationPips": '159',
        "Original_Stage1_BufferPips": '149',
        "Original_Stage1_TrailPips": '951',
        "Original_Stage2_TrailPips": '637',
        "Original_Stage3_TrailPips": '314',
        "Original_MinTrailStepPips": '78',
        "Original_LondonDelayMinutes": '60',
        "Original_NYDelayMinutes": '120',
        "Original_UseSessionAdxFilter1": 'true',
        "Original_SessionAdx_Timeframe1": 'PERIOD_M1',
        "Original_SessionAdx_Period1": '14',
        "Original_SessionAdx_Threshold1": '30.0',
        "Original_UseSessionAdxFilter2": 'true',
        "Original_SessionAdx_Timeframe2": 'PERIOD_M5',
        "Original_SessionAdx_Period2": '14',
        "Original_SessionAdx_Threshold2": '25.0',
        "Original_UseDailyAdxFilter1": 'false',
        "Original_DailyAdx_Timeframe1": 'PERIOD_M15',
        "Original_DailyAdx_Period1": '14',
        "Original_DailyAdx_Threshold1": '23.0',
        "Original_UseDailyAdxFilter2": 'true',
        "Original_DailyAdx_Timeframe2": 'PERIOD_M15',
        "Original_DailyAdx_Period2": '14',
        "Original_DailyAdx_Threshold2": '25.0',
        "Heaven_ApproachDistancePips": '20',
        "Heaven_CancelIfMultipleFail": 'true',
        "Heaven_MinFiltersToPass": '2',
        "Heaven_ApproachAdxThreshold1": '50.0',
        "Heaven_UseApproachAdxFilter2": 'true',
        "Heaven_ApproachAdxThreshold2": '45.0',
        "Heaven_VolumeMultiplierRequired": '2.5',
        "Heaven_VolumeLookbackPeriod": '20',
        "Heaven_CheckVolumeCluster": 'true',
        "Heaven_VolumeClusterBars": '5',
        "Heaven_OBV_MA_Period": '14',
        "Heaven_OBV_DivergenceThreshold": '5.0',
        "Heaven_OBV_MustTrendWithBreak": 'true',
        "Heaven_ApproachFVGTF": '5',
        "Heaven_Stage1_ActivationPct": '33.0',
        "Heaven_Stage2_ActivationPct": '66.0',
        "Heaven_Stage3_ActivationPct": '100.0',
        "Heaven_Stage1_TrailDistance": '500',
        "Heaven_Stage2_TrailDistance": '350',
        "Heaven_Stage3_TrailDistance": '175',
        "Heaven_BreakevenBufferPips": '150',
        "PD_UseUTC_M1": 'true',
        "PD_PlaceAtTokyoHour": 'true',
        "TokyoOrderLocalHour": '9',
        "NYSessionStartHour_UTC": '13',
        "EOD_CleanupMinute": '55',
        "DailyMaxOrders": '6',
        "UseDailyBaselineRisk": 'true',
        "DynamicTP_ActivationPips": '220',
        "DynamicTP_BufferPips": '450',
        "DynamicTP_MinStepPips": '60',
        "PreventSameDirectionStack": 'false',
        "RequireDailyPositionClosed": 'false',
        "RecalculateLevelsOnReentry": 'false',
        "EnableReentryAfterDailyClose": 'false',
        "ReentryCooldownSeconds": '60',
        "ReentryOnlyOnM1Bar": 'false',
        "Reentry_UsePolarityRetest": 'false',
        "Reentry_UseFVGFill": 'false',
        "Reentry_ConfirmTF": '5',
        "Reentry_RetestTolerancePips": '10.0',
        "IgnoreRiskFreeExposureBlocks": 'false',
        "RiskFreeMinLockedPips": '0.0',
        "AllowRecheckBypass_SessionExposure": 'false',
        "AllowRecheckBypass_Smartboard": 'false',
        "AllowRecheckBypass_FVG": 'false',
        "AllowRecheckBypass_Direction": 'false',
        "AllowRecheckBypass_Structure": 'false',
        "GateTelemetry_Enable": 'true',
        "GateTelemetry_ReportEveryM1Bars": '30',
        "GateTelemetry_LogPerDecision": 'false',
        "GateReplay_Enable": 'false',
        "GateReplay_LogPasses": 'false',
        "GateReplay_UseCommonFile": 'true',
    }

    def generate(self, df: pd.DataFrame) -> Signals:
        p = {**self.INPUT_DEFAULTS, **self.params}
        profile = p.get("profile", 'hybrid')
        preset = self.PROFILE_PRESETS.get(profile, {})
        p = {**p, **preset}
        idx = df.index

        entries = pd.Series(False, index=idx)
        exits = pd.Series(False, index=idx)
        direction = pd.Series(0, index=idx, dtype=int)

        # Resolve profile-specific effective params (from ApplyPreset)
        tp_pct = float(preset.get("TP_Percent_Eff") or p.get("Cons_TakeProfitPercent", p.get("Hunter_TakeProfitPercent", "1.26")))
        sl_pct = float(preset.get("SL_Percent_Eff") or p.get("Cons_StopLossPercent", p.get("Hunter_StopLossPercent", "1.80")))

        # Session timing
        london_start = int(p.get("LondonSessionStartHour_UTC", "7"))
        ny_start = int(p.get("NYSessionStartHour_UTC", "13"))
        hour_utc = df.index.hour
        london_session = (hour_utc >= london_start) & (hour_utc < ny_start)
        ny_session = (hour_utc >= ny_start) & (hour_utc < 21)

        # NewUTC day — recalculate session levels daily
        daily_mask = london_session | ny_session

        # Calculate session ranges (London = high/low during London hours)
        london_high = df["high"].where(london_session).groupby(df.index.date).transform("max")
        london_low = df["low"].where(london_session).groupby(df.index.date).transform("min")
        ny_high = df["high"].where(ny_session).groupby(df.index.date).transform("max")
        ny_low = df["low"].where(ny_session).groupby(df.index.date).transform("min")

        # PD (Previous Day) levels
        pd_high = df["high"].groupby(df.index.date).transform("max").shift(1)
        pd_low = df["low"].groupby(df.index.date).transform("min").shift(1)

        # ADX filter for sessions
        use_session_adx1 = bool(p.get("UseSessionAdxFilter1", True))
        adx_threshold = float(p.get("SessionAdx_Threshold1", p.get("SessionAdx_Threshold1_Eff", "30.0")))
        adx_period = int(p.get("SessionAdx_Period1", p.get("SessionAdx_Period1_Eff", "14")))

        if use_session_adx1:
            adx_val, _, _ = adx(df["high"], df["low"], df["close"], adx_period)
            session_adx_ok = adx_val >= adx_threshold
        else:
            session_adx_ok = pd.Series(True, index=idx)

        # Daily ADX filter (for PD orders)
        use_daily_adx = bool(p.get("UseDailyAdxFilter2", False))
        if use_daily_adx:
            daily_adx_threshold = float(p.get("DailyAdx_Threshold2", p.get("DailyAdx_Threshold2_Eff", "25.0")))
            daily_adx, _, _ = adx(df["high"], df["low"], df["close"], int(p.get("DailyAdx_Period2", p.get("DailyAdx_Period2_Eff", "14"))))
            daily_adx_ok = daily_adx >= daily_adx_threshold
        else:
            daily_adx_ok = pd.Series(True, index=idx)

        # News filter
        if p.get("News_Filter_Enabled", "true").lower() in ("true", "1"):
            news_pause = int(p.get("News_Pause_Before_Min", "60"))
            # In backtest, assume no news events — pass through
            news_active = pd.Series(False, index=idx)
        else:
            news_active = pd.Series(False, index=idx)

        # Dead zone
        deadzone_active = bool(p.get("DeadZone_Enable", "false").lower() in ("true", "1"))

        # === PLACE ORDERS — mimics OnTick session logic ===
        # Tokyo time for PD orders
        pd_place_tokyo = bool(p.get("PD_PlaceAtTokyoHour", "true").lower() in ("true", "1"))
        tokyo_hour_utc = (int(p.get("TokyoOrderLocalHour", "9")) + 9) % 24
        pd_tokyo_time = (hour_utc == tokyo_hour_utc) & (df.index.minute < 5)

        # PD (Previous Day) Buy/Sell — Tokyo session
        pd_time = pd.Series(pd_place_tokyo, index=idx)
        pd_ok = (pd_time == True) & ~news_active & daily_adx_ok
        pd_buy_cond = pd_ok & (df["high"] >= pd_high) & pd_high.notna()
        pd_sell_cond = pd_ok & (df["low"] <= pd_low) & pd_low.notna()
        entries[pd_buy_cond] = True
        direction[pd_buy_cond] = 1
        entries[pd_sell_cond] = True
        direction[pd_sell_cond] = -1

        # London session Buy/Sell — breakout of previous London session range
        london_ok = session_adx_ok & ~news_active & ~deadzone_active
        # Use previous day high/low for breakout (London session vs PD levels)
        london_buy = london_ok & (df["high"] >= pd_high) & pd_high.notna()
        london_sell = london_ok & (df["low"] <= pd_low) & pd_low.notna()
        entries[london_buy] = True
        direction[london_buy] = 1
        entries[london_sell] = True
        direction[london_sell] = -1

        # NY session Buy/Sell — breakout of previous NY session range
        ny_ok = session_adx_ok & ~news_active & ~deadzone_active
        ny_buy = ny_ok & (df["high"] >= pd_high) & pd_high.notna()
        ny_sell = ny_ok & (df["low"] <= pd_low) & pd_low.notna()
        entries[ny_buy] = True
        direction[ny_buy] = 1
        entries[ny_sell] = True
        direction[ny_sell] = -1

        # Weekly breakout (vs previous week high/low)
        weekly_enable = bool(p.get("Weekly_Enable", "true").lower() in ("true", "1"))
        if weekly_enable:
            weekly_high = df["high"].resample("W").max()
            weekly_low = df["low"].resample("W").min()
            weekly_high_shifted = weekly_high.shift(1).resample("D").ffill().reindex(df.index)
            weekly_low_shifted = weekly_low.shift(1).resample("D").ffill().reindex(df.index)
            weekly_buy = (df["close"] > weekly_high_shifted).fillna(False)
            weekly_sell = (df["close"] < weekly_low_shifted).fillna(False)
            entries[weekly_buy] = True
            direction[weekly_buy] = 1
            entries[weekly_sell] = True
            direction[weekly_sell] = -1

        # FVG filter — block entries into unfilled FVG zones
        # Simplified: skip for performance (would need FVG zone tracking)

        # Energy gauge filter
        energy_enable = bool(p.get("EnergyGauge_Enable", "false").lower() in ("true", "1"))
        if energy_enable:
            min_score = int(p.get("EnergyGauge_MinScore", "4"))
            # ADX-based energy: score 0-4 based on ADX magnitude
            energy_score = ((adx_val >= 25).astype(int) + (adx_val >= 30).astype(int) +
                              (adx_val >= 35).astype(int) + (adx_val >= 40).astype(int))
            energy_ok = energy_score >= min_score
            entries[~energy_ok] = False

        # Smartboard gates
        if bool(p.get("Smartboard_StochGate_Enable", "false").lower() in ("true", "1")):
            stoch_k, stoch_d = stochastic(df["high"], df["low"], df["close"])
            stoch_overbought = float(p.get("Smartboard_Stoch_Overbought", "80.0"))
            stoch_oversold = float(p.get("Smartboard_Stoch_Oversold", "20.0"))
            entries[stoch_k >= stoch_overbought] = False  # Block buys in overbought
            entries[stoch_k <= stoch_oversold] = False  # Block sells in oversold

        # Session-end exits: close positions at session boundaries
        # Tokyo (hour 0 UTC), London start, NY start — MQL5 closes positions at session opens
        tokyo_close = (hour_utc == 0) & (df.index.minute < 5)
        london_close = (hour_utc == london_start) & (df.index.minute < 5)
        ny_close = (hour_utc == ny_start) & (df.index.minute < 5)
        exits = (tokyo_close | london_close | ny_close).astype(bool)

        return Signals(entries=entries, exits=exits, direction=direction)

    def get_trailing_params(self) -> dict:
        """Return profile-specific trailing stop params for the engine."""
        profile = self.params.get("profile", 'hybrid')
        preset = self.PROFILE_PRESETS.get(profile, {})
        defaults = self.INPUT_DEFAULTS
        # TP/SL: try resolved preset keys first, then fall back through chain
        def _resolve(*keys, default):
            for k in keys:
                v = preset.get(k) or defaults.get(k)
                if v is not None and not str(v).endswith("_Eff") and not str(v).endswith("_eff"):
                    return float(v)
            return float(default)
        tp = _resolve("TP_Percent", "TP1_Percent", "Cons_TakeProfitPercent", default="1.26")
        sl = _resolve("SL_Percent", "SL1_Percent", "Cons_StopLossPercent", default="1.80")
        trail = _resolve("TrailingDistancePips", "Heaven_Stage1_TrailDistance", "Stage1_TrailDistance", default="951")
        be_act = _resolve("BreakevenActivationPips", default="169")
        be_buf = _resolve("BreakevenBufferPips", default="159")
        return {
            "trail_pips": trail,
            "breakeven_activation_pips": be_act,
            "breakeven_buffer_pips": be_buf,
            "tp_pct": tp,
            "sl_pct": sl,
        }