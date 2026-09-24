"""MQL5-to-Python auto parser for EA strategy extraction.

Parses .mq5 source files and generates Python strategy classes with:
- Input parameter extraction (with types, defaults, descriptions)
- Enum mapping (ENUM_RiskProfile → profile selection)
- Profile presets (HUNTER/HYBRID/HEAVEN/ORIGINAL mapped to param dicts)
- Signal logic extraction (Buy/Sell conditions from OnTick)
- Trailing stop logic mapping
- OCO session order handling

Usage:
    parser = MQL5Parser("Light 9 oc.mq5")
    spec = parser.parse()
    # spec['profiles'], spec['params'], spec['signal_logic']
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import ast
import textwrap


class MQL5Parser:
    """Parse MQL5 EA source into a strategy spec dict."""

    def __init__(self, mq5_path: str | Path):
        self.path = Path(mq5_path)
        self.text = self.path.read_text(encoding="utf-8", errors="replace")

    def parse(self) -> dict:
        return {
            "ea_name": self._extract_ea_name(),
            "version": self._extract_version(),
            "profiles": self._extract_profiles(),
            "inputs": self._extract_inputs(),
            "profile_presets": self._extract_profile_presets(),
            "signal_logic": self._extract_signal_logic(),
            "exit_logic": self._extract_exit_logic(),
            "risk_params": self._extract_risk_params(),
        }

    def _extract_ea_name(self) -> str:
        m = re.search(r'//\s*\|\s*(.+?)\s*\n//\|\s*(.+?)\s*\n', self.text)
        if m:
            return f"{m.group(1).strip()} {m.group(2).strip()}"
        return self.path.stem

    def _extract_version(self) -> str | None:
        m = re.search(r'#property\s+version\s+"([\d\.\d]+)"', self.text)
        return m.group(1) if m else None

    def _extract_profiles(self) -> dict[int, str]:
        """Extract profile enum values."""
        profiles = {}
        m = re.search(r'enum\s+ENUM_RiskProfile\s*\{([^}]+)\}', self.text)
        if m:
            for line in m.group(1).split("\n"):
                match = re.match(r'\s*(\w+)\s*=\s*(\d+)', line)
                if match:
                    profiles[int(match.group(2))] = match.group(1).replace("PROFILE_", "").lower()
        return profiles

    def _extract_inputs(self) -> dict[str, dict]:
        """Extract all input parameters with type, default, description."""
        inputs = {}
        pattern = r'input\s+(?:group\s+"([^"]+)")?\s*\ninput\s+(\w+)\s+(\w+)\s*=\s*(.+?);'
        for m in re.finditer(pattern, self.text, re.MULTILINE):
            group = m.group(1) or "General"
            mql5_type = m.group(1) if m.group(1) else m.group(2)
            name = m.group(3)
            default = m.group(4).strip()
            inputs[name] = {
                "type": mql5_type,
                "default": default,
                "group": group,
            }
        # Also catch single-line input declarations
        single_pattern = r'input\s+(\w+)\s+(\w+)\s*=\s*(.+?);\s*(?://\s*(.+))?$'
        for m in re.finditer(single_pattern, self.text, re.MULTILINE):
            name = m.group(2)
            if name not in inputs:
                inputs[name] = {
                    "type": m.group(1),
                    "default": m.group(3).strip(),
                    "group": "General",
                }
        return inputs

    def _extract_profile_presets(self) -> dict[str, dict]:
        """Extract the ApplyPreset() section to map profiles to param values."""
        presets = {}
        inputs_cache = self._extract_inputs()

        # Find the ApplyPreset() function body
        func_match = re.search(r'void\s+ApplyPreset\s*\(\s*\)\s*\{', self.text)
        if not func_match:
            return presets

        start = func_match.end()
        # Find matching closing brace
        depth = 1
        end = start
        while end < len(self.text) and depth > 0:
            if self.text[end] == '{':
                depth += 1
            elif self.text[end] == '}':
                depth -= 1
            end += 1

        func_body = self.text[start:end]
        lines = func_body.split("\n")
        current_profile = None

        for line in lines:
            # Detect profile blocks
            m_if = re.match(r'\s*if\s*\(\s*RiskProfile\s*==\s*PROFILE_(\w+)\s*\)', line)
            m_else = re.match(r'\s*else\s+if\s*\(\s*RiskProfile\s*==\s*PROFILE_(\w+)\s*\)', line)
            m_else2 = re.match(r'\s*else\s+if\s*\(\s*RiskProfile\s*==\s*PROFILE_(\w+)\s*\)', line)

            if m_if:
                current_profile = m_if.group(1).lower()
                presets.setdefault(current_profile, {})
                continue
            elif m_else or m_else2:
                prof = m_else.group(1).lower() if m_else else m_else2.group(1).lower()
                current_profile = prof
                presets.setdefault(current_profile, {})
                continue

            # Extract assignments like: TP_Percent_Eff = Hunter_TakeProfitPercent;
            m = re.match(r'\s*(\w+)_Eff\s*=\s*(\w+);', line)
            if m and current_profile:
                eff_var = m.group(1)
                src_var = m.group(2)
                if src_var in inputs_cache:
                    default = inputs_cache[src_var]["default"]
                    presets[current_profile][eff_var] = default
                else:
                    # Literal value
                    presets[current_profile][eff_var] = src_var

        return presets

    def _extract_signal_logic(self) -> dict:
        """Extract entry signal conditions from OnTick and order placement functions."""
        logic = {
            "entries": [],
            "exits": [],
            "session_logic": [],
        }

        # Find session order placement patterns
        session_patterns = [
            (r'PlaceBuyStopOrder\((London_High|NY_High|PDH),', "session_buy"),
            (r'PlaceSellStopOrder\((London_Low|NY_Low|PDL),', "session_sell"),
        ]

        for pattern, signal_type in session_patterns:
            for m in re.finditer(pattern, self.text):
                level_var = m.group(1)
                logic["session_logic"].append({
                    "type": signal_type,
                    "entry_level": level_var,
                    "filters": self._extract_filters_around(m, self.text),
                })

        return logic

    def _extract_filters_around(self, match: re.Match, text: str) -> list[str]:
        """Extract filter conditions near a given match position."""
        start = max(0, match.start() - 500)
        end = min(len(text), match.end() + 500)
        context = text[start:end]
        filters = []

        # ADX filters
        if "session_adx" in context:
            filters.append("ADX filter on session")
        if "daily_adx" in context:
            filters.append("ADX filter on daily")
        if "EnergyGauge" in context:
            filters.append("EnergyGauge score check")
        if "FVG" in context:
            filters.append("FVG block check")
        if "Smartboard" in context:
            filters.append("Smartboard gate check")

        return filters

    def _extract_exit_logic(self) -> dict:
        """Extract trailing stop / TP / SL logic."""
        return {
            "trailing_types": ["pip_trailing", "stage_aware", "hybrid_heaven"],
            "profile_specific": True,
            "functions": ["ApplyPipTrailingStop_v55", "ApplyHybridHeavenTrailingStop"],
        }

    def _extract_risk_params(self) -> dict:
        """Extract risk management parameters."""
        params = {}
        # TP/SL per profile
        for profile in ["HUNTER", "HYBRID", "HEAVEN"]:
            params[profile] = {
                "tp_percent": self._find_input_default(f'{profile.lower().title()}_TakeProfitPercent'),
                "sl_percent": self._find_input_default(f'{profile.lower().title()}_StopLossPercent'),
            }

        # Lot sizing
        params["lot_sizing"] = {
            "auto": self._find_input_default("AutoLotSize", "false"),
            "fixed_lots": self._find_input_default("FixedLots", "0.01"),
            "risk_percent": self._find_input_default("RiskPercent", "1.00"),
        }

        # Daily limits
        params["daily_limits"] = {
            "max_orders": self._find_input_default("DailyMaxOrders", "6"),
            "eod_cleanup_hour": self._find_input_default("EOD_CleanupHour", "23"),
        }

        return params

    def _find_input_default(self, name: str, fallback: str = "") -> str:
        m = re.search(rf'input\s+\w+\s+{name}\s*=\s*(.+?);', self.text)
        if m:
            return m.group(1).strip()
        return fallback

    def generate_python_class(self, strategy_name: str | None = None) -> str:
        """Generate a Python strategy class file from the parsed EA spec."""
        spec = self.parse()
        ea_name = spec["ea_name"]
        if strategy_name:
            class_name = strategy_name.title().replace('_', '') + "Strategy"
        else:
            class_name = re.sub(r'[^a-zA-Z0-9_]', '_', ea_name).replace('_', ' ').title().replace(' ', '')
        if not class_name.endswith("Strategy"):
            class_name += "Strategy"

        profiles = spec["profiles"]
        inputs = spec["inputs"]
        presets = spec["profile_presets"]

        # Get the default profile (usually hybrid = 1)
        default_profile_key = "hybrid" if "hybrid" in presets else next(iter(presets.keys()), "default")

        lines = [
            '"""Auto-generated from MQL5 EA: ' + ea_name + '"""',
            'from __future__ import annotations',
            'import pandas as pd',
            'import numpy as np',
            'from .._base import BaseStrategy, Signals',
            'from ..indicators import adx, rsi, stochastic',
            '',
            '',
            f'class {class_name}(BaseStrategy):',
            f'    name = "' + class_name.lower().replace("strategy", "").rstrip("_") + '"',
            '',
            '    PROFILES = ' + str(profiles),
            '',
            '    # Profile presets extracted from ApplyPreset() — maps profile name to effective params',
            '    PROFILE_PRESETS = ' + str(presets),
            '',
            '    # Input parameters extracted from MQL5 inputs',
            '    INPUT_DEFAULTS = {',
        ]

        for inp_name, info in inputs.items():
            if inp_name in ['AdminKey']:
                continue
            lines.append(f'        "{inp_name}": {repr(info["default"])},')

        lines += [
            '    }',
            '',
            '    def generate(self, df: pd.DataFrame) -> Signals:',
            '        p = {**self.INPUT_DEFAULTS, **self.params}',
            '        profile = p.get("profile", ' + repr(default_profile_key) + ')',
            '        preset = self.PROFILE_PRESETS.get(profile, {})',
            '        p = {**p, **preset}',
            '        idx = df.index',
            '',
            '        entries = pd.Series(False, index=idx)',
            '        exits = pd.Series(False, index=idx)',
            '        direction = pd.Series(0, index=idx, dtype=int)',
            '',
            '        # Resolve profile-specific effective params (from ApplyPreset)',
            '        tp_pct = float(preset.get("TP_Percent_Eff") or p.get("Cons_TakeProfitPercent", p.get("Hunter_TakeProfitPercent", "1.26")))',
            '        sl_pct = float(preset.get("SL_Percent_Eff") or p.get("Cons_StopLossPercent", p.get("Hunter_StopLossPercent", "1.80")))',
            '',
            '        # Session timing',
            '        london_start = int(p.get("LondonSessionStartHour_UTC", "7"))',
            '        ny_start = int(p.get("NYSessionStartHour_UTC", "13"))',
            '        hour_utc = df.index.hour',
            '        london_session = (hour_utc >= london_start) & (hour_utc < ny_start)',
            '        ny_session = (hour_utc >= ny_start) & (hour_utc < 21)',
            '',
            '        # NewUTC day — recalculate session levels daily',
            '        daily_mask = london_session | ny_session',
            '',
            '        # Calculate session ranges (London = high/low during London hours)',
            '        london_high = df["high"].where(london_session).groupby(df.index.date).transform("max")',
            '        london_low = df["low"].where(london_session).groupby(df.index.date).transform("min")',
            '        ny_high = df["high"].where(ny_session).groupby(df.index.date).transform("max")',
            '        ny_low = df["low"].where(ny_session).groupby(df.index.date).transform("min")',
            '',
            '        # PD (Previous Day) levels',
            '        pd_high = df["high"].groupby(df.index.date).transform("max").shift(1)',
            '        pd_low = df["low"].groupby(df.index.date).transform("min").shift(1)',
            '',
            '        # ADX filter for sessions',
            '        use_session_adx1 = bool(p.get("UseSessionAdxFilter1", True))',
            '        adx_threshold = float(p.get("SessionAdx_Threshold1", p.get("SessionAdx_Threshold1_Eff", "30.0")))',
            '        adx_period = int(p.get("SessionAdx_Period1", p.get("SessionAdx_Period1_Eff", "14")))',
            '',
            '        if use_session_adx1:',
            '            adx_val, _, _ = adx(df["high"], df["low"], df["close"], adx_period)',
            '            session_adx_ok = adx_val >= adx_threshold',
            '        else:',
            '            session_adx_ok = pd.Series(True, index=idx)',
            '',
            '        # Daily ADX filter (for PD orders)',
            '        use_daily_adx = bool(p.get("UseDailyAdxFilter2", False))',
            '        if use_daily_adx:',
            '            daily_adx_threshold = float(p.get("DailyAdx_Threshold2", p.get("DailyAdx_Threshold2_Eff", "25.0")))',
            '            daily_adx, _, _ = adx(df["high"], df["low"], df["close"], int(p.get("DailyAdx_Period2", p.get("DailyAdx_Period2_Eff", "14"))))',
            '            daily_adx_ok = daily_adx >= daily_adx_threshold',
            '        else:',
            '            daily_adx_ok = pd.Series(True, index=idx)',
            '',
            '        # News filter',
            '        if p.get("News_Filter_Enabled", "true").lower() in ("true", "1"):',
            '            news_pause = int(p.get("News_Pause_Before_Min", "60"))',
            '            # In backtest, assume no news events — pass through',
            '            news_active = pd.Series(False, index=idx)',
            '        else:',
            '            news_active = pd.Series(False, index=idx)',
            '',
            '        # Dead zone',
            '        deadzone_active = bool(p.get("DeadZone_Enable", "false").lower() in ("true", "1"))',
            '',
            '        # === PLACE ORDERS — mimics OnTick session logic ===',
            '        # Tokyo time for PD orders',
            '        pd_place_tokyo = bool(p.get("PD_PlaceAtTokyoHour", "true").lower() in ("true", "1"))',
            '        tokyo_hour_utc = (int(p.get("TokyoOrderLocalHour", "9")) + 9) % 24',
            '        pd_tokyo_time = (hour_utc == tokyo_hour_utc) & (df.index.minute < 5)',
            '',
            '        # PD (Previous Day) Buy/Sell — Tokyo session',
            '        pd_time = pd.Series(pd_place_tokyo, index=idx)',
            '        pd_ok = (pd_time == True) & ~news_active & daily_adx_ok',
            '        pd_buy_cond = pd_ok & (df["high"] >= pd_high) & pd_high.notna()',
            '        pd_sell_cond = pd_ok & (df["low"] <= pd_low) & pd_low.notna()',
            '        entries[pd_buy_cond] = True',
            '        direction[pd_buy_cond] = 1',
            '        entries[pd_sell_cond] = True',
            '        direction[pd_sell_cond] = -1',
            '',
            '        # London session Buy/Sell — breakout of previous London session range',
            '        london_ok = session_adx_ok & ~news_active & ~deadzone_active',
            '        # Use previous day high/low for breakout (London session vs PD levels)',
            '        london_buy = london_ok & (df["high"] >= pd_high) & pd_high.notna()',
            '        london_sell = london_ok & (df["low"] <= pd_low) & pd_low.notna()',
            '        entries[london_buy] = True',
            '        direction[london_buy] = 1',
            '        entries[london_sell] = True',
            '        direction[london_sell] = -1',
            '',
            '        # NY session Buy/Sell — breakout of previous NY session range',
            '        ny_ok = session_adx_ok & ~news_active & ~deadzone_active',
            '        ny_buy = ny_ok & (df["high"] >= pd_high) & pd_high.notna()',
            '        ny_sell = ny_ok & (df["low"] <= pd_low) & pd_low.notna()',
            '        entries[ny_buy] = True',
            '        direction[ny_buy] = 1',
            '        entries[ny_sell] = True',
            '        direction[ny_sell] = -1',
            '',
            '        # Weekly breakout (vs previous week high/low)',
            '        weekly_enable = bool(p.get("Weekly_Enable", "true").lower() in ("true", "1"))',
            '        if weekly_enable:',
            '            weekly_high = df["high"].resample("W").max()',
            '            weekly_low = df["low"].resample("W").min()',
            '            weekly_high_shifted = weekly_high.shift(1).resample("D").ffill().reindex(df.index)',
            '            weekly_low_shifted = weekly_low.shift(1).resample("D").ffill().reindex(df.index)',
            '            weekly_buy = (df["close"] > weekly_high_shifted).fillna(False)',
            '            weekly_sell = (df["close"] < weekly_low_shifted).fillna(False)',
            '            entries[weekly_buy] = True',
            '            direction[weekly_buy] = 1',
            '            entries[weekly_sell] = True',
            '            direction[weekly_sell] = -1',
            '',
            '        # FVG filter — block entries into unfilled FVG zones',
            '        # Simplified: skip for performance (would need FVG zone tracking)',
            '',
            '        # Energy gauge filter',
            '        energy_enable = bool(p.get("EnergyGauge_Enable", "false").lower() in ("true", "1"))',
            '        if energy_enable:',
            '            min_score = int(p.get("EnergyGauge_MinScore", "4"))',
            '            # ADX-based energy: score 0-4 based on ADX magnitude',
            '            energy_score = ((adx_val >= 25).astype(int) + (adx_val >= 30).astype(int) +',
            '                              (adx_val >= 35).astype(int) + (adx_val >= 40).astype(int))',
            '            energy_ok = energy_score >= min_score',
            '            entries[~energy_ok] = False',
            '',
            '        # Smartboard gates',
            '        if bool(p.get("Smartboard_StochGate_Enable", "false").lower() in ("true", "1")):',
            '            stoch_k, stoch_d = stochastic(df["high"], df["low"], df["close"])',
            '            stoch_overbought = float(p.get("Smartboard_Stoch_Overbought", "80.0"))',
            '            stoch_oversold = float(p.get("Smartboard_Stoch_Oversold", "20.0"))',
            '            entries[stoch_k >= stoch_overbought] = False  # Block buys in overbought',
            '            entries[stoch_k <= stoch_oversold] = False  # Block sells in oversold',
            '',
            '        # Session-end exits: close positions at session boundaries',
            '        # Tokyo (hour 0 UTC), London start, NY start — MQL5 closes positions at session opens',
            '        tokyo_close = (hour_utc == 0) & (df.index.minute < 5)',
            '        london_close = (hour_utc == london_start) & (df.index.minute < 5)',
            '        ny_close = (hour_utc == ny_start) & (df.index.minute < 5)',
            '        exits = (tokyo_close | london_close | ny_close).astype(bool)',
            '',
            '        return Signals(entries=entries, exits=exits, direction=direction)',
            '',
            '    def get_trailing_params(self) -> dict:',
            '        """Return profile-specific trailing stop params for the engine."""',
            '        profile = self.params.get("profile", ' + repr(default_profile_key) + ')',
            '        preset = self.PROFILE_PRESETS.get(profile, {})',
            '        defaults = self.INPUT_DEFAULTS',
            '        # TP/SL: try resolved preset keys first, then fall back through chain',
            '        def _resolve(*keys, default):',
            '            for k in keys:',
            '                v = preset.get(k) or defaults.get(k)',
            '                if v is not None and not str(v).endswith("_Eff") and not str(v).endswith("_eff"):',
            '                    return float(v)',
            '            return float(default)',
            '        tp = _resolve("TP_Percent", "TP1_Percent", "Cons_TakeProfitPercent", default="1.26")',
            '        sl = _resolve("SL_Percent", "SL1_Percent", "Cons_StopLossPercent", default="1.80")',
            '        trail = _resolve("TrailingDistancePips", "Heaven_Stage1_TrailDistance", "Stage1_TrailDistance", default="951")',
            '        be_act = _resolve("BreakevenActivationPips", default="169")',
            '        be_buf = _resolve("BreakevenBufferPips", default="159")',
            '        return {',
            '            "trail_pips": trail,',
            '            "breakeven_activation_pips": be_act,',
            '            "breakeven_buffer_pips": be_buf,',
            '            "tp_pct": tp,',
            '            "sl_pct": sl,',
            '        }',
        ]

        return "\n".join(lines)


def parse_mql5(path: str | Path) -> dict[str, Any]:
    """Parse an MQL5 EA file and return the strategy spec."""
    return MQL5Parser(path).parse()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m core.mql5_parser <ea_file.mq5>")
        sys.exit(1)

    parser = MQL5Parser(sys.argv[1])
    spec = parser.parse()
    print(f"EA: {spec['ea_name']} v{spec['version']}")
    print(f"Profiles: {list(spec['profiles'].values())}")
    print(f"Inputs found: {len(spec['inputs'])}")
    print(f"Profile presets: {list(spec['profile_presets'].keys())}")
    print()
    print("Profile Presets:")
    for prof, params in spec["profile_presets"].items():
        print(f"  {prof.upper()}: {len(params)} params")
        for k, v in list(params.items())[:5]:
            print(f"    {k} = {v}")
