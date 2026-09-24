from .indicators import ac, ao, adx, mfi, macd, stochastic, bollinger, dem, force_index, linear_regression_slope, rsi, obv, cvd, wma, volume_rising, stc
from .ac_ao import AC_AO_Strategy
from .adx import ADX_Strategy
from .dem import DeM_Strategy
from .fbb import FBB_Strategy
from .bb_rsi import BBRsiStrategy
from .triple_rsi import TripleRSIStrategy
from .mfi import MFI_Strategy
from .ms import MS_Strategy
from .mtf_stoch import QuadStochStrategy, resample_htf
from .quad_stoch import QuadStochSameTF
from .stoch533_mtf import Stoch533MTF
from .macd_confluence import MACDConfluenceStrategy
from .crypto_9 import CryptoNineStrategy
from .light9_v2.light9_v2_strategy import Light9V2Strategy
from .light9.light9_strategy import Light9Strategy
from .crypto_9 import PATTERN_VARIANTS as CRYPTO_PATTERN_VARIANTS, PATTERN_NAMES as CRYPTO_PATTERN_NAMES, DEFAULT_PARAMS as CRYPTO_DEFAULT_PARAMS
from core.universal_strategy import UniversalStrategy
from core.regime import RegimeAwareStrategy, detect_regimes, regime_performance_summary

__all__ = [
    "AC_AO_Strategy", "ADX_Strategy", "DeM_Strategy", "FBB_Strategy",
    "BBRsiStrategy", "TripleRSIStrategy",
    "MFI_Strategy", "MS_Strategy", "QuadStochStrategy",
    "QuadStochSameTF", "Stoch533MTF", "MACDConfluenceStrategy",
    "CryptoNineStrategy",
    "Light9Strategy",
    "Light9V2Strategy",
]

# MQL5 product family 1: Multi Strat EA = 12 standalone strategies.
MULTI_STRAT_EA_REGISTRY = {
    "ac_ao": AC_AO_Strategy,
    "adx": ADX_Strategy,
    "dem": DeM_Strategy,
    "fbb": FBB_Strategy,
    "bb_rsi": BBRsiStrategy,
    "triple_rsi": TripleRSIStrategy,
    "mfi": MFI_Strategy,
    "ms": MS_Strategy,
    "mtf_stoch": QuadStochStrategy,
    "quad_stoch": QuadStochSameTF,
    "stoch533_mtf": Stoch533MTF,
    "macd_confluence": MACDConfluenceStrategy,
    "light9": Light9Strategy,
}

# MQL5 product family 2: Crypto 9 Strategy EA, kept separate from Multi Strat EA.
CRYPTO_STRAT_EA_REGISTRY = {
    "crypto_9": CryptoNineStrategy,
}

# NewMeta backtester can research both MQL5 product families.
TRADABLE_STRATEGY_REGISTRY = {
    **MULTI_STRAT_EA_REGISTRY,
    **CRYPTO_STRAT_EA_REGISTRY,
    "light9_v2": Light9V2Strategy,
}

# Research/composition engines belong to the NewMeta backtester, not a single MQL5 EA.
COMPOSITE_STRATEGY_REGISTRY = {
    "universal": UniversalStrategy,
    "regime_aware": RegimeAwareStrategy,
}

STRATEGY_REGISTRY = {
    **TRADABLE_STRATEGY_REGISTRY,
    **COMPOSITE_STRATEGY_REGISTRY,
}
