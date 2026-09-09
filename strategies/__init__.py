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
from core.universal_strategy import UniversalStrategy
from core.regime import RegimeAwareStrategy, detect_regimes, regime_performance_summary

__all__ = [
    "AC_AO_Strategy", "ADX_Strategy", "DeM_Strategy", "FBB_Strategy",
    "BBRsiStrategy", "TripleRSIStrategy",
    "MFI_Strategy", "MS_Strategy", "QuadStochStrategy",
    "QuadStochSameTF", "Stoch533MTF", "MACDConfluenceStrategy",
    "UniversalStrategy", "RegimeAwareStrategy",
    "STRATEGY_REGISTRY",
]

STRATEGY_REGISTRY = {
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
    "universal": UniversalStrategy,
    "regime_aware": RegimeAwareStrategy,
}