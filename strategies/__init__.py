from .indicators import ac, ao, adx, mfi, macd, stochastic, bollinger, dem, force_index, linear_regression_slope
from .ac_ao import AC_AO_Strategy
from .adx import ADX_Strategy
from .dem import DeM_Strategy
from .fbb import FBB_Strategy
from .mfi import MFI_Strategy
from .ms import MS_Strategy
from .mtf_stoch import QuadStochStrategy, resample_htf
from core.universal_strategy import UniversalStrategy

STRATEGY_REGISTRY = {
    "ac_ao": AC_AO_Strategy,
    "adx": ADX_Strategy,
    "dem": DeM_Strategy,
    "fbb": FBB_Strategy,
    "mfi": MFI_Strategy,
    "ms": MS_Strategy,
    "mtf_stoch": QuadStochStrategy,
    "universal": UniversalStrategy,
}