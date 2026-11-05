"""
models/__init__.py
------------------
Exposes all four models at the package level.
"""

from .deepjscc    import PlainDeepJSCC        # Model 1
from .resnet_jscc  import ResNetJSCC           # Model 2
from .resnet_variants import (
    ResNetJSCC_2b,                             # Model 2b (Control: Reg Layer-4)
    ResNetJSCC_2c,                             # Model 2c (Layer-3 + Conv proj)
    ResNetJSCC_2d,                             # Model 2d (Layer-3 unfrozen)
    ResNetJSCC_2e,                             # Model 2e (128x128 upsample + Layer-4)
)
# Stage 3:
from .swin_refine  import SwinRefineJSCC      # Model 3 (SwinIR + FiLM / NoSNR / CNN)

# Stage 4:
from .swin_jscc    import SwinJSCC            # Model 4
