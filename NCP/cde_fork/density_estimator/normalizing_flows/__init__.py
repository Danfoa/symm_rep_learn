from .AffineFlow import AffineFlow
from .IdentityFlow import IdentityFlow
from .PlanarFlow import InvertedPlanarFlow
from .RadialFlow import InvertedRadialFlow

FLOWS = {
    'planar': InvertedPlanarFlow,
    'radial': InvertedRadialFlow,
    'identity': IdentityFlow,
    'affine': AffineFlow
}

