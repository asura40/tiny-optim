"""tiny-optim: first-order optimizers from scratch, pure Python."""

from .optimizers import Adam, Momentum, Muon, SGD, newton_schulz5

__all__ = ["SGD", "Momentum", "Adam", "Muon", "newton_schulz5"]
__version__ = "0.2.0"
