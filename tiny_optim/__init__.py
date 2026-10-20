"""tiny-optim: first-order optimizers from scratch, pure Python."""

from .optimizers import Adam, Momentum, SGD

__all__ = ["SGD", "Momentum", "Adam"]
__version__ = "0.1.0"
