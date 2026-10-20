"""First-order optimizers over flat lists of floats, from scratch.

Each optimizer keeps its own state and exposes a single ``step(params, grads)``
method that returns the updated parameters. Parameters and gradients are
plain Python lists of equal length.
"""


class SGD:
    """Vanilla gradient descent: p <- p - lr * g."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


class Momentum:
    """Heavy-ball momentum: v <- mu * v - lr * g; p <- p + v."""

    def __init__(self, lr=0.01, mu=0.9):
        self.lr = lr
        self.mu = mu
        self._v = None

    def step(self, params, grads):
        if self._v is None:
            self._v = [0.0] * len(params)
        self._v = [self.mu * v - self.lr * g for v, g in zip(self._v, grads)]
        return [p + v for p, v in zip(params, self._v)]


class Adam:
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params, grads):
        if self._m is None:
            self._m = [0.0] * len(params)
            self._v = [0.0] * len(params)
        self._t += 1
        b1, b2 = self.beta1, self.beta2
        self._m = [b1 * m + (1 - b1) * g for m, g in zip(self._m, grads)]
        self._v = [b2 * v + (1 - b2) * g * g for v, g in zip(self._v, grads)]
        m_hat = [m / (1 - b1 ** self._t) for m in self._m]
        v_hat = [v / (1 - b2 ** self._t) for v in self._v]
        return [
            p - self.lr * mh / (vh ** 0.5 + self.eps)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]
