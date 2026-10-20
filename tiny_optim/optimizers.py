"""First-order optimizers from scratch.

``SGD``, ``Momentum``, and ``Adam`` operate on flat lists of floats.
``Muon`` operates on 2D weight matrices (lists of lists): it orthogonalizes
the momentum update with a Newton-Schulz iteration before applying it.

Each optimizer keeps its own state and exposes a single ``step(params, grads)``
method that returns the updated parameters.
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


# ---------------- matrix helpers (for Muon) ----------------

def _transpose(m):
    return [list(col) for col in zip(*m)]


def _matmul(a, b):
    bt = _transpose(b)
    return [[sum(x * y for x, y in zip(row_a, col_b)) for col_b in bt] for row_a in a]


def _frobenius(m):
    return sum(x * x for row in m for x in row) ** 0.5


def newton_schulz5(m, steps=5):
    """Orthogonalize a 2D matrix (list of lists) via Newton-Schulz iteration.

    Quintic iteration with the coefficients from Keller Jordan's Muon
    reference implementation: it pushes every singular value of ``m`` into a
    band around 1, returning a near-semi-orthogonal matrix. (The coefficients
    are tuned for a fixed small step count — 5 by default — not for exact
    convergence under unlimited iteration.) Runs the iteration on the
    smaller dimension by transposing when the matrix is tall.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = [row[:] for row in m]
    transposed = False
    if len(x) > len(x[0]):
        x = _transpose(x)
        transposed = True
    norm = _frobenius(x) + 1e-7
    x = [[v / norm for v in row] for row in x]
    for _ in range(steps):
        gram = _matmul(x, _transpose(x))            # A = X X^T
        gram2 = _matmul(gram, gram)                 # A^2
        # B = b*A + c*A^2, then X <- a*X + B X
        bx = _matmul(
            [[b * gram[i][j] + c * gram2[i][j] for j in range(len(gram))]
             for i in range(len(gram))],
            x,
        )
        x = [[a * x[i][j] + bx[i][j] for j in range(len(x[0]))]
             for i in range(len(x))]
    if transposed:
        x = _transpose(x)
    return x


class Muon:
    """Muon (Jordan et al., 2024): MomentUm Orthogonalized by Newton-Schulz.

    For 2D parameters only — params and grads are lists of lists with the
    same shape. The momentum buffer is orthogonalized with
    :func:`newton_schulz5` before the update is applied, so every singular
    direction of the update moves at the same rate. Mirrors the reference
    defaults: lr=0.02, momentum=0.95, Nesterov on.
    """

    def __init__(self, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self._buf = None

    def step(self, params, grads):
        if self._buf is None:
            self._buf = [[0.0] * len(row) for row in params]
        mu = self.momentum
        self._buf = [
            [mu * buf + g for buf, g in zip(buf_row, g_row)]
            for buf_row, g_row in zip(self._buf, grads)
        ]
        if self.nesterov:
            update = [
                [g + mu * buf for buf, g in zip(buf_row, g_row)]
                for buf_row, g_row in zip(self._buf, grads)
            ]
        else:
            update = [row[:] for row in self._buf]
        update = newton_schulz5(update, steps=self.ns_steps)
        return [
            [p - self.lr * u for p, u in zip(p_row, u_row)]
            for p_row, u_row in zip(params, update)
        ]
