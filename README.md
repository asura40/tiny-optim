# tiny-optim

A tiny, dependency-free playground for optimizers, written from scratch in
pure Python: SGD, momentum, and Adam over flat parameter lists, plus Muon
for 2D weight matrices.

No numpy, no torch — just lists and the standard library, so the update rules
are easy to read end to end.

## Run the demo

```bash
python -m tiny_optim.demo
```

Fits `y = 3x + 2` on synthetic data with each flat optimizer and prints the
final loss and recovered parameters, then fits a small matrix regression
(learn a 4x3 weight matrix) with Muon vs flattened momentum.

## Run the tests

```bash
python -m unittest discover -s tests
```

## Layout

- `tiny_optim/optimizers.py` — `SGD`, `Momentum`, and `Adam` operating on
  flat lists of floats; `Muon` and `newton_schulz5` operating on matrices
  (lists of lists).
- `tiny_optim/demo.py` — least-squares demo comparing the flat optimizers,
  plus the matrix-regression comparison.
- `tests/test_optimizers.py` — sanity checks: loss decreases on a quadratic,
  Adam recovers the minimum, a zero gradient leaves parameters unchanged,
  Newton-Schulz orthogonalizes a matrix, Muon fits the matrix regression.

## Muon

Muon (Jordan et al., 2024) is SGD with momentum whose update, for each 2D
weight matrix, is first orthogonalized by a Newton-Schulz iteration — every
singular direction of the update then moves at the same rate. The
implementation here mirrors the reference defaults (lr 0.02, momentum 0.95,
Nesterov) and uses the quintic iteration coefficients from Keller Jordan's
reference implementation: https://github.com/KellerJordan/Muon

## License

MIT
