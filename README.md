# tiny-optim

A tiny, dependency-free playground for first-order optimizers, written from
scratch in pure Python: SGD, momentum, and Adam, plus a small demo that fits
a line with each of them.

No numpy, no torch — just lists and the standard library, so the update rules
are easy to read end to end.

## Run the demo

```bash
python -m tiny_optim.demo
```

Fits `y = 3x + 2` on synthetic data with each optimizer and prints the final
loss and the recovered parameters.

## Run the tests

```bash
python -m unittest discover -s tests
```

## Layout

- `tiny_optim/optimizers.py` — `SGD`, `Momentum`, and `Adam` operating on
  flat lists of floats.
- `tiny_optim/demo.py` — least-squares demo comparing the three.
- `tests/test_optimizers.py` — sanity checks: loss decreases on a quadratic,
  Adam recovers the minimum, a zero gradient leaves parameters unchanged.

## License

MIT
