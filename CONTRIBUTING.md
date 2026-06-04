# Contributing to RL Model Zoo

Thank you for your interest in contributing to the RL Model Zoo! This project
is part of the comp-psych-org initiative for open-source computational
psychiatry tools.

## How to Contribute

### Adding a New Model

1. **Create the model file** in `models/`:
   - Follow the existing file structure
   - Include a class with `simulate()` and `fit()` methods
   - Add comprehensive numpy-style docstrings
   - Include mathematical formulation and clinical relevance

2. **Write tests** in `tests/`:
   - Test that simulation runs without errors
   - Test parameter recovery (simulate → fit → check within 20%)
   - Test edge cases (invalid parameters, extreme values)
   - Use random seeds for reproducibility

3. **Add an example** in `examples/`:
   - Show basic simulation and parameter recovery
   - Include clear print statements showing results

4. **Update `models/__init__.py`** to export the new model

### Code Standards

- Use type hints for all function signatures
- Follow numpy docstring conventions
- Use `scipy.optimize.minimize` for parameter fitting
- Support `np.random.RandomState` for reproducibility
- Minimum 200 trials for simulations, 10+ restarts for fitting

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests for a specific model
python -m pytest tests/test_rescorla_wagner.py -v
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Ensure all tests pass (`python -m pytest tests/ -v`)
4. Submit a pull request with a clear description

### Model Requirements Checklist

- [ ] Model class with clear docstring
- [ ] `simulate()` method generates synthetic data
- [ ] `fit()` method recovers parameters via MLE
- [ ] Type hints on all functions
- [ ] numpy-style docstrings
- [ ] Test file with simulation, recovery, and edge case tests
- [ ] Example file demonstrating usage
- [ ] Parameter recovery within 20% tolerance

## Reporting Issues

Please open an issue on GitHub with:
- A clear description of the bug or feature request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.
