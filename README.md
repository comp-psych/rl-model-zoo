# RL Model Zoo 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A collection of reinforcement learning models used in **computational psychiatry** research. Each model includes simulation, parameter recovery via maximum likelihood estimation, and documented clinical relevance.

Part of the [comp-psych-org](https://github.com/comp-psych-org) initiative.

---

## Model Comparison

| Model | Parameters | Clinical Application | Complexity |
|-------|-----------|---------------------|------------|
| **Rescorla-Wagner** | α (learning rate) | Depression (blunted α), Addiction (elevated α) | ⭐ |
| **TD Learning** | α, γ (discount) | Dopamine prediction errors, Addiction | ⭐⭐ |
| **Q-Learning + Softmax** | α, γ, β (inv. temp.) | Explore-exploit in Anxiety, Schizophrenia | ⭐⭐ |
| **Hierarchical Gaussian Filter** | ω (tonic vol.), κ (coupling) | Psychosis (aberrant precision), Anxiety | ⭐⭐⭐ |
| **Dual Learning Rate** | α⁺, α⁻, β | Depression (blunted α⁺), Mania (enhanced α⁺) | ⭐⭐ |

## Installation

```bash
# Clone the repository
git clone https://github.com/comp-psych-org/rl-model-zoo.git
cd rl-model-zoo

# Install with pip
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.21
- SciPy >= 1.7
- Matplotlib >= 3.4
- pytest >= 7.0 (dev)

## Quick Start

### Rescorla-Wagner Model

```python
from models.rescorla_wagner import RescorlaWagner
import numpy as np

# Create model with known learning rate
model = RescorlaWagner(alpha=0.4)

# Simulate a binary reward task
data = model.simulate(n_trials=300, reward_prob=0.7, rng=np.random.RandomState(42))

# Recover parameters from behavioral data
result = model.fit(data["choices"], data["rewards"])
print(f"Recovered alpha: {result['alpha']:.3f}")
```

### Q-Learning with Softmax

```python
from models.q_learning import QLearning
import numpy as np

# Two-armed bandit with drifting rewards
model = QLearning(alpha=0.3, gamma=0.95, beta=5.0)
data = model.simulate(n_trials=400, rng=np.random.RandomState(42))

# Fit to recover alpha and beta
result = model.fit(data["choices"], data["rewards"])
print(f"Recovered alpha: {result['alpha']:.3f}, beta: {result['beta']:.3f}")
```

### Hierarchical Gaussian Filter

```python
from models.hgf import HierarchicalGaussianFilter
import numpy as np

# Volatile binary outcome task
model = HierarchicalGaussianFilter(omega=-3.0, kappa=1.0)
data = model.simulate(n_trials=400, volatile=True, rng=np.random.RandomState(42))

# Recover tonic volatility and coupling strength
result = model.fit(data["outcomes"], data["choices"])
print(f"Recovered omega: {result['omega']:.3f}, kappa: {result['kappa']:.3f}")
```

### Dual Learning Rate

```python
from models.dual_lr import DualLearningRate
import numpy as np

# Asymmetric learning from positive vs negative prediction errors
model = DualLearningRate(alpha_pos=0.6, alpha_neg=0.3, beta=5.0)
data = model.simulate(n_trials=400, reward_prob=0.7, rng=np.random.RandomState(42))

result = model.fit(data["choices"], data["rewards"])
print(f"alpha+: {result['alpha_pos']:.3f}, alpha-: {result['alpha_neg']:.3f}")
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests for a specific model
python -m pytest tests/test_rescorla_wagner.py -v
```

## Running Examples

```bash
python examples/example_rescorla_wagner.py
python examples/example_td_learning.py
python examples/example_q_learning.py
python examples/example_hgf.py
python examples/example_dual_lr.py
```

## Project Structure

```
rl-model-zoo/
├── models/               # Model implementations
│   ├── rescorla_wagner.py
│   ├── td_learning.py
│   ├── q_learning.py
│   ├── hgf.py
│   └── dual_lr.py
├── tests/                # Unit tests with parameter recovery
├── examples/             # Usage examples
├── pyproject.toml
├── LICENSE
└── CONTRIBUTING.md
```

## Related Resources

- [comp-psych-org](https://github.com/comp-psych-org) — Organization for open-source computational psychiatry tools
- [Computational Psychiatry Syllabus](https://github.com/comp-psych-org/syllabus) — Course materials and reading lists

## Key References

- Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- Mathys, C. D., et al. (2014). Uncertainty in perception and the HGF. *Frontiers in Human Neuroscience*.
- Daw, N. D., et al. (2006). Cortical substrates for exploratory decisions. *Nature*.
- Frank, M. J., et al. (2007). Genetic triple dissociation. *PNAS*.

## Credits

Built by [Peter Zhou](https://github.com/peterzhou) as part of the **PRAXIS** computational psychiatry research program.

## License

MIT License — see [LICENSE](LICENSE) for details.
