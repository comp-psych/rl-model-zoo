"""
RL Model Zoo — Reinforcement learning models for computational psychiatry.

This package provides implementations of common RL models used in
computational psychiatry research, including simulation and parameter
recovery functionality.
"""

from models.rescorla_wagner import RescorlaWagner
from models.td_learning import TDLearning
from models.q_learning import QLearning
from models.hgf import HierarchicalGaussianFilter
from models.dual_lr import DualLearningRate

__all__ = [
    "RescorlaWagner",
    "TDLearning",
    "QLearning",
    "HierarchicalGaussianFilter",
    "DualLearningRate",
]

__version__ = "0.1.0"
