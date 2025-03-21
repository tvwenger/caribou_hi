"""
test_emission_absorption_model.py - tests for EmissionAbsorptionModel

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

from bayes_spec import SpecData
from caribou_hi import EmissionAbsorptionModel


def test_emission_absorption_model():
    emission_velocity = np.linspace(-20.0, 20.0, 1000)
    absorption_velocity = np.linspace(-20.0, 20.0, 500)
    emission = np.random.randn(1000)
    absorption = np.random.randn(500)
    data = {
        "emission": SpecData(emission_velocity, emission, 1.0),
        "absorption": SpecData(absorption_velocity, absorption, 1.0),
    }
    model = EmissionAbsorptionModel(data, 2, baseline_degree=1)
    model.add_priors()
    model.add_likelihood()
    assert model._validate()


def test_emission_absorption_model_ordered():
    emission_velocity = np.linspace(-20.0, 20.0, 1000)
    absorption_velocity = np.linspace(-20.0, 20.0, 500)
    emission = np.random.randn(1000)
    absorption = np.random.randn(500)
    data = {
        "emission": SpecData(emission_velocity, emission, 1.0),
        "absorption": SpecData(absorption_velocity, absorption, 1.0),
    }
    model = EmissionAbsorptionModel(data, 2, baseline_degree=1)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()
