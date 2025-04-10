"""
test_absorption_model.py - tests for AbsorptionModel

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

from bayes_spec import SpecData
from caribou_hi import AbsorptionModel


def test_absorption_model():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "absorption": SpecData(velocity, brightness, 1.0),
    }
    model = AbsorptionModel(data, 2, baseline_degree=1)
    model.add_priors()
    model.add_likelihood()
    assert model._validate()


def test_emission_model_ordered():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "absorption": SpecData(velocity, brightness, 1.0),
    }
    model = AbsorptionModel(data, 2, baseline_degree=1)
    model.add_priors(ordered=True)
    model.add_likelihood()
    assert model._validate()
