"""
test_hi_model.py - tests for models based on HIModel

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest
import numpy as np

from bayes_spec import SpecData
from caribou_hi.hi_model import HIModel
from caribou_hi import EmissionModel, AbsorptionModel, EmissionAbsorptionModel


def test_hi_model():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {"emission": SpecData(velocity, brightness, 1.0)}
    with pytest.raises(TypeError):
        _ = HIModel(data, n_clouds=1)


def test_emission_model():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {
        "emission": SpecData(velocity, brightness, 1.0),
    }
    model = EmissionModel(data, 2, baseline_degree=1)
    model.add_priors(prior_fwhm_L=1.0)
    model.add_likelihood()
    assert model._validate()


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


def test_emission_absorption_model_lognormal():
    emission_velocity = np.linspace(-20.0, 20.0, 1000)
    absorption_velocity = np.linspace(-20.0, 20.0, 500)
    emission = np.random.randn(1000)
    absorption = np.random.randn(500)
    data = {
        "emission": SpecData(emission_velocity, emission, 1.0),
        "absorption": SpecData(absorption_velocity, absorption, 1.0),
    }
    model = EmissionAbsorptionModel(data, 2, baseline_degree=1)
    model.add_priors(prior_sigma_log10_NHI=0.5)
    model.add_likelihood()
    assert model._validate()
