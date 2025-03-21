"""
test_hi_model.py - tests for HIModel

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest
import numpy as np

from bayes_spec import SpecData
from caribou_hi.hi_model import HIModel


def test_hi_model():
    velocity = np.linspace(-20.0, 20.0, 1000)
    brightness = np.random.randn(1000)
    data = {"emission": SpecData(velocity, brightness, 1.0)}
    with pytest.raises(TypeError):
        _ = HIModel(data, n_clouds=1)
