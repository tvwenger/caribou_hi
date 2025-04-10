"""
test_physics.py - tests for physics.py

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from caribou_hi import physics


def test_calc_spin_temp():
    kinetic_temp = 1000.0
    density = 1000.0
    n_alpha = 1.0e-6
    spin_temp = physics.calc_spin_temp(kinetic_temp, density, n_alpha).eval()
    assert spin_temp == pytest.approx(kinetic_temp, abs=1.0)
    kinetic_temp = 8000.0
    density = 1.0
    n_alpha = 1.0e-6
    spin_temp = physics.calc_spin_temp(kinetic_temp, density, n_alpha).eval()
    assert spin_temp < kinetic_temp


def test_calc_thermal_fwhm():
    kinetic_temp = 1000.0
    thermal_fwhm = physics.calc_thermal_fwhm(kinetic_temp).eval()
    assert 0.2139 * np.sqrt(kinetic_temp) == pytest.approx(thermal_fwhm)


def test_calc_nonthermal_fwhm():
    depth = 2.0
    nth_fwhm_1pc = 2.0
    depth_nth_fwhm_power = 2.0
    nonthermal_fwhm = physics.calc_nonthermal_fwhm(
        depth, nth_fwhm_1pc, depth_nth_fwhm_power
    )
    assert 8.0 == pytest.approx(nonthermal_fwhm)


def test_calc_line_profile():
    velo_axis = np.linspace(-10.0, 10.0, 1001)
    velocity = np.array([0.0, 1.0])
    fwhm = np.array([1.0, 2.0])
    line_profile = physics.calc_line_profile(velo_axis, velocity, fwhm).eval()
    assert line_profile.shape == (1001, 2)
    assert_allclose(
        line_profile.sum(axis=0) * (velo_axis[1] - velo_axis[0]), np.ones(2)
    )
    exp_line_profile = np.array(
        [
            np.sqrt(4.0 * np.log(2.0) / np.pi),
            0.5 * np.sqrt(np.log(2.0) / np.pi),
        ]
    )
    assert_allclose(line_profile[500, :], exp_line_profile, rtol=0.001, atol=0.001)


def test_calc_optical_depth():
    velo_axis = np.linspace(-10.0, 10.0, 1001)
    velocity = np.array([0.0, 1.0])
    fwhm = np.array([10.0, 20.0])
    NHI = np.array([1.0e20, 1.0e21])
    tspin = np.array([1000.0, 5000.0])
    fwhm_L = 1.0
    optical_depth = physics.calc_optical_depth(
        velo_axis, velocity, NHI, tspin, fwhm, fwhm_L
    ).eval()
    assert optical_depth.shape == (1001, 2)
    assert np.all(optical_depth >= 0)


def test_radiative_transfer():
    velo_axis = np.linspace(-10.0, 10.0, 1001)
    velocity = np.array([0.0, 1.0])
    fwhm = np.array([10.0, 20.0])
    NHI = np.array([1.0e20, 1.0e21])
    tspin = np.array([1000.0, 5000.0])
    fwhm_L = 1.0
    optical_depth = physics.calc_optical_depth(
        velo_axis, velocity, NHI, tspin, fwhm, fwhm_L
    ).eval()
    bg_temp = 2.7
    filling_factor = 1.0
    tb = physics.radiative_transfer(
        optical_depth, tspin, filling_factor, bg_temp
    ).eval()
    assert tb.shape == (1001,)
