"""
test_physics.py - tests for physics.py

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pytest
import numpy as np

from caribou_hi import physics


def test_gaussian():
    x = 1.0
    center = 0.0
    fwhm = 1.0
    gauss = physics.gaussian(x, center, fwhm).eval()
    val = np.exp(-4.0 * np.log(2.0)) * np.sqrt(4.0 * np.log(2.0) / np.pi)
    assert pytest.approx(gauss) == val


def test_lorentzian():
    x = 1.0
    center = 0.0
    fwhm = 1.0
    lorent = physics.lorentzian(x, center, fwhm)
    val = 1.0 / (2.0 * np.pi) / 1.25
    assert pytest.approx(lorent) == val


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


def test_calc_thermal_fwhm2():
    kinetic_temp = 1000.0
    thermal_fwhm2 = physics.calc_thermal_fwhm2(kinetic_temp)
    assert 0.2139**2.0 * kinetic_temp == pytest.approx(thermal_fwhm2)


def test_calc_nonthermal_fwhm():
    depth = 2.0
    nth_fwhm_1pc = 2.0
    depth_nth_fwhm_power = 2.0
    nonthermal_fwhm = physics.calc_nonthermal_fwhm(
        depth, nth_fwhm_1pc, depth_nth_fwhm_power
    )
    assert 8.0 == pytest.approx(nonthermal_fwhm)


def test_calc_depth_nonthermal():
    fwhm_nonthermal = 8.0
    nth_fwhm_1pc = 2.0
    depth_nth_fwhm_power = 2.0
    depth = physics.calc_depth_nonthermal(
        fwhm_nonthermal, nth_fwhm_1pc, depth_nth_fwhm_power
    )
    assert 2.0 == pytest.approx(depth)


def test_calc_log10_column_density():
    tau_total = 1.0
    spin_temp = 1.0
    log10_NHI = physics.calc_log10_column_density(tau_total, spin_temp).eval()
    assert np.log10(1.82243e18) == pytest.approx(log10_NHI)


def test_calc_log10_density():
    log10_column_density = 18.489351
    log10_depth = 0.0
    log10_nHI = physics.calc_log10_density(log10_column_density, log10_depth)
    assert 0.0 == pytest.approx(log10_nHI)


def test_calc_tau_total():
    column_density = 1.82243e18
    spin_temp = 1.0
    tau_total = physics.calc_tau_total(column_density, spin_temp)
    assert 1.0 == pytest.approx(tau_total)


def test_calc_log10_nonthermal_pressure():
    log10_density = 0.0
    fwhm2_nonthermal = 1.0
    log10_Pnth = physics.calc_log10_nonthermal_pressure(
        log10_density, fwhm2_nonthermal
    ).eval()
    assert np.log10(671.8) == pytest.approx(log10_Pnth)


def test_calc_psuedo_voight():
    velo_axis = np.linspace(-100.0, 100.0, 101)
    velocity = np.array([-5.0, 5.0])
    fwhm2 = np.array([100.0, 200.0])
    fwhm_L = 0.0
    line_profile = physics.calc_pseudo_voigt(velo_axis, velocity, fwhm2, fwhm_L).eval()
    assert line_profile.shape == (101, 2)
    assert not np.any(np.isnan(line_profile))


def test_radiative_transfer():
    velo_axis = np.linspace(-100.0, 100.0, 101)
    velocity = np.array([-5.0, 5.0])
    fwhm2 = np.array([100.0, 200.0])
    fwhm_L = 0.0
    tau_total = np.array([5.0, 3.0])
    optical_depth = (
        tau_total * physics.calc_pseudo_voigt(velo_axis, velocity, fwhm2, fwhm_L).eval()
    )
    tspin = np.array([1000.0, 5000.0])
    fwhm_L = 1.0
    filling_factor = 1.0
    bg_temp = 2.7
    emission = physics.radiative_transfer(
        optical_depth, tspin, filling_factor, bg_temp
    ).eval()
    assert emission.shape == (101,)
    assert not np.any(np.isnan(emission))
