"""
physics.py
Physics utilities

Copyright(C) 2024 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Changelog:
Trey Wenger - July 2024
"""

from typing import Iterable

import numpy as np
import pytensor.tensor as pt

from bayes_spec.utils import gaussian


def calc_spin_temp(kinetic_temp: float, density: float, n_alpha: float) -> float:
    """Calculate the spin temperature following Kim et al. (2014) eq. 4

    Parameters
    ----------
    kinetic_temp : float
        Kinetic temperature (K)
    density : float
        HI volume density (cm-3)
    n_alpha : float
        Lyman alpha photon density (cm-3)

    Returns
    -------
    float
        Spin temperature (K)
    """
    # CMB + background Galactic synchrotron emission
    TR = 3.77  # K

    # Lya photon transition probability (Field 1958)
    # assuming T_alpha = T_kinetic
    y_alpha = 5.9e11 * n_alpha / kinetic_temp**1.5

    # T0 = h * nu / k_B
    T0 = 0.0681  # K

    # Einstein A
    A10 = 2.8843e-15  # s-1

    # T2 = T_kinetic / 100
    T2 = kinetic_temp / 100.0

    # collisional de-excitation rate coefficient (Draine 2011)
    k10 = pt.switch(
        pt.lt(kinetic_temp, 300.0),  # K
        1.19e-10 * T2 ** (0.74 - 0.20 * pt.log(T2)),
        2.24e-10 * T2**0.207 * pt.exp(-0.876 / T2),
    )  # cm-3 s-1

    # de-excitation rate
    R10 = density * k10

    # collisional de-excitation probability
    y_c = T0 * R10 / (kinetic_temp * A10)

    # spin temperature (K)
    spin_temp = (TR + y_c * kinetic_temp + y_alpha * kinetic_temp) / (1 + y_c + y_alpha)
    return spin_temp


def calc_thermal_fwhm(kinetic_temp: float) -> float:
    """Calculate the thermal line broadening assuming a Maxwellian velocity distribution
    (Condon & Ransom eq. 7.35)

    Parameters
    ----------
    kinetic_temp : float
        Kinetic temperature (K)

    Returns
    -------
    float
        Thermal FWHM line width (km s-1)
    """
    const = 0.2139  # km/s K-1/2
    return const * pt.sqrt(kinetic_temp)


def calc_depth(NHI: float, nHI: float) -> float:
    """Calculate the line-of-sight depth assuming a homogenous medium.

    Parameters
    ----------
    NHI : float
        HI column density (cm-2)
    nHI : float
        HI density (cm-3)

    Returns
    -------
    float
        Line-of-sight depth (pc)
    """
    CM_PER_PC = 3.085677581491367e18
    return NHI / nHI / CM_PER_PC


def calc_nonthermal_fwhm(depth: float, larson_linewidth: float, larson_power: float) -> float:
    """Calculate the non-thermal line broadening assuming a Larson law relationship

    Parameters
    ----------
    depth : float
        Line-of-sight depth (pc)
    larson_linewidth : float
        Larson non-thermal broadening at 1 pc (km s-1)
    larson_power : float
        Larson exponent

    Returns
    -------
    float
        Non-thermal FWHM line width (km s-1)
    """
    return larson_linewidth * depth**larson_power


def calc_line_profile(velo_axis: Iterable[float], velocity: Iterable[float], fwhm: Iterable[float]) -> Iterable[float]:
    """Evaluate the Gaussian line profile, ensuring normalization.

    Parameters
    ----------
    velo_axis : Iterable[float]
        Observed velocity axis (km s-1; length S)
    velocity : Iterable[float]
        Cloud center velocity (km s-1; length C x N)
    fwhm : Iterable[float]
        Cloud FWHM line widths (km s-1; length C x N)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x N)
    """
    amp = pt.sqrt(4.0 * pt.log(2.0) / (np.pi * fwhm**2.0))
    profile = gaussian(velo_axis[:, None], amp, velocity, fwhm)

    # normalize
    channel_size = pt.abs(velo_axis[1] - velo_axis[0])
    profile_int = pt.sum(profile, axis=0)
    norm = pt.switch(pt.lt(profile_int, 1.0e-6), 1.0, profile_int * channel_size)
    return profile / norm


def calc_optical_depth(
    velo_axis: Iterable[float],
    velocity: Iterable[float],
    NHI: Iterable[float],
    tspin: Iterable[float],
    fwhm: Iterable[float],
) -> Iterable[float]:
    """Evaluate the optical depth spectra following Marchal et al. (2019) eq. 15
    assuming a homogeneous and isothermal cloud.

    Parameters
    ----------
    velo_axis : Iterable[float]
        Observed velocity axis (km s-1) (length S)
    velocity : Iterable[float]
        Cloud velocities (km s-1) (length N)
    NHI : Iterable[float]
        HI column density (cm-2) (length N)
    tspin : Iterable[float]
        Spin tempearture (K) (length N)
    fwhm : Iterable[float]
        FWHM line width (km s-1)

    Returns
    -------
    Iterable[float]
        Optical depth spectra (shape S x N)
    """
    # Evaluate line profile
    line_profile = calc_line_profile(velo_axis, velocity, fwhm)

    # Evaluate the optical depth spectra
    const = 1.82243e18  # cm-2 (K km s-1)-1
    optical_depth = NHI * line_profile / tspin / const
    return optical_depth


def radiative_transfer(
    tau: Iterable[float],
    tspin: Iterable[float],
    bg_temp: float,
) -> Iterable[float]:
    """Evaluate the radiative transfer to predict the emission spectrum. The emission
    spectrum is ON - OFF, where ON includes the attenuated emission of the background and
    the clouds, and the OFF is the emission of the background. Order of N clouds is
    assumed to be [nearest, ..., farthest].

    Parameters
    ----------
    tau : Iterable[float]
        Optical depth spectra (shape S x N)
    tspin : Iterable[float]
        Spin temperatures (K) (length N)
    bg_temp : float
        Assumed background temperature

    Returns
    -------
    Iterable[float]
        Predicted emission brightness temperature spectrum (K) (length S)
    """
    front_tau = pt.zeros_like(tau[:, 0:1])
    # cumulative optical depth through clouds
    sum_tau = pt.concatenate([front_tau, pt.cumsum(tau, axis=1)], axis=1)

    # radiative transfer, assuming filling factor = 1.0
    emission_bg_attenuated = bg_temp * pt.exp(-sum_tau[:, -1])
    emission_clouds = tspin * (1.0 - pt.exp(-tau))
    emission_clouds_attenuated = emission_clouds * pt.exp(-sum_tau[:, :-1])
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=1)

    # ON - OFF
    return emission - bg_temp
