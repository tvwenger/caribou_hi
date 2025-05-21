"""
physics.py
Physics utilities

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import numpy as np
import pytensor.tensor as pt


def gaussian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Gaussian function

    Parameters
    ----------
    x : float
        Position at which to evaluate
    center : float
        Gaussian centroid
    fwhm : float
        Gaussian FWHM line width

    Returns
    -------
    float
        Gaussian evaluated at x
    """
    return pt.exp(-4.0 * np.log(2.0) * (x - center) ** 2.0 / fwhm**2.0) * pt.sqrt(
        4.0 * np.log(2.0) / (np.pi * fwhm**2.0)
    )


def lorentzian(x: float, center: float, fwhm: float) -> float:
    """Evaluate a normalized Lorentzian function

    Parameters
    ----------
    x : float
        Position at which to evaluate
    center : float
        Centroid
    fwhm : float
        FWHM

    Returns
    -------
    float
        Lorentzian evaluated at x
    """
    return fwhm / (2.0 * np.pi) / ((x - center) ** 2.0 + (fwhm / 2.0) ** 2.0)


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


def calc_thermal_fwhm2(kinetic_temp: float) -> float:
    """Calculate the thermal line broadening assuming a Maxwellian velocity distribution
    (Condon & Ransom eq. 7.35)

    Parameters
    ----------
    kinetic_temp : float
        Kinetic temperature (K)

    Returns
    -------
    float
        Thermal FWHM^2 line width (km2 s-2)
    """
    const = 0.2139  # km/s K-1/2
    return const**2 * kinetic_temp


def calc_kinetic_temp(thermal_fwhm2: float) -> float:
    """Calculate the kinetic temperature from the thermal line broadening. This
    is the inverse of calc_thermal_fwhm.
    (Condon & Ransom eq. 7.35)

    Parameters
    ----------
    thermal_fwhm2 : float
        Thermal FWHM^2 (km2 s-2)

    Returns
    -------
    float
        Kinetic temperature (K)
    """
    const = 0.2139  # km/s K-1/2
    return thermal_fwhm2 / const**2


def calc_nonthermal_fwhm(
    depth: float, nth_fwhm_1pc: float, depth_nth_fwhm_power: float
) -> float:
    """Calculate the non-thermal line broadening assuming a size-linewidth relationship

    Parameters
    ----------
    depth : float
        Line-of-sight depth (pc)
    nth_fwhm_1pc : float
        Non-thermal line width at 1 pc(km s-1)
    depth_nth_fwhm_power : float
        Depth vs. non-thermal line width power law index

    Returns
    -------
    float
        Non-thermal FWHM line width (km s-1)
    """
    return nth_fwhm_1pc * depth**depth_nth_fwhm_power


def calc_depth_nonthermal(
    fwhm_nonthermal: float, nth_fwhm_1pc: float, depth_nth_fwhm_power: float
) -> float:
    """Calculate the line-of-sight depth from the non-thermal line broadening. This is the
    inverse of calc_nonthermal_fwhm.

    Parameters
    ----------
    fwhm_nonthermal : float
        Non-thermal FWHM line width (km s-1)
    nth_fwhm_1pc : float
        Non-thermal line width at 1 pc(km s-1)
    depth_nth_fwhm_power : float
        Depth vs. non-thermal line width power law index

    Returns
    -------
    float
        Line-of-sight depth (pc)
    """
    return (fwhm_nonthermal / nth_fwhm_1pc) ** (1.0 / depth_nth_fwhm_power)


def calc_log10_column_density(tau_total: float, spin_temp: float) -> float:
    """Calculate the column density.

    Parameters
    ----------
    tau_total : float
        Total optical depth
    spin_temp : float
        Spin temperature (K)

    Returns
    -------
    float
        log10 column density (cm-2)
    """
    const = 1.82243e18  # cm-2 (K km s-1)-1
    return pt.log10(const * spin_temp * tau_total)


def calc_log10_density(log10_column_density: float, log10_depth: float):
    """Calculate the density.

    Parameters
    ----------
    log10_column_density : float
        log10 column density (cm-2)
    log10_depth : float
        log10 depth (pc)

    Returns
    -------
    float
        log10 volume density (cm-3)
    """
    return log10_column_density - log10_depth - 18.489351


def calc_tau_total(column_density: float, spin_temp: float) -> float:
    """Calculate the total optical depth.

    Parameters
    ----------
    column_density : float
        Column density (cm-2)
    spin_temp : float
        Spin temperature (K)

    Returns
    -------
    float
        total optical depth
    """
    const = 1.82243e18  # cm-2 (K km s-1)-1
    return column_density / spin_temp / const


def calc_log10_nonthermal_pressure(log10_density, fwhm2_nonthermal):
    """Calculate the non-thermal pressure.

    Parameters
    ----------
    log10_column_density : float
        log10 column density (cm-2)
    fwhm2_nonthermal : float
        Non-thermal FWHM^2 (km2 s-2)

    Returns
    -------
    float
        log10 non-thermal pressure (K cm-3)
    """
    return np.log10(671.8) + log10_density + pt.log10(fwhm2_nonthermal)


def calc_pseudo_voigt(
    velo_axis: Iterable[float],
    velocity: Iterable[float],
    fwhm2: Iterable[float],
    fwhm_L: float,
) -> Iterable[float]:
    """Evaluate a pseudo Voight profile in order to aid in posterior exploration
    of the parameter space. This parameterization includes a latent variable fwhm_L, which
    can be conditioned on zero to analyze the posterior. We also consider the spectral
    channelization. We do not perform a full boxcar convolution, rather
    we approximate the convolution by assuming an equivalent FWHM for the
    boxcar kernel of 4 ln(2) / pi * channel_width ~= 0.88 * channel_width

    Parameters
    ----------
    velo_axis : Iterable[float]
        Observed velocity axis (km s-1; length S)
    velocity : Iterable[float]
        Cloud center velocity (km s-1; length N)
    fwhm2 : Iterable[float]
        Cloud FWHM^2 line widths (km2 s-2; length N)
    fwhm_L : float
        Latent pseudo-Voigt profile Lorentzian FWHM (km s-1)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x N)
    """
    channel_size = pt.abs(velo_axis[1] - velo_axis[0])
    channel_fwhm = 4.0 * np.log(2.0) * channel_size / np.pi
    fwhm_conv = pt.sqrt(fwhm2 + channel_fwhm**2.0 + fwhm_L**2.0)
    fwhm_L_frac = fwhm_L / fwhm_conv
    eta = (
        1.36603 * fwhm_L_frac - 0.47719 * fwhm_L_frac**2.0 + 0.11116 * fwhm_L_frac**3.0
    )

    # gaussian component
    gauss_part = gaussian(velo_axis[:, None], velocity, fwhm_conv)

    # lorentzian component
    lorentz_part = lorentzian(velo_axis[:, None], velocity, fwhm_conv)

    # linear combination
    return eta * lorentz_part + (1.0 - eta) * gauss_part


def radiative_transfer(
    tau: Iterable[float],
    tspin: Iterable[float],
    filling_factor: Iterable[float],
    bg_temp: float,
) -> Iterable[float]:
    """Evaluate the radiative transfer to predict the emission spectrum. The emission
    spectrum is ON - OFF, where ON includes the attenuated emission of the background and
    the clouds, and the OFF is the emission of the background. Order of N clouds is
    assumed to be [nearest, ..., farthest]. The contribution of each cloud is diluted by the
    filling factor, a number between zero and one.

    Parameters
    ----------
    tau : Iterable[float]
        Optical depth spectra (shape S x N)
    tspin : Iterable[float]
        Spin temperatures (K) (shape N)
    filling_factor : Iterable[float]
        Filling factor (between zero and one) (shape N)
    bg_temp : float
        Assumed background temperature

    Returns
    -------
    Iterable[float]
        Predicted emission brightness temperature spectrum (K) (length S)
    """
    # Attenuation factors (shape S, N+1)
    # This is the attenuation due to clouds between us and cloud N
    # [[1.0]*S, atten[0], atten[0]*atten[1], ..., prod(atten)]
    # where atten[i] = 1.0 - filling_factor[i] + filling_factor[i]*exp(tau[i])
    # This is a geometric mean weighted by the filling factors
    attenuation = pt.concatenate(
        [
            pt.ones_like(tau[:, 0:1]),
            pt.cumprod(1.0 - filling_factor + filling_factor * pt.exp(-tau), axis=1),
        ],
        axis=1,
    )

    # Background is attenuated by all clouds (shape S)
    emission_bg_attenuated = bg_temp * attenuation[..., -1]

    # Emission of each cloud (shape S, N)
    emission_clouds = filling_factor * tspin * (1.0 - pt.exp(-tau))

    # Attenuation by foreground clouds (shape S, N)
    # [TB(N=0), TB(N=1)*exp(-tau(N=0)), TB(N=2)*exp(-tau(N=0)-tau(N=1)), ...]
    emission_clouds_attenuated = emission_clouds * attenuation[..., :-1]
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=1)

    # ON - OFF
    return emission - bg_temp
