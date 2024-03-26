"""
utils.py - Utility functions for physical calculations and radiative
transfer

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
Trey Wenger - March 2024
"""

from collections.abc import Iterable

import pytensor.tensor as pt


def gaussian(x: float, amp: float, center: float, fwhm: float) -> float:
    """
    Evaluate a Gaussian.

    Inputs:
        x :: scalar
            Position at which to evaluate
        amp, center, fwhm :: scalars
            Gaussian parameters
    Returns: y
        y :: scalar
            Evaluated Gaussian at x
    """
    return amp * pt.exp(-4.0 * pt.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)


def calc_log10_spin_temp(
    log10_kinetic_temp: float, log10_density: float, log10_n_alpha: float
) -> float:
    """
    Calculate the spin temperature following Kim et al. (2014) eq. 4.

    Inputs:
        log10_kinetic_temp :: 1-D array of scalars
            log10 kinetic temperature (K)
        log10_density :: 1-D array of scalars
            log10 density (cm-3)
        log10_n_alpha :: 1-D array of scalars
            log10 Ly-alpha photon density (cm-3)

    Returns:
        log10_spin_temp :: 1-D array of scalars
            log10 spin temperature (K)
    """
    log10_TR = 0.576341  # log10(TCMB = 3.77 K)
    log10_y_alpha = 11.77 + log10_n_alpha - 1.5 * log10_kinetic_temp
    log10_T0 = -1.1669  # log10(T0 = 0.0681 K)
    log10_A10 = -14.53995  # s-1 Einstein A
    log10_T2 = log10_kinetic_temp - 2.0
    log10_k10 = pt.switch(
        log10_kinetic_temp < 2.477,  # log10(300 K)
        -9.92445 + log10_T2 * (0.74 - 0.20 * log10_T2 / 0.43429),
        -9.64975 + 0.207 * log10_T2 - 0.3804 * pt.exp(-2.3026 * log10_T2),
    )
    log10_R10 = log10_density + log10_k10
    log10_y_c = log10_T0 + log10_R10 - log10_kinetic_temp - log10_A10
    log10_spin_temp = pt.log10(
        10.0**log10_TR
        + 10.0 ** (log10_y_c + log10_kinetic_temp)
        + 10.0 ** (log10_y_alpha + log10_kinetic_temp)
    ) - pt.log10(1.0 + 10.0**log10_y_c + 10.0**log10_y_alpha)
    return log10_spin_temp


def calc_log10_Geff(log10_G0: float, log10_NH: float) -> float:
    """
    Calculate the effective FUV radiation field.
    Bellomi et al. (2020) equations 11 and 12

    Inputs:
        log10_G0 :: scalar
            log10 normalization factor (Habing)
            (1 Habing = 1.2e-4 erg cm-2 s-1 sr-1 = 10^8 photons cm-2 s-1)
            (1.6 Habing = Draine 1978 value for solar neighborhood)
        log10_NH :: scalar
            log10 ydrogen column density (cm-2)

    Returns:
        log10_Geff :: scalar
            log10 effective FUV radiation field (Habing)
    """
    log10_AV = log10_NH - 21.27246
    AV = 10.0**log10_AV
    log10_Geff = log10_G0 - 2.5 * AV / 2.3025
    return log10_Geff


def calc_log10_ne(
    log10_nH: float,
    log10_cr_ion_rate: float,
    log10_kinetic_temp: float,
    log10_Geff: float,
    log10_xCII: float,
    log10_inv_pah_recomb: float,
) -> float:
    """
    Calculate the electron density assuming ionization equilibrium.
    Bellomi et al. (2020) equation B.1

    Inputs:
        log10_nH :: scalar
            log10 hydrogen density (cm-3)
        log10_cr_ion_rate :: scalar
            log10 soft X-ray and cosmic ray ionization rate (s-1)
        log10_kinetic_temp :: scalar
            log10 kinetic temperature (K)
        log10_Geff :: scalar
            log10 effective FUV radiation field (Habing)
        log10_xCII :: scalar
            log10 C+/H ratio by number
        log10_inv_pah_recomb :: scalar
            log10 inverse electron-PAH recombination parameter

    Returns:
        log10_ne :: scalar
            log10 electron density (cm-3)
    """
    log10_ne_equil = (
        0.5 * (log10_cr_ion_rate + 16.0)
        + 0.25 * (log10_kinetic_temp - 2.0)
        + 0.5 * log10_Geff
        + log10_inv_pah_recomb
        - 2.6197
    )
    log10_ne_carbon = log10_nH + log10_xCII
    log10_ne = pt.log10(10.0**log10_ne_equil + 10.0**log10_ne_carbon)
    return log10_ne


def calc_log10_kappa(
    log10_ne: float,
    log10_kinetic_temp: float,
    log10_Geff: float,
    log10_inv_pah_recomb: float,
) -> float:
    """
    Calculate the grain and PAH photoelectric heating constant.
    Bellomi et al. (2020) equation B.4

    Inputs:
        log10_ne :: scalar
            log10 electron density  (cm-3)
        log10_kinetic_temp :: scalar
            log10 kinetic temperature (K)
        log10_Geff :: scalar
            log10 effective FUV radiation field (Habing)
        log10_inv_pah_recomb :: scalar
            log10 inverse electron-PAH recombination parameter

    Returns:
        log10_kappa :: scalar
            log10 grain photoelectric heating constant
    """
    log10_kappa = (
        log10_Geff + 0.5 * log10_kinetic_temp + log10_inv_pah_recomb - log10_ne
    )
    return log10_kappa


def calc_log10_gamma_ph(
    log10_kinetic_temp: float, log10_Geff: float, log10_kappa: float
) -> float:
    """
    Calculate the heating induced by photoelectric effect on grains and PAHs.
    Bellomi et al. (2020) equation B.2 and B.3

    Inputs:
        log10_kinetic_temp :: scalar
            log10 kinetic temperature (K)
        log10_Geff :: scalar
            log10 effective FUV radiation field (Habing)
        log10_kappa :: scalar
            log10 grain photoelectric heating constant

    Returns:
        log10_gamma_ph :: scalar
            log10 heating induced by photoelectric effect on grains and PAHs  (erg s-1)
    """
    kappa = 10.0**log10_kappa
    eff1 = 4.9e-2 / (1.0 + (kappa / 1925.0) ** 0.73)
    log10_eff2 = (
        0.7 * (log10_kinetic_temp - 4.0) - 1.4318 - pt.log10(1.0 + kappa / 5000.0)
    )
    log10_eff = pt.log10(eff1 + 10.0**log10_eff2)
    log10_gamma_ph = log10_eff + log10_Geff - 23.886
    return log10_gamma_ph


def calc_log10_gamma_cr(cr_ion_rate: float) -> float:
    """
    Calculate the heating induced by soft X-ray and CR ionizations.
    Bellomi et al. (2020) equation B.5

    Inputs:
        log10_cr_ion_rate :: scalar
            log10 soft X-ray and cosmic ray ionization rate  (s-1)

    Returns:
        log10_gamma_cr :: scalar
            log10 heating induced by soft X-ray and CR ionizations (erg s-1)
    """
    log10_gamma_cr = cr_ion_rate - 11.0
    return log10_gamma_cr


def calc_log10_lambda_cii(
    log10_nH: float, log10_ne: float, log10_kinetic_temp: float, log10_xCII: float
) -> float:
    """
    Calculate the cooling rate due to collisional excitation of the fine
    structure levels of C+
    Bellomi et al. (2020) equation B.6

    Inputs:
        log10_nH :: scalar
            log10 ydrogen density (cm-3)
        log10_ne :: scalar
            log10 electron density (cm-3)
        log10_kinetic_temp :: scalar
            log10 kinetic temperature (K)
        log10_xCII :: scalar
            log10 C+/H ratio by number

    Returns:
        log10_lambda_cii :: scalar
            log10 cooling rate due to C+ (erg cm3 s-1)
    """
    log10_term_1 = -22.649752
    log10_term_2 = -0.5 * (log10_kinetic_temp - 2.0) + log10_ne - log10_nH - 20.0
    log10_term = pt.log10(10.0**log10_term_1 + 10.0**log10_term_2)
    kinetic_temp = 10.0**log10_kinetic_temp
    log10_lambda_cii = log10_term - 92.0 / kinetic_temp / 2.3025 + log10_xCII
    return log10_lambda_cii


def calc_log10_lambda_oi(log10_kinetic_temp: float, log10_xO: float) -> float:
    """
    Calculate the cooling rate by collisional excitation of the fine-structure
    levels of OI
    Bellomi et al. (2020) equation B.7

    Inputs:
        log10_kinetic_temp :: scalar
            log10 kinetic temperature (K)
        log10_xO :: scalar
            log10 O/H ratio by number

    Returns:
        log10_lambda_oi :: scalar
            log10 cooling rate due to O (erg cm3 s-1)
    """
    kinetic_temp = 10.0**log10_kinetic_temp
    log10_lambda_oi = (
        0.4 * (log10_kinetic_temp - 2.0)
        - 228.0 / kinetic_temp / 2.3025
        + log10_xO
        - 23.10735
    )
    return log10_lambda_oi


def calc_lambda_hi(nH: float, ne: float, kinetic_temp: float) -> float:
    """
    Calculate the cooling induced by the excitation of Lyman alpha
    Bellomi et al. (2020) equation B.8

    Inputs:
        nH :: scalar (cm-3)
            Hydrogen density
        ne :: scalar (cm-3)
            Electron density
        kinetic_temp :: scalar (K)
            Kinetic temperature

    Returns:
        lambda_hi :: scalar (erg cm3 s-1)
            Cooling due to Lyman alpha excitation
    """
    lambda_hi = 7.3e-19 * ne / nH * pt.exp(-118400.0 / kinetic_temp)
    return lambda_hi


def calc_log10_lambda_hi(
    log10_nH: float, log10_ne: float, log10_kinetic_temp: float
) -> float:
    """
    Calculate the cooling induced by the excitation of Lyman alpha
    Bellomi et al. (2020) equation B.8

    Inputs:
        log10_nH :: scalar
            log10 hydrogen density (cm-3)
        log10_ne :: scalar
            log10 electron density (cm-3)
        log10_kinetic_temp :: scalar
            log10 kinetic temperature (K)

    Returns:
        log10_lambda_hi :: scalar
            log10 cooling due to Lyman alpha excitation (erg cm3 s-1)
    """
    kinetic_temp = 10.0**log10_kinetic_temp
    log10_lambda_hi = log10_ne - log10_nH - 118400.0 / kinetic_temp / 2.3025 - 18.136677
    return log10_lambda_hi


def calc_log10_lambda_rec(
    log10_nH: float,
    log10_ne: float,
    log10_kinetic_temp: float,
    log10_kappa: float,
    log10_inv_pah_recomb: float,
) -> float:
    """
    Calculate the cooling due to electron recombination onto charged grains and PAHs
    Bellomi et al. (2020) equation B.9

    Inputs:
        log10_nH :: scalar
            log10 hydrogen density (cm-3)
        log10_ne :: scalar
            log10 electron density (cm-3)
        log10_kinetic_temp :: scalar
            log10 kinetic temperature (K)
        log10_kappa :: scalar
            log10 grain photoelectric heating constant
        log10_inv_pah_recomb :: scalar
            log10 inverse electron-PAH recombination parameter

    Returns:
        log10_lambda_rec :: scalar
            log10 cooling due to electron recombination on charged grains and PAHs (erg cm3 s-1)
    """
    log10_beta = -0.13076 - 0.068 * log10_kinetic_temp
    beta = 10.0**log10_beta
    log10_lambda_rec = (
        0.94 * log10_kinetic_temp
        + beta * log10_kappa
        + log10_ne
        - log10_nH
        - log10_inv_pah_recomb
        - 29.33254
    )
    return log10_lambda_rec


def calc_log10_heating_cooling(
    log10_nH: float,
    log10_G0: float,
    log10_NH: float,
    log10_kinetic_temp: float,
    log10_cr_ion_rate: float,
    log10_xCII: float,
    log10_xO: float,
    log10_inv_pah_recomb: float,
) -> tuple[float, float]:
    """
    Calculate the heating rate.

    Inputs:
        log10_nH :: scalar
            log10 hydrogen density (cm-3)
        log10_G0 :: scalar
            log10 normalization factor (Habing)
            (1 Habing = 1.2e-4 erg cm-2 s-1 sr-1 = 10^8 photons cm-2 s-1)
            (1.6 Habing = Draine 1978 value for solar neighborhood)
        log10_NH :: scalar
            log10 hydrogen column density (cm-2)
        log10_kinetic_temp :: scalar
            log10 gas kinetic temperature (K)
        log10_cr_ion_rate :: scalar
            log10 soft X-ray and cosmic ray ionization rate (s-1)
        log10_xCII :: scalar
            log10 C+/H ratio by number
        log10_xO :: scalar
            log10 O/H ratio by number
        log10_inv_pah_recomb :: scalar
            log10 inverse electron-PAH recombination parameter

    Returns:
        log10_heating :: scalar
            log10 heating rate (erg s-1 cm-3)
        log10_cooling :: scalar
            log10 cooling rate (erg s-1 cm-3)
    """
    log10_Geff = calc_log10_Geff(log10_G0, log10_NH)
    log10_ne = calc_log10_ne(
        log10_nH,
        log10_cr_ion_rate,
        log10_kinetic_temp,
        log10_Geff,
        log10_xCII,
        log10_inv_pah_recomb,
    )
    log10_kappa = calc_log10_kappa(
        log10_ne, log10_kinetic_temp, log10_Geff, log10_inv_pah_recomb
    )
    gamma_ph = 10.0 ** calc_log10_gamma_ph(log10_kinetic_temp, log10_Geff, log10_kappa)
    gamma_cr = 10.0 ** calc_log10_gamma_cr(log10_cr_ion_rate)
    log10_heating = log10_nH + pt.log10(gamma_ph + gamma_cr)

    lambda_cii = 10.0 ** calc_log10_lambda_cii(
        log10_nH, log10_ne, log10_kinetic_temp, log10_xCII
    )
    lambda_oi = 10.0 ** calc_log10_lambda_oi(log10_kinetic_temp, log10_xO)
    lambda_hi = 10.0 ** calc_log10_lambda_hi(log10_nH, log10_ne, log10_kinetic_temp)
    lambda_rec = 10.0 ** calc_log10_lambda_rec(
        log10_nH, log10_ne, log10_kinetic_temp, log10_kappa, log10_inv_pah_recomb
    )
    log10_cooling = 2.0 * log10_nH + pt.log10(
        lambda_cii + lambda_oi + lambda_hi + lambda_rec
    )
    return log10_heating, log10_cooling


def radiative_transfer(
    emission_velocity: Iterable[float],
    absorption_velocity: Iterable[float],
    peak_tau: Iterable[float],
    spin_temp: Iterable[float],
    velocity: Iterable[float],
    fwhm: Iterable[float],
) -> tuple[
    Iterable[float],
    Iterable[float],
    Iterable[Iterable[float]],
    Iterable[Iterable[float]],
]:
    """
    Calculate radiative transfer to prediction emission and absorption spectra.
    Order of N clouds is [nearest, ..., farthest]

    Inputs:
        emission_velocity :: 1-D array of scalars (km/s)
            Emission spectrum velocity axis (length E)
        absorption_velocity :: 1-D array of scalars (km/s)
            Absorption spectrum velocity axis (length A)
        peak_tau :: 1-D array of scalars
            Peak optical depth (length N)
        spin_temp :: 1-D array of scalars (K)
            Spin temperature (length N)
        velocity :: 1-D array of scalars (km/s)
            Velocity centroids (length N)
        fwhm :: 1-D array of scalars (km/s)
            Cloud FWHM line widths (length N)

    Returns:
        pred_emission :: 1-D array of scalars (K)
            Predicted emission spectrum (length E)
        pred_absorption :: 1-D array of scalars
            Predicted absorption spectrum (length A)
        cloud_pred_emission :: 2-D array of scalars (K)
            Predicted emission spectrum per cloud (shape ExN)
        cloud_pred_absorption :: 2-D array of scalars
            Predicted absorption spectrum per cloud (shape AxN)
    """
    # Optical depth per cloud
    emission_tau = gaussian(emission_velocity[:, None], peak_tau, velocity, fwhm)
    front_tau = pt.zeros_like(emission_velocity[:, None])
    emission_sum_tau = pt.concatenate([front_tau, emission_tau.cumsum(axis=1)], axis=1)
    absorption_tau = gaussian(absorption_velocity[:, None], peak_tau, velocity, fwhm)

    # Emission per cloud, attenuated by foreground
    emission = (
        spin_temp[None, :]
        * (1.0 - pt.exp(-emission_tau))
        * pt.exp(-emission_sum_tau[:, :-1])
    )

    # Radiative transfer
    pred_absorption = pt.sum(absorption_tau, axis=1)
    pred_emission = pt.sum(
        emission * pt.exp(-emission_sum_tau[:, :-1]),
        axis=1,
    )
    return pred_emission, pred_absorption, emission, absorption_tau
