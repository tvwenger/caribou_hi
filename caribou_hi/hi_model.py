"""
hi_model.py
HIModel definition

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
"""

from typing import Iterable, Optional
from abc import abstractmethod

import pymc as pm
import pytensor.tensor as pt

from bayes_spec import BaseModel

from caribou_hi import physics


class HIModel(BaseModel):
    """Definition of the HIModel model. This model is further extended by other models."""

    def __init__(self, *args, **kwargs):
        """Initialize a new HIModel instance"""
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Select features used for posterior clustering
        self._cluster_features += [
            "log10_NHI",
            "log10_nHI",
            "log10_tkin",
            "velocity",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_NHI": r"log$_{10}$ $N_{\rm HI}$ (cm$^{-2}$)",
                "log10_nHI": r"log$_{10}$ $n_{\rm HI}$ (cm$^{-3}$)",
                "log10_tkin": r"log$_{10}$ $T_K$ (K)",
                "log10_n_alpha": r"log$_{10}$ $n_\alpha$ (cm$^{-3}$)",
                "log10_larson_linewidth": r"log$_{10}$ $\Delta V_{\rm 1 pc}$ (km s$^{-1}$)",
                "larson_power": r"$\alpha$",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "tspin": r"$T_S$ (K)",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "depth": r"$d$ (pc)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nth}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_log10_NHI: Iterable[float] = [20.0, 1.0],
        prior_log10_nHI: Iterable[float] = [0.0, 1.0],
        prior_log10_tkin: Iterable[float] = [3.0, 1.0],
        prior_log10_n_alpha: Iterable[float] = [-6.0, 1.0],
        prior_log10_larson_linewidth: Iterable[float] = [0.2, 0.1],
        prior_larson_power: Iterable[float] = [0.4, 0.1],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        ordered: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_NHI : Iterable[float], optional
            Prior distribution on log10 HI column density (cm-2), by default [20.0, 1.0], where
            log10_NHI ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_nHI : Iterable[float], optional
            Prior distribution on log10 HI volume density (cm-3), by default [0.0, 1.0], where
            log10_nHI ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_tkin : Iterable[float], optional
            Prior distribution on log10 kinetic temperature (K), by default [3.0, 1.0], where
            log10_tkin ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_n_alpha : Iterable[float], optional
            Prior distribution on log10 n_alpha (cm-3), by default [-6.0, 1.0], where
            n_alpha ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_larson_linewidth : Iterable[float], optional
            Prior distribution on log10 larson_linewidth (km s-1), by default [0.2, 0.1], where
            log10_larson_linewidth ~ Normal(mu=prior[0], sigma=prior[1])
        prior_larson_power : Iterable[float], optional
            Prior distribution on larson_power, by default [0.4, 0.1], where
            larson_power ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False. If True, the prior
            distribution on the velocity becomes
            velocity(cloud = n) ~ prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # HI column density (cm-2; shape: clouds)
            log10_NHI_norm = pm.Normal("log10_NHI_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_NHI = pm.Deterministic(
                "log10_NHI",
                prior_log10_NHI[0] + prior_log10_NHI[1] * log10_NHI_norm,
                dims="cloud",
            )

            # HI volume density (cm-3; shape: clouds)
            log10_nHI_norm = pm.Normal("log10_nHI_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_nHI = pm.Deterministic(
                "log10_nHI",
                prior_log10_nHI[0] + prior_log10_nHI[1] * log10_nHI_norm,
                dims="cloud",
            )

            # kinetic temperature temperature (K; shape: clouds)
            log10_tkin_norm = pm.Normal("log10_tkin_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_tkin = pm.Deterministic(
                "log10_tkin",
                prior_log10_tkin[0] + prior_log10_tkin[1] * log10_tkin_norm,
                dims="cloud",
            )

            # Lyman-alpha photon density (cm-3; shape: clouds)
            log10_n_alpha_norm = pm.Normal("log10_n_alpha_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_n_alpha = pm.Deterministic(
                "log10_n_alpha",
                prior_log10_n_alpha[0] + prior_log10_n_alpha[1] * log10_n_alpha_norm,
                dims="cloud",
            )

            # Larson linewidth (km s-1; shape: clouds)
            log10_larson_linewidth_norm = pm.Normal("log10_larson_linewidth_norm", mu=0.0, sigma=1.0, dims="cloud")
            log10_larson_linewidth = pm.Deterministic(
                "log10_larson_linewidth",
                prior_log10_larson_linewidth[0] + prior_log10_larson_linewidth[1] * log10_larson_linewidth_norm,
                dims="cloud",
            )

            # Larson power (shape: clouds)
            larson_power_norm = pm.Normal("larson_power_norm", mu=0.0, sigma=1.0, dims="cloud")
            larson_power = pm.Deterministic(
                "larson_power",
                prior_larson_power[0] + prior_larson_power[1] * larson_power_norm,
                dims="cloud",
            )

            # Velocity (km/s; shape: clouds)
            if ordered:
                velocity_offset_norm = pm.Gamma("velocity_norm", alpha=2.0, beta=1.0, dims="cloud")
                velocity_offset = velocity_offset_norm * prior_velocity[1]
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + pm.math.cumsum(velocity_offset),
                    dims="cloud",
                )
            else:
                velocity_norm = pm.Normal(
                    "velocity_norm",
                    mu=0.0,
                    sigma=1.0,
                    dims="cloud",
                )
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + prior_velocity[1] * velocity_norm,
                    dims="cloud",
                )

            # Spin temperature (K; shape: clouds)
            _ = pm.Deterministic(
                "tspin", physics.calc_spin_temp(10.0**log10_tkin, 10.0**log10_nHI, 10.0**log10_n_alpha), dims="cloud"
            )

            # Thermal line width (km/s; shape: clouds)
            fwhm_thermal = pm.Deterministic("fwhm_thermal", physics.calc_thermal_fwhm(10.0**log10_tkin), dims="cloud")

            # Line-of-sight depth (pc; shape: clouds)
            depth = pm.Deterministic("depth", physics.calc_depth(10.0**log10_NHI, 10.0**log10_nHI), dims="cloud")

            # Non-thermal line width (km/s; shape: clouds)
            fwhm_nonthermal = pm.Deterministic(
                "fwhm_nonthermal",
                physics.calc_nonthermal_fwhm(depth, 10.0 * log10_larson_linewidth, larson_power),
                dims="cloud",
            )

            # Total line width (km/s; shape: clouds)
            _ = pm.Deterministic("fwhm", pt.sqrt(fwhm_thermal**2.0 + fwhm_nonthermal**2.0), dims="cloud")

    @abstractmethod
    def add_likelihood(self, *args, **kwargs):  # pragma: no cover
        """Must be defined in inhereted class."""
        pass
