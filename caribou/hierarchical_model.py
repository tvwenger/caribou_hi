"""
hierarchical_model.py - HierarchicalModel definition

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

import pymc as pm

from caribou.base_model import BaseModel


class HierarchicalModel(BaseModel):
    """
    HierarchicalModel extends BaseModel to fit physical parameters using
    a non-centered hierarchical model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model parameters
        self.hyper_params += [
            "log10_NHI_mu",
            "log10_NHI_sigma",
            "log10_density_mu",
            "log10_density_sigma",
            "log10_kinetic_temp_mu",
            "log10_kinetic_temp_sigma",
            "log10_n_alpha_mu",
            "log10_n_alpha_sigma",
            "log10_larson_linewidth_mu",
            "log10_larson_linewidth_sigma",
            "larson_power_mu",
            "larson_power_sigma",
            "velocity_mu",
            "velocity_sigma",
        ]
        self.cloud_params += [
            "log10_NHI_offset",
            "log10_density_offset",
            "log10_kinetic_temp_offset",
            "log10_n_alpha_offset",
            "log10_larson_linewidth_offset",
            "larson_power_offset",
            "velocity_offset",
        ]

        # Deterministic quantities
        self.deterministics += [
            "log10_NHI",
            "log10_density",
            "log10_kinetic_temp",
            "log10_n_alpha",
            "log10_larson_linewidth",
            "larson_power",
            "velocity",
            "log10_spin_temp",
            "log10_thermal_fwhm",
            "log10_depth",
            "log10_nonthermal_fwhm",
            "log10_fwhm",
            "log10_peak_tau",
        ]

        self._cluster_features += [
            "log10_NHI",
            "log10_density",
            "log10_kinetic_temp",
            "log10_n_alpha",
            "log10_larson_linewidth",
            "larson_power",
            "velocity",
        ]

        self.var_name_map.update(
            {
                "log10_NHI": r"log$_{10}$ $N_{\rm HI}$ (cm$^{-2}$)",
                "log10_density": r"log$_{10}$ $n_{\rm HI}$ (cm$^{-3}$)",
                "log10_kinetic_temp": r"log$_{10}$ $T_K$ (K)",
                "log10_n_alpha": r"log$_{10}$ $n_\alpha$ (cm$^{-3}$)",
                "log10_larson_linewidth": r"log$_{10}$ $\Delta V_{\rm 1 pc}$ (km s$^{-1}$)",
                "larson_power": r"$\alpha$",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "log10_spin_temp": r"log$_{10}$ $T_S$ (K)",
                "log10_thermal_fwhm": r"log$_{10}$ $\Delta V_{\rm th}$ (km s$^{-1}$)",
                "log10_depth": r"log$_{10}$ $d$ (pc)",
                "log10_nonthermal_fwhm": r"log$_{10}$ $\Delta V_{\rm nth}$ (km s$^{-1}$)",
                "log10_fwhm": r"log$_{10}$ $\Delta V$ (km s$^{-1}$)",
                "log10_peak_tau": r"log$_{10}$ $\tau_0$",
            }
        )

    def set_priors(
        self,
        prior_log10_NHI: Iterable[float] = [20.0, 0.5, 0.5],
        prior_log10_kinetic_temp: Iterable[float] = [3.0, 0.5, 0.5],
        prior_log10_density: Iterable[float] = [0.0, 0.5, 0.5],
        prior_log10_n_alpha: Iterable[float] = [-6.0, 0.5, 0.5],
        prior_log10_larson_linewidth: Iterable[float] = [0.2, 0.05, 0.05],
        prior_larson_power: Iterable[float] = [0.4, 0.05, 0.05],
        prior_velocity: Iterable[float] = [0.0, 5.0, 5.0],
    ):
        """
        Add priors and likelihood to the model.

        Inputs:
            prior_log10_NHI :: 3-length array of scalars
                Prior distribution on log10(NHI) (cm-2), where:
                log10_NHI_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_NHI_sigma ~ HalfNormal(sigma=prior[2])
                log10_NHI ~ Normal(mu=log10_NHI_mu, sigma=log10_NHI_sigma)
            prior_log10_kinetic_temp :: 3-length array of scalars
                Prior distribution on log10(kinetic_temp) (K), where:
                log10_kinetic_temp_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_kinetic_temp_sigma ~ HalfNormal(sigma=prior[2])
                log10_kinetic_temp ~ Normal(mu=log10_kinetic_temp_mu, sigma=log10_kinetic_temp_sigma)
            prior_log10_density :: 3-length array of scalars
                Prior distribution on log10(density) (cm-3), where:
                log10_density_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_density_sigma ~ HalfNormal(sigma=prior[2])
                log10_density ~ Normal(mu=log10_density_mu, sigma=log10_density_sigma)
            prior_log10_n_alpha :: 3-length array of scalars
                Prior distribution on log10(n_alpha) (cm-3), where:
                log10_n_alpha_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_n_alpha_sigma ~ HalfNormal(sigma=prior[2])
                log10_n_alpha ~ Normal(mu=log10_n_alpha_mu, sigma=log10_n_alpha_sigma)
            prior_log10_larson_linewidth :: 3-length array of scalars
                Prior distribution on log10(larson_linewidth) (km/s), where
                log10_larson_linewidth_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_larson_linewidth_sigma ~ HalfNormal(sigma=prior[2])
                log10_larson_linewidth ~ Normal(mu=log10_larson_linewidth_mu, sigma=log10_larson_linewidth_sigma)
            prior_larson_power :: 3-length array of scalars
                Prior distribution on larson_power, where
                larson_power_mu ~ Normal(mu=prior[0], sigma=prior[1])
                larson_power_sigma ~ HalfNormal(sigma=prior[2])
                larson_power ~ Normal(mu=larson_power_mu, sigma=larson_power_sigma)
            prior_velocity :: 3-length array of scalars
                Prior distribution on velocity (km s-1), where:
                velocity_mu ~ Normal(mu=prior[0], sigma=prior[1])
                velocity_sigma ~ HalfNormal(sigma=prior[2])
                velocity ~ Normal(mu=velocity_mu, sigma=velocity_sigma)

        Returns: Nothing
        """
        # add baseline priors
        super().set_priors()

        with self.model:
            # Column density (cm-2)
            log10_NHI_mu = pm.Normal(
                "log10_NHI_mu",
                mu=prior_log10_NHI[0],
                sigma=prior_log10_NHI[1],
            )
            log10_NHI_sigma = pm.HalfNormal("log10_NHI_sigma", sigma=prior_log10_NHI[2])
            log10_NHI_offset = pm.Normal(
                "log10_NHI_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_NHI",
                log10_NHI_mu + log10_NHI_offset * log10_NHI_sigma,
                dims="cloud",
            )

            # Kinetic temperature (K)
            log10_kinetic_temp_mu = pm.Normal(
                "log10_kinetic_temp_mu",
                mu=prior_log10_kinetic_temp[0],
                sigma=prior_log10_kinetic_temp[1],
            )
            log10_kinetic_temp_sigma = pm.HalfNormal(
                "log10_kinetic_temp_sigma", sigma=prior_log10_kinetic_temp[2]
            )
            log10_kinetic_temp_offset = pm.Normal(
                "log10_kinetic_temp_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_kinetic_temp",
                log10_kinetic_temp_mu
                + log10_kinetic_temp_offset * log10_kinetic_temp_sigma,
                dims="cloud",
            )

            # Density (cm-3)
            log10_density_mu = pm.Normal(
                "log10_density_mu",
                mu=prior_log10_density[0],
                sigma=prior_log10_density[1],
            )
            log10_density_sigma = pm.HalfNormal(
                "log10_density_sigma", sigma=prior_log10_density[2]
            )
            log10_density_offset = pm.Normal(
                "log10_density_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_density",
                log10_density_mu + log10_density_offset * log10_density_sigma,
                dims="cloud",
            )

            # Lyman alpha photon n_alpha (cm-3)
            log10_n_alpha_mu = pm.Normal(
                "log10_n_alpha_mu",
                mu=prior_log10_n_alpha[0],
                sigma=prior_log10_n_alpha[1],
            )
            log10_n_alpha_sigma = pm.HalfNormal(
                "log10_n_alpha_sigma", sigma=prior_log10_n_alpha[2]
            )
            log10_n_alpha_offset = pm.Normal(
                "log10_n_alpha_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_n_alpha",
                log10_n_alpha_mu + log10_n_alpha_offset * log10_n_alpha_sigma,
                dims="cloud",
            )

            # Larson's line width (km/s)
            log10_larson_linewidth_mu = pm.Normal(
                "log10_larson_linewidth_mu",
                mu=prior_log10_larson_linewidth[0],
                sigma=prior_log10_larson_linewidth[1],
            )
            log10_larson_linewidth_sigma = pm.HalfNormal(
                "log10_larson_linewidth_sigma", sigma=prior_log10_larson_linewidth[2]
            )
            log10_larson_linewidth_offset = pm.Normal(
                "log10_larson_linewidth_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_larson_linewidth",
                log10_larson_linewidth_mu
                + log10_larson_linewidth_offset * log10_larson_linewidth_sigma,
                dims="cloud",
            )

            # Larson's power
            larson_power_mu = pm.Normal(
                "larson_power_mu",
                mu=prior_larson_power[0],
                sigma=prior_larson_power[1],
            )
            larson_power_sigma = pm.HalfNormal(
                "larson_power_sigma", sigma=prior_larson_power[2]
            )
            larson_power_offset = pm.Normal(
                "larson_power_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "larson_power",
                larson_power_mu + larson_power_offset * larson_power_sigma,
                dims="cloud",
            )

            # Centroid velocity (km s-1)
            velocity_mu = pm.Normal(
                "velocity_mu", mu=prior_velocity[0], sigma=prior_velocity[1]
            )
            velocity_sigma = pm.HalfNormal("velocity_sigma", sigma=prior_velocity[2])
            velocity_offset = pm.Normal(
                "velocity_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "velocity",
                velocity_mu + velocity_offset * velocity_sigma,
                dims="cloud",
            )

        self._add_spectral_likelihood()
