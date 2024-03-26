"""
simple_model.py - SimpleModel definition

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


class SimpleModel(BaseModel):
    """
    SimpleModel extends BaseModel to fit fundamental physical parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model parameters
        self.cloud_params += [
            "log10_NHI",
            "log10_density",
            "log10_kinetic_temp",
            "log10_n_alpha",
            "log10_larson_linewidth",
            "larson_power",
            "velocity",
        ]

        # Deterministic quantities
        self.deterministics += [
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
        prior_log10_NHI: Iterable[float] = [20.0, 1.0],
        prior_log10_kinetic_temp: Iterable[float] = [3.0, 1.0],
        prior_log10_density: Iterable[float] = [0.0, 1.0],
        prior_log10_n_alpha: Iterable[float] = [-6.0, 1.0],
        prior_log10_larson_linewidth: Iterable[float] = [0.2, 0.1],
        prior_larson_power: Iterable[float] = [0.4, 0.1],
        prior_velocity: Iterable[float] = [0.0, 10.0],
    ):
        """
        Add priors and likelihood to the model.

        Inputs:
            prior_log10_NHI :: 2-length array of scalars
                Prior distribution on log10(NHI) (cm-2), where:
                log10_NHI ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_kinetic_temp :: 2-length array of scalars
                Prior distribution on log10(kinetic_temp) (K), where:
                log10_kinetic_temp ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_density :: 2-length array of scalars
                Prior distribution on log10(density) (cm-3), where:
                log10_density ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_n_alpha :: 2-length array of scalars
                Prior distribution on log10(n_alpha) (cm-3), where:
                log10_n_alpha ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_larson_linewidth :: 2-length array of scalars
                Prior distribution on log10(larson_linewidth) (km/s), where
                log10_larson_linewidth ~ Normal(mu=prior[0], sigma=prior[1])
            prior_larson_power :: 2-length array of scalars
                Prior distribution on larson_power, where
                larson_power ~ Normal(mu=prior[0], sigma=prior[1])
            prior_velocity :: 3-length array of scalars
                Prior distribution on velocity (km s-1), where:
                velocity ~ Normal(mu=prior[0], sigma=prior[1])

        Returns: Nothing
        """
        # add baseline priors
        super().set_priors()

        with self.model:
            # Column density (cm-2)
            _ = pm.Normal(
                "log10_NHI",
                mu=prior_log10_NHI[0],
                sigma=prior_log10_NHI[1],
                dims="cloud",
            )

            # Kinetic temperature (K)
            _ = pm.Normal(
                "log10_kinetic_temp",
                mu=prior_log10_kinetic_temp[0],
                sigma=prior_log10_kinetic_temp[1],
                dims="cloud",
            )

            # Density (cm-3)
            _ = pm.Normal(
                "log10_density",
                mu=prior_log10_density[0],
                sigma=prior_log10_density[1],
                dims="cloud",
            )

            # Lyman alpha photon n_alpha (cm-3)
            _ = pm.Normal(
                "log10_n_alpha",
                mu=prior_log10_n_alpha[0],
                sigma=prior_log10_n_alpha[1],
                dims="cloud",
            )

            # Larson's line width (km/s)
            _ = pm.Normal(
                "log10_larson_linewidth",
                mu=prior_log10_larson_linewidth[0],
                sigma=prior_log10_larson_linewidth[1],
                dims="cloud",
            )

            # Larson's power
            _ = pm.Normal(
                "larson_power",
                mu=prior_larson_power[0],
                sigma=prior_larson_power[1],
                dims="cloud",
            )

            # Centroid velocity (km s-1)
            _ = pm.Normal(
                "velocity", mu=prior_velocity[0], sigma=prior_velocity[1], dims="cloud"
            )

        self._add_spectral_likelihood()
