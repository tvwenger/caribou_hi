"""
thermal_model.py - ThermalModel definition

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

from caribou.simple_model import SimpleModel


class ThermalModel(SimpleModel):
    """
    ThermalModel extends SimpleModel to additionally constrain thermal balance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model parameters
        self.cloud_params += [
            "log10_cr_ion_rate",
            "log10_G0",
            "log10_xCII",
            "log10_xO",
            "log10_inv_pah_recomb",
        ]
        self.deterministics += [
            "log10_cooling",
            "log10_heating",
            "log10_thermal_ratio",
        ]

        self._cluster_features += []

        self.var_name_map.update(
            {
                "log10_cr_ion_rate": r"log$_{10}$ $\zeta_{\rm CR}$ (s$^{-1}$)",
                "log10_G0": r"log$_{10}$ $G_0$ (Habing)",
                "log10_xCII": r"log$_{10}$ $x_{{\rm C}^+}$",
                "log10_xO": r"log$_{10}$ $x_{\rm O}$",
                "log10_inv_pah_recomb": r"log$_{10}$ $\phi_{\rm PAH}^{-1}$",
                "log10_cooling": r"log$_{10}$ $\mathcal{L}$ (erg s$^{-1}$ cm$^{-3}$)",
                "log10_heating": r"log$_{10}$ $\mathcal{G}$ (erg s$^{-1}$ cm$^{-3}$)",
                "log10_thermal_ratio": r"log$_{10}(\mathcal{L}/\mathcal{G})$",
            }
        )

        with self.model:
            _ = pm.ConstantData(
                "log10_equal_thermal_ratio", [0.0] * self.n_clouds, dims="cloud"
            )

    def set_priors(
        self,
        prior_log10_cr_ion_rate: Iterable[float] = [-16.0, 1.0],
        prior_log10_G0: Iterable[float] = [0.0, 1.0],
        prior_log10_xCII: Iterable[float] = [-4.0, 1.0],
        prior_log10_xO: Iterable[float] = [-4.0, 1.0],
        prior_log10_inv_pah_recomb: Iterable[float] = [0.2, 0.1],
        prior_log10_thermal_ratio: Iterable[float] = [0.0, 1.0],
        **kwargs,
    ):
        """
        Add priors and likelihood to the model.

        Inputs:
            prior_log10_cr_ion_rate :: 2-length array of scalars
                Prior distribution on log10(cr_ion_rate) (s-1), where:
                log10_cr_ion_rate ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_G0 :: 2-length array of scalars
                Prior distribution on log10(G0) (Habing), where:
                log10_G0 ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_xCII :: 2-length array of scalars
                Prior distribution on log10(xCII), where:
                log10_xCII ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_xO :: 2-length array of scalars
                Prior distribution on log10(xO), where:
                log10_xO ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_inv_pah_recomb :: 2-length array of scalars
                Prior distribution on log10(1/pah_recomb), where:
                log10_inv_pah_recomb ~ Normal(mu=prior[0], sigma=prior[1])
            prior_log10_thermal_ratio :: 2-length array of scalars
                Prior distribution on log10(cooling/heating), where:
                log10_thermal_ratio ~ Normal(mu=prior[0], sigma=prior[1])
            **kwargs ::
                Additional keyword arguments passed to SimpleModel.set_priors()

        Returns: Nothing
        """
        super().set_priors(**kwargs)

        with self.model:
            # Soft X-ray and CR ionization rate (s-1)
            _ = pm.Normal(
                "log10_cr_ion_rate",
                mu=prior_log10_cr_ion_rate[0],
                sigma=prior_log10_cr_ion_rate[1],
                dims="cloud",
            )

            # FUV field intensity (Habing)
            _ = pm.Normal(
                "log10_G0",
                mu=prior_log10_G0[0],
                sigma=prior_log10_G0[1],
                dims="cloud",
            )

            # CII abundance
            _ = pm.Normal(
                "log10_xCII",
                mu=prior_log10_xCII[0],
                sigma=prior_log10_xCII[1],
                dims="cloud",
            )

            # O abundance
            _ = pm.Normal(
                "log10_xO",
                mu=prior_log10_xO[0],
                sigma=prior_log10_xO[1],
                dims="cloud",
            )

            # PAH recombination parameter
            _ = pm.Normal(
                "log10_inv_pah_recomb",
                mu=prior_log10_inv_pah_recomb[0],
                sigma=prior_log10_inv_pah_recomb[1],
                dims="cloud",
            )

        self._add_thermal_balance_likelihood(prior_log10_thermal_ratio)
