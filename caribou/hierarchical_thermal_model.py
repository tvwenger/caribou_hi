"""
hierarchical_thermal_model.py - HierarchicalThermalModel definition

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

from caribou.hierarchical_model import HierarchicalModel


class HierarchicalThermalModel(HierarchicalModel):
    """
    HierarchicalThermalModel extends HierarchicalModel to additionally
    constrain net cooling using a non-centered hierarchical model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Model parameters
        self.hyper_params += [
            "log10_cr_ion_rate_mu",
            "log10_cr_ion_rate_sigma",
            "log10_G0_mu",
            "log10_G0_sigma",
            "log10_xCII_mu",
            "log10_xCII_sigma",
            "log10_xO_mu",
            "log10_xO_sigma",
            "log10_inv_pah_recomb_mu",
            "log10_inv_pah_recomb_sigma",
            "log10_thermal_ratio_mu",
            "log10_thermal_ratio_sigma",
        ]
        self.cloud_params += [
            "log10_cr_ion_rate_offset",
            "log10_G0_offset",
            "log10_xCII_offset",
            "log10_xO_offset",
            "log10_inv_pah_recomb_offset",
        ]
        self.deterministics += [
            "log10_cr_ion_rate",
            "log10_G0",
            "log10_xCII",
            "log10_xO",
            "log10_inv_pah_recomb",
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
        prior_log10_cr_ion_rate: Iterable[float] = [-16.0, 0.5, 0.5],
        prior_log10_G0: Iterable[float] = [0.0, 0.5, 0.5],
        prior_log10_xCII: Iterable[float] = [-4.0, 0.5, 0.5],
        prior_log10_xO: Iterable[float] = [-4.0, 0.5, 0.5],
        prior_log10_inv_pah_recomb: Iterable[float] = [0.2, 0.05, 0.05],
        prior_log10_thermal_ratio: Iterable[float] = [0.0, 0.5, 0.5],
        **kwargs,
    ):
        """
        Add priors and likelihood to the model.

        Inputs:
            prior_log10_cr_ion_rate :: 3-length array of scalars
                Prior distribution on log10(cr_ion_rate) (s-1), where:
                log10_cr_ion_rate_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_cr_ion_rate_sigma ~ HalfNormal(sigma=prior[2])
                log10_cr_ion_rate ~ Normal(mu=log10_cr_ion_rate_mu, sigma=log10_cr_ion_rate_sigma)
            prior_log10_G0 :: 3-length array of scalars
                Prior distribution on log10(G0) (Habing), where:
                log10_G0_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_G0_sigma ~ HalfNormal(sigma=prior[2])
                log10_G0 ~ Normal(mu=log10_G0_mu, sigma=log10_G0_sigma)
            prior_log10_xCII :: 3-length array of scalars
                Prior distribution on log10(xCII), where:
                log10_xCII_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_xCII_sigma ~ HalfNormal(sigma=prior[2])
                log10_xCII ~ Normal(mu=log10_xCII_mu, sigma=log10_xCII_sigma)
            prior_log10_xO :: 3-length array of scalars
                Prior distribution on log10(xO), where:
                log10_xO_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_xO_sigma ~ HalfNormal(sigma=prior[2])
                log10_xO ~ Normal(mu=log10_xO_mu, sigma=log10_xO_sigma)
            prior_log10_inv_pah_recomb :: 3-length array of scalars
                Prior distribution on log10(cooling/heating), where:
                log10_inv_pah_recomb_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_inv_pah_recomb_sigma ~ HalfNormal(sigma=prior[2])
                log10_inv_pah_recomb ~ Normal(mu=log10_inv_pah_recomb_mu, sigma=log10_inv_pah_recomb_sigma)
            prior_log10_thermal_ratio :: 3-length array of scalars
                Prior distribution on log10(cooling/heating), where:
                log10_thermal_ratio_mu ~ Normal(mu=prior[0], sigma=prior[1])
                log10_thermal_ratio_sigma ~ HalfNormal(sigma=prior[2])

        Returns: Nothing
        """
        super().set_priors(**kwargs)

        with self.model:
            # Soft X-ray and CR ionization rate (s-1)
            log10_cr_ion_rate_mu = pm.Normal(
                "log10_cr_ion_rate_mu",
                mu=prior_log10_cr_ion_rate[0],
                sigma=prior_log10_cr_ion_rate[1],
            )
            log10_cr_ion_rate_sigma = pm.HalfNormal(
                "log10_cr_ion_rate_sigma", sigma=prior_log10_cr_ion_rate[2]
            )
            log10_cr_ion_rate_offset = pm.Normal(
                "log10_cr_ion_rate_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_cr_ion_rate",
                log10_cr_ion_rate_mu
                + log10_cr_ion_rate_offset * log10_cr_ion_rate_sigma,
                dims="cloud",
            )

            # FUV field intensity (Habing)
            log10_G0_mu = pm.Normal(
                "log10_G0_mu",
                mu=prior_log10_G0[0],
                sigma=prior_log10_G0[1],
            )
            log10_G0_sigma = pm.HalfNormal("log10_G0_sigma", sigma=prior_log10_G0[2])
            log10_G0_offset = pm.Normal(
                "log10_G0_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_G0",
                log10_G0_mu + log10_G0_offset * log10_G0_sigma,
                dims="cloud",
            )

            # CII abundance
            log10_xCII_mu = pm.Normal(
                "log10_xCII_mu",
                mu=prior_log10_xCII[0],
                sigma=prior_log10_xCII[1],
            )
            log10_xCII_sigma = pm.HalfNormal(
                "log10_xCII_sigma", sigma=prior_log10_xCII[2]
            )
            log10_xCII_offset = pm.Normal(
                "log10_xCII_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_xCII",
                log10_xCII_mu + log10_xCII_offset * log10_xCII_sigma,
                dims="cloud",
            )

            # O abundance
            log10_xO_mu = pm.Normal(
                "log10_xO_mu",
                mu=prior_log10_xO[0],
                sigma=prior_log10_xO[1],
            )
            log10_xO_sigma = pm.HalfNormal("log10_xO_sigma", sigma=prior_log10_xO[2])
            log10_xO_offset = pm.Normal(
                "log10_xO_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_xO",
                log10_xO_mu + log10_xO_offset * log10_xO_sigma,
                dims="cloud",
            )

            # PAH recombination parameter
            log10_inv_pah_recomb_mu = pm.Normal(
                "log10_inv_pah_recomb_mu",
                mu=prior_log10_inv_pah_recomb[0],
                sigma=prior_log10_inv_pah_recomb[1],
            )
            log10_inv_pah_recomb_sigma = pm.HalfNormal(
                "log10_inv_pah_recomb_sigma", sigma=prior_log10_inv_pah_recomb[2]
            )
            log10_inv_pah_recomb_offset = pm.Normal(
                "log10_inv_pah_recomb_offset", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_inv_pah_recomb",
                log10_inv_pah_recomb_mu
                + log10_inv_pah_recomb_offset * log10_inv_pah_recomb_sigma,
                dims="cloud",
            )

            # Thermal ratio
            log10_thermal_ratio_mu = pm.Normal(
                "log10_thermal_ratio_mu",
                mu=prior_log10_thermal_ratio[0],
                sigma=prior_log10_thermal_ratio[1],
            )
            log10_thermal_ratio_sigma = pm.HalfNormal(
                "log10_thermal_ratio_sigma",
                sigma=prior_log10_thermal_ratio[2],
            )

        self._add_thermal_balance_likelihood(
            [log10_thermal_ratio_mu, log10_thermal_ratio_sigma]
        )
