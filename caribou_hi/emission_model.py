"""
emission_model.py
EmissionPhysicalModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import numpy as np

import pytensor.tensor as pt
import pymc as pm
from caribou_hi import physics

from caribou_hi.hi_model import HIModel


class EmissionModel(HIModel):
    """Definition of the EmissionModel model."""

    def __init__(self, *args, bg_temp: float = 3.77, **kwargs):
        """Initialize a new EmissionModel instance

        Parameters
        ----------
        bg_temp : float, optional
            Assumed background temperature (K), by default 3.77
        """
        # Initialize HIModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.bg_temp = bg_temp

    def add_priors(
        self,
        prior_TB_fwhm: float = 50.0,
        prior_tkin_factor: Iterable[float] = [2.0, 2.0],
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_TB_fwhm : float, optional
            Prior distribution on brightness temperature x FWHM (K km s), by default 50.0, where
            TB_fwhm ~ HalfNormal(sigma=prior)
        prior_tkin_factor : Iterable[float], optional
            Prior distribution on kinetic temperature factor, by default [2.0, 2.0], where
            tkin_factor ~ Beta(alpha=prior[0], beta=prior[1])
            tkin = tkin_factor * tkin_max
        """
        # Add HIModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # TB_fwhm = ff * Tspin * (1 - exp(-tau_peak)) * fwhm (K km s-1; shape: clouds)
            TB_fwhm_norm = pm.HalfNormal("TB_fwhm_norm", sigma=1.0, dims="cloud")
            TB_fwhm = pm.Deterministic(
                "TB_fwhm",
                prior_TB_fwhm * TB_fwhm_norm,
                dims="cloud",
            )

            # minimum kinetic temperature == TB (K; shape: clouds)
            tkin_min = TB_fwhm / pt.sqrt(self.model["fwhm2"])

            # maximum kinetic temperature (K; shape: clouds)
            tkin_max = physics.calc_kinetic_temp(self.model["fwhm2"])

            # kinetic temperature (K; shape: clouds)
            tkin_factor_norm = pm.Beta(
                "tkin_factor_norm",
                alpha=prior_tkin_factor[0],
                beta=prior_tkin_factor[1],
                dims="cloud",
            )
            tkin = pm.Deterministic(
                "tkin",
                pt.switch(
                    pt.gt(tkin_min, tkin_max),
                    tkin_max,
                    tkin_factor_norm * (tkin_max - tkin_min) + tkin_min,
                ),
                dims="cloud",
            )

            # Spin temperature (K; shape: clouds)
            tspin = pm.Deterministic(
                "tspin",
                physics.calc_spin_temp(
                    tkin,
                    10.0 ** self.model["log10_nHI"],
                    self.model["n_alpha"],
                ),
                dims="cloud",
            )

            # Minimum filling factor == TB / Tspin
            filling_factor_min = tkin_min / tspin

            # filling factor
            filling_factor_norm = pm.Uniform(
                "filling_factor_norm", lower=0.0, upper=1.0, dims="cloud"
            )
            filling_factor = pm.Deterministic(
                "filling_factor",
                pt.switch(
                    pt.gt(filling_factor_min, 1.0),
                    1.0,
                    filling_factor_norm * (1.0 - filling_factor_min)
                    + filling_factor_min,
                ),
                dims="cloud",
            )

            # (1 - exp(-tau_peak))
            exp_tau_peak = filling_factor_min / filling_factor
            exp_tau_peak = pt.clip(exp_tau_peak, 0.0, 0.9999)
            tau_peak = -pt.log(1.0 - exp_tau_peak)

            # total optical depth (km s-1; shape: clouds)
            const = np.sqrt(2.0 * np.pi) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            _ = pm.Deterministic(
                "tau_total",
                tau_peak * pt.sqrt(self.model["fwhm2"]) * const,
                dims="cloud",
            )

        # Add HIModel deterministics
        super().add_deterministics()

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "emission"."""
        # Evaluate line profile (shape: spectral, clouds)
        line_profile = physics.calc_pseudo_voigt(
            self.data["emission"].spectral,
            self.model["velocity"],
            self.model["fwhm2"],
            self.model["fwhm_L"],
        )

        # Optical depth spectra (shape: spectral, clouds)
        optical_depth = self.model["tau_total"] * line_profile

        # Evaluate radiative transfer
        predicted_emission = physics.radiative_transfer(
            optical_depth,
            self.model["tspin"],
            self.model["filling_factor"],
            self.bg_temp,
        )

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted_emission = predicted_emission + baseline_models["emission"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "emission",
                mu=predicted_emission,
                sigma=self.data["emission"].noise,
                observed=self.data["emission"].brightness,
            )
