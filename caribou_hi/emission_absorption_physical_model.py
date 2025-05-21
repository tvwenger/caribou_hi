"""
emission_absorption_physical_model.py
EmissionAbsorptionPhysicalModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import numpy as np

import pymc as pm
import pytensor.tensor as pt

from caribou_hi.emission_physical_model import EmissionPhysicalModel
from caribou_hi import physics


class EmissionAbsorptionPhysicalModel(EmissionPhysicalModel):
    """Definition of the EmissionAbsorptionPhysicalModel model. SpecData keys must be "emission" and "absorption"."""

    def add_priors(
        self,
        prior_sigma_log10_NHI: float = 0.5,
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_sigma_log10_NHI : float, optional
            Shape parameter that defines the prior distribution on
            absorption weight / filling factor / thermal FWHM^2, by default 0.5, where
            log10(ff/ww/fwhm2_thermal) ~ Normal(mu=-ln(10)*prior^2/2-log10(fwhm2_thermal), sigma=prior)
            i.e., assuming the cloud has a log-normal column density distribution with this width
        """
        # Add EmissionPhysicalModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # Absorption weight / filling factor / thermal FWHM2 (km-2 s2; shape: clouds)
            mu = -np.log(10.0) * prior_sigma_log10_NHI**2.0 / 2.0 - pt.log10(
                self.model["fwhm2_thermal"]
            )
            log10_wt_ff_fwhm2_thermal = pm.Normal(
                "log10_wt_ff_fwhm2_thermal",
                mu=mu,
                sigma=prior_sigma_log10_NHI,
                dims="cloud",
            )

            # Absorption weight (shape: clouds)
            _ = pm.Deterministic(
                "absorption_weight",
                10.0**log10_wt_ff_fwhm2_thermal
                * self.model["filling_factor"]
                * self.model["fwhm2_thermal"],
                dims="cloud",
            )

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "emission" and "absorption"."""
        # Evaluate line profile (shape: spectral, clouds)
        line_profile_emission = physics.calc_pseudo_voigt(
            self.data["emission"].spectral,
            self.model["velocity"],
            self.model["fwhm2"],
            self.model["fwhm_L"],
        )
        line_profile_absorption = physics.calc_pseudo_voigt(
            self.data["absorption"].spectral,
            self.model["velocity"],
            self.model["fwhm2"],
            self.model["fwhm_L"],
        )

        # Optical depth spectra (shape: spectral, clouds)
        optical_depth_emission = self.model["tau_total"] * line_profile_emission
        optical_depth_absorption = (
            self.model["absorption_weight"]
            * self.model["tau_total"]
            * line_profile_absorption
        )

        # Evaluate radiative transfer
        predicted_emission = physics.radiative_transfer(
            optical_depth_emission,
            self.model["tspin"],
            self.model["filling_factor"],
            self.bg_temp,
        )

        # Predict absorption
        predicted_absorption = 1.0 - pt.exp(-optical_depth_absorption.sum(axis=1))

        # Add baseline models
        baseline_models = self.predict_baseline()
        predicted_absorption = predicted_absorption + baseline_models["absorption"]
        predicted_emission = predicted_emission + baseline_models["emission"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "absorption",
                mu=predicted_absorption,
                sigma=self.data["absorption"].noise,
                observed=self.data["absorption"].brightness,
            )
            _ = pm.Normal(
                "emission",
                mu=predicted_emission,
                sigma=self.data["emission"].noise,
                observed=self.data["emission"].brightness,
            )
