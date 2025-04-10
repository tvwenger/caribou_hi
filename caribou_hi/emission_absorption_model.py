"""
emission_absorption_model.py
EmissionAbsorptionModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm
import pytensor.tensor as pt

from caribou_hi.hi_model import HIModel
from caribou_hi import physics


class EmissionAbsorptionModel(HIModel):
    """Definition of the EmissionAbsorptionModel model. SpecData keys must be "emission" and "absorption"."""

    def __init__(self, *args, bg_temp: float = 3.77, **kwargs):
        """Initialize a new EmissionAbsorptionModel instance

        Parameters
        ----------
        bg_temp : float, optional
            Assumed background temperature (K), by default 3.77
        """
        # Initialize HIModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.bg_temp = bg_temp

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "filling_factor": r"$f$",
                "absorption_weight": r"$w_\tau$",
            }
        )

    def add_priors(self, *args, **kwargs):
        """Add priors and deterministics to the model"""
        super().add_priors(*args, **kwargs)

        with self.model:
            # Filling factor
            filling_factor = pm.Beta(
                "filling_factor", alpha=1.0, beta=1.0, dims="cloud"
            )

            # Absorption weight
            _ = pm.Beta(
                "absorption_weight",
                alpha=1.0,
                beta=1.0 - filling_factor,
                dims="cloud",
            )

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "emission"."""
        # Predict optical depth spectrum (shape: spectral, clouds)
        absorption_optical_depth = self.model["absorption_weight"] * (
            physics.calc_optical_depth(
                self.data["absorption"].spectral,
                self.model["velocity"],
                10.0 ** self.model["log10_NHI"],
                self.model["tspin"],
                self.model["fwhm"],
                self.model["fwhm_L"],
            )
        )
        emission_optical_depth = physics.calc_optical_depth(
            self.data["emission"].spectral,
            self.model["velocity"],
            10.0 ** self.model["log10_NHI"],
            self.model["tspin"],
            self.model["fwhm"],
            self.model["fwhm_L"],
        )

        # Sum over clouds
        predicted_absorption = 1.0 - pt.exp(-absorption_optical_depth.sum(axis=1))

        # Evaluate radiative transfer
        predicted_emission = physics.radiative_transfer(
            emission_optical_depth,
            self.model["tspin"],
            self.model["filling_factor"],
            self.bg_temp,
        )

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
