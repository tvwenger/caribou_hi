"""
emission_model.py
EmissionModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm

from caribou_hi.hi_model import HIModel
from caribou_hi import physics


class EmissionModel(HIModel):
    """Definition of the EmissionModel model. SpecData keys must be "emission"."""

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

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "filling_factor": r"$f$",
            }
        )

    def add_priors(self, *args, **kwargs):
        """Add priors and deterministics to the model"""
        super().add_priors(*args, **kwargs)

        with self.model:
            # Filling factor
            _ = pm.Beta("filling_factor", alpha=1.0, beta=1.0, dims="cloud")

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "emission"."""
        # Predict optical depth spectrum (shape: spectral, clouds)
        optical_depth = physics.calc_optical_depth(
            self.data["emission"].spectral,
            self.model["velocity"],
            10.0 ** self.model["log10_NHI"],
            self.model["tspin"],
            self.model["fwhm"],
            self.model["fwhm_L"],
        )

        # Evaluate radiative transfer
        predicted_line = physics.radiative_transfer(
            optical_depth,
            self.model["tspin"],
            self.model["filling_factor"],
            self.bg_temp,
        )

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["emission"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "emission",
                mu=predicted,
                sigma=self.data["emission"].noise,
                observed=self.data["emission"].brightness,
            )
