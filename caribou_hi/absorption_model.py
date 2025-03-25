"""
absorption_model.py
AbsorptionModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

import pymc as pm
import pytensor.tensor as pt

from caribou_hi.hi_model import HIModel
from caribou_hi import physics


class AbsorptionModel(HIModel):
    """Definition of the AbsorptionModel model. SpecData keys must be "absorption"."""

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "absorption"."""
        # Predict optical depth spectrum (shape: spectral, clouds)
        optical_depth = physics.calc_optical_depth(
            self.data["absorption"].spectral,
            self.model["velocity"],
            10.0 ** self.model["log10_NHI"],
            self.model["tspin"],
            self.model["fwhm"],
            self.model["fwhm_L"],
        )

        # Sum over clouds
        predicted_line = 1.0 - pt.exp(-optical_depth.sum(axis=1))

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["absorption"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "absorption",
                mu=predicted,
                sigma=self.data["absorption"].noise,
                observed=self.data["absorption"].brightness,
            )
