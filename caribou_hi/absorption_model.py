"""
absorption_model.py
AbsorptionModel definition

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

import pymc as pm


from caribou_hi.hi_model import HIModel
from caribou_hi import physics


class AbsorptionModel(HIModel):
    """Definition of the AbsorptionModel model. SpecData keys must be "absorption"."""

    def __init__(self, *args, **kwargs):
        """Initialize a new AbsorptionModel instance"""
        # Initialize HIModel
        super().__init__(*args, **kwargs)

    def add_priors(self, *args, prior_rms_absorption: float = 0.01, **kwargs):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_rms_absorption : float, optional
            Prior distribution on optical depth rms, by default 0.01, where
            rms_absorption ~ HalfNormal(sigma=prior)
        """
        super().add_priors(*args, **kwargs)

        with self.model:
            # Optical depth rms
            rms_absorption_norm = pm.HalfNormal("rms_absorption_norm", sigma=1.0)
            _ = pm.Deterministic("rms_absorption", rms_absorption_norm * prior_rms_absorption)

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "absorption"."""
        # Predict optical depth spectrum (shape: spectral, clouds)
        optical_depth = physics.calc_optical_depth(
            self.data["absorption"].spectral,
            self.model["velocity"],
            10.0 ** self.model["log10_NHI"],
            self.model["tspin"],
            self.model["fwhm"],
        )

        # Sum over clouds
        predicted_line = optical_depth.sum(axis=1)

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["absorption"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "absorption",
                mu=predicted,
                sigma=self.model["rms_absorption"],
                observed=self.data["absorption"].brightness,
            )
