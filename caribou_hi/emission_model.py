"""
emission_model.py
EmissionModel definition

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


class EmissionModel(HIModel):
    """Definition of the EmissionModel model. SpecData keys must be "emission"."""

    def __init__(self, *args, bg_temp: float = 2.7, **kwargs):
        """Initialize a new EmissionModel instance

        Parameters
        ----------
        bg_temp : float, optional
            Assumed background temperature (K), by default 2.7
        """
        # Initialize HIModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.bg_temp = bg_temp

    def add_priors(self, *args, prior_rms_emission: float = 1.0, **kwargs):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_rms_emission : float, optional
            Prior distribution on emission rms (K), by default 1.0, where
            rms_emission ~ HalfNormal(sigma=prior)
        """
        super().add_priors(*args, **kwargs)

        with self.model:
            # Spectral rms (K)
            rms_emission_norm = pm.HalfNormal("rms_emission_norm", sigma=1.0)
            _ = pm.Deterministic("rms_emission", rms_emission_norm * prior_rms_emission)

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "emission"."""
        # Predict optical depth spectrum (shape: spectral, clouds)
        optical_depth = physics.calc_optical_depth(
            self.data["emission"].spectral,
            self.model["velocity"],
            10.0 ** self.model["log10_NHI"],
            self.model["tspin"],
            self.model["fwhm"],
        )

        # Evaluate radiative transfer
        predicted_line = physics.radiative_transfer(optical_depth, self.model["tspin"], self.bg_temp)

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted = predicted_line + baseline_models["emission"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "emission",
                mu=predicted,
                sigma=self.model["rms_emission"],
                observed=self.data["emission"].brightness,
            )
