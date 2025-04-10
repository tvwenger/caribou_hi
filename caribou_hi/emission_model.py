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

import numpy as np


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

        self.var_name_map.update(
            {
                "log10_NHI": r"log$_{10}$ $N_{\rm HI}/f$ (cm$^{-2}$)",
            }
        )

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
        # Assuming filling_factor = 1.0
        filling_factor = np.ones(self.n_clouds)
        predicted_line = physics.radiative_transfer(
            optical_depth,
            self.model["tspin"],
            filling_factor,
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
