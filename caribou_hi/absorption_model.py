"""
absorption_model.py
AbsorptionModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import pymc as pm
import pytensor.tensor as pt
from caribou_hi import physics

from caribou_hi.hi_model import HIModel


class AbsorptionModel(HIModel):
    """Definition of the AbsorptionModel model."""

    def add_priors(
        self,
        prior_tau_total: float = 1.0,
        prior_tkin_factor: Iterable[float] = [2.0, 2.0],
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_tau_total : float, optional
            Prior distribution on total optical depth (km s-1), by default 1.0, where
            tau_total ~ HalfNormal(sigma=prior)
        prior_tkin_factor : Iterable[float], optional
            Prior distribution on kinetic temperature factor, by default [2.0, 2.0], where
            tkin_factor ~ Beta(alpha=prior[0], beta=prior[1])
            tkin = tkin_factor * tkin_max
        """
        # Add HIModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # total optical depth (km s-1; shape: clouds)
            tau_total_norm = pm.HalfNormal("tau_total_norm", sigma=1.0, dims="cloud")
            _ = pm.Deterministic(
                "tau_total", prior_tau_total * tau_total_norm, dims="cloud"
            )

            # maximum kinetic temperature (K; shape: clouds)
            tkin_max = physics.calc_kinetic_temp(self.model["fwhm2"])

            # kinetic temperature (K; shape: clouds)
            tkin_factor = pm.Beta(
                "tkin_factor",
                alpha=prior_tkin_factor[0],
                beta=prior_tkin_factor[1],
                dims="cloud",
            )
            tkin = pm.Deterministic("tkin", tkin_factor * tkin_max, dims="cloud")

            # Spin temperature (K; shape: clouds)
            _ = pm.Deterministic(
                "tspin",
                physics.calc_spin_temp(
                    tkin,
                    10.0 ** self.model["log10_nHI"],
                    self.model["n_alpha"],
                ),
                dims="cloud",
            )

        # Add HIModel deterministics
        super().add_deterministics()

    def add_likelihood(self):
        """Add likelihood to the model. SpecData key must be "absorption"."""
        # Evaluate line profile (shape: spectral, clouds)
        line_profile = physics.calc_pseudo_voigt(
            self.data["absorption"].spectral,
            self.model["velocity"],
            self.model["fwhm2"],
            self.model["fwhm_L"],
        )

        # Optical depth spectra (shape: spectral, clouds)
        optical_depth = self.model["tau_total"] * line_profile

        # Sum over clouds
        predicted_absorption = 1.0 - pt.exp(-optical_depth.sum(axis=1))

        # Add baseline model
        baseline_models = self.predict_baseline()
        predicted_absorption = predicted_absorption + baseline_models["absorption"]

        with self.model:
            # Evaluate likelihood
            _ = pm.Normal(
                "absorption",
                mu=predicted_absorption,
                sigma=self.data["absorption"].noise,
                observed=self.data["absorption"].brightness,
            )
