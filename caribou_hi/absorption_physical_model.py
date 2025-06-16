"""
absorption_physical_model.py
AbsorptionPhysicalModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import pymc as pm
import pytensor.tensor as pt
from caribou_hi import physics

from caribou_hi.hi_physical_model import HIPhysicalModel


class AbsorptionPhysicalModel(HIPhysicalModel):
    """Definition of the AbsorptionPhysicalModel model."""

    def add_priors(
        self,
        prior_NHI_fwhm2_thermal: float = 1.0e20,
        prior_fwhm2_thermal_fraction: Iterable[float] = [2.0, 2.0],
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_NHI_fwhm2_thermal : float, optional
            Prior distribution on column density / thermal FWHM2 (cm-2 km-2 s2), by default 1.0e20, where
            NHI_fwhm2_thermal ~ HalfNormal(sigma=prior)
        prior_fwhm2_thermal_fraction : Iterable[float], optional
            Prior distribution on thermal FWHM^2 / total FWHM^2, by default [2.0, 2.0], where
            fwhm2_thermal_fraction ~ Beta(alpha=prior[0], beta=prior[1])
        """
        # Add HIPhysicalModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # column density (cm-2; shape: clouds)
            NHI_fwhm2_thermal_norm = pm.HalfNormal(
                "NHI_fwhm2_thermal_norm", sigma=1.0, dims="cloud"
            )
            NHI_fwhm2_thermal = prior_NHI_fwhm2_thermal * NHI_fwhm2_thermal_norm

            # thermal FWHM^2 fraction (shape: clouds)
            fwhm2_thermal_fraction = pm.Beta(
                "fwhm2_thermal_fraction",
                alpha=prior_fwhm2_thermal_fraction[0],
                beta=prior_fwhm2_thermal_fraction[1],
                dims="cloud",
            )

            # Thermal FWHM2 (km2 s-2; shape: clouds)
            fwhm2_thermal = pm.Deterministic(
                "fwhm2_thermal",
                self.model["fwhm2"] * fwhm2_thermal_fraction,
                dims="cloud",
            )

            # Kinetic temperature
            tkin = pm.Deterministic(
                "tkin",
                physics.calc_kinetic_temp(fwhm2_thermal),
                dims="cloud",
            )

            # Non-thermal FWHM2 (km2 s-2; shape: clouds)
            fwhm2_nonthermal = pm.Deterministic(
                "fwhm2_nonthermal",
                self.model["fwhm2"] - fwhm2_thermal,
                dims="cloud",
            )

            # Depth (pc; shape: clouds)
            depth = pm.Deterministic(
                "depth",
                physics.calc_depth_nonthermal(
                    pt.sqrt(fwhm2_nonthermal),
                    self.model["nth_fwhm_1pc"],
                    self.depth_nth_fwhm_power,
                ),
                dims="cloud",
            )

            # Column density (cm-2; shape: clouds)
            log10_NHI = pm.Deterministic(
                "log10_NHI",
                pt.log10(NHI_fwhm2_thermal * fwhm2_thermal),
                dims="cloud",
            )

            # density (cm-3; shape: clouds)
            log10_nHI = pm.Deterministic(
                "log10_nHI",
                physics.calc_log10_density(log10_NHI, pt.log10(depth)),
                dims="cloud",
            )

            # Spin temperature (K; shape: clouds)
            tspin = pm.Deterministic(
                "tspin",
                physics.calc_spin_temp(
                    tkin,
                    10.0**log10_nHI,
                    self.model["n_alpha"],
                ),
                dims="cloud",
            )

            # total optical depth (km s-1; shape: clouds)
            _ = pm.Deterministic(
                "tau_total",
                physics.calc_tau_total(10.0**log10_NHI, tspin),
                dims="cloud",
            )

        # Add HIPhysicalModel deterministics
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
