"""
emission_absorption_physical_model.py
EmissionAbsorptionPhysicalModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable

import numpy as np

import pymc as pm
import pytensor.tensor as pt

from caribou_hi.hi_physical_model import HIPhysicalModel
from caribou_hi import physics


class EmissionAbsorptionPhysicalModel(HIPhysicalModel):
    """Definition of the EmissionAbsorptionPhysicalModel model. SpecData keys must be "emission" and "absorption"."""

    def __init__(self, *args, bg_temp: float = 3.77, **kwargs):
        """Initialize a new EmissionPhysicalModel instance

        Parameters
        ----------
        bg_temp : float, optional
            Assumed background temperature (K), by default 3.77
        """
        # Initialize HIPhysicalModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.bg_temp = bg_temp

    def add_priors(
        self,
        prior_ff_NHI: float = 1.0e21,
        prior_fwhm2_thermal_fraction: Iterable[float] = [2.0, 2.0],
        prior_sigma_log10_NHI: float = 0.5,
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_ff_NHI : float, optional
            Prior distribution on filling factor * column density (cm-2), by default 1.0e21, where
            ff_NHI ~ HalfNormal(sigma=prior)
        prior_fwhm2_thermal_fraction : Iterable[float], optional
            Prior distribution on thermal FWHM^2 / total FWHM^2, by default [2.0, 2.0], where
            fwhm2_thermal_fraction ~ Beta(alpha=prior[0], beta=prior[1])
        prior_sigma_log10_NHI : float, optional
            Shape parameter that defines the prior distribution on
            absorption weight / filling factor, by default 0.5, where
            wt/ff ~ LogNormal(mu=-0.5*ln(10)*prior^2, sigma=prior)
            i.e., assuming the cloud has a log-normal column density distribution with this width
        """
        # Add HIPhysicalModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # filling factor * column density (cm-2; shape: clouds)
            ff_NHI_norm = pm.HalfNormal("ff_NHI_norm", sigma=1.0, dims="cloud")
            ff_NHI = prior_ff_NHI * ff_NHI_norm

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

            # filling factor
            filling_factor = pm.Uniform(
                "filling_factor", lower=0.0, upper=1.0, dims="cloud"
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
                pt.log10(ff_NHI / filling_factor),
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

            # Absorption weight / filling factor / fwhm2_thermal (km-2 s2; shape: clouds)
            # Mean assuming underlying column density distribution is log-normal
            mu = -0.5 * (np.log(10.0) * prior_sigma_log10_NHI) ** 2.0
            wt_ff_fwhm2_thermal = pm.LogNormal(
                "wt_ff_fwhm2_thermal",
                mu=mu - pt.log(fwhm2_thermal),
                sigma=np.log(10.0) * prior_sigma_log10_NHI,
                dims="cloud",
            )
            wt_ff = wt_ff_fwhm2_thermal * fwhm2_thermal

            # Absorption weight (shape: clouds)
            wt = pm.Deterministic(
                "absorption_weight",
                wt_ff * filling_factor,
                dims="cloud",
            )

            # total optical depth (km s-1; shape: clouds)
            _ = pm.Deterministic(
                "tau_total",
                physics.calc_tau_total(wt_ff * ff_NHI / wt, tspin),
                dims="cloud",
            )

        # Add HIPhysicalModel deterministics
        super().add_deterministics()

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
