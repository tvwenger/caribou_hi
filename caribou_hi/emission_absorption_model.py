"""
emission_absorption_model.py
EmissionAbsorptionModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Optional, Iterable

import numpy as np

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

    def add_priors(
        self,
        prior_TB_fwhm: float = 50.0,
        prior_tkin_factor: Iterable[float] = [2.0, 2.0],
        prior_sigma_log10_NHI: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_TB_fwhm : float, optional
            Prior distribution on brightness temperature x FWHM (K km s), by default 50.0, where
            TB_fwhm ~ HalfNormal(sigma=prior)
        prior_tkin_factor : Iterable[float], optional
            Prior distribution on kinetic temperature factor, by default [2.0, 2.0], where
            tkin_factor ~ Beta(alpha=prior[0], beta=prior[1])
            tkin = tkin_factor * tkin_max
        prior_sigma_log10_NHI : Optional[float], optional
            Shape parameter that defines the prior distribution on
            absorption weight / filling factor, by default None, where
            wt/ff ~ LogNormal(mu=-0.5*ln(10)*prior^2, sigma=prior)
            i.e., assuming the cloud has a log-normal column density distribution with this width
            If None, then the absorption weight is assumed to be 1 (i.e., same column density
            probed in emission and absorption)
        """
        # Add HIModel priors
        super().add_priors(*args, **kwargs)

        with self.model:
            # TB_fwhm = ff * Tspin * (1 - exp(-tau_peak)) * fwhm (K km s-1; shape: clouds)
            TB_fwhm_norm = pm.HalfNormal("TB_fwhm_norm", sigma=1.0, dims="cloud")
            TB_fwhm = pm.Deterministic(
                "TB_fwhm",
                prior_TB_fwhm * TB_fwhm_norm,
                dims="cloud",
            )

            # minimum kinetic temperature == TB (K; shape: clouds)
            tkin_min = TB_fwhm / pt.sqrt(self.model["fwhm2"])

            # maximum kinetic temperature (K; shape: clouds)
            tkin_max = physics.calc_kinetic_temp(self.model["fwhm2"])

            # kinetic temperature (K; shape: clouds)
            tkin_factor_norm = pm.Beta(
                "tkin_factor_norm",
                alpha=prior_tkin_factor[0],
                beta=prior_tkin_factor[1],
                dims="cloud",
            )
            tkin = pm.Deterministic(
                "tkin",
                pt.switch(
                    pt.gt(tkin_min, tkin_max),
                    tkin_max,
                    tkin_factor_norm * (tkin_max - tkin_min) + tkin_min,
                ),
                dims="cloud",
            )

            # Spin temperature (K; shape: clouds)
            tspin = pm.Deterministic(
                "tspin",
                physics.calc_spin_temp(
                    tkin,
                    10.0 ** self.model["log10_nHI"],
                    self.model["n_alpha"],
                ),
                dims="cloud",
            )

            # Minimum filling factor == TB / Tspin
            filling_factor_min = tkin_min / tspin

            # filling factor
            filling_factor_norm = pm.Uniform(
                "filling_factor_norm", lower=0.0, upper=1.0, dims="cloud"
            )
            filling_factor = pm.Deterministic(
                "filling_factor",
                pt.switch(
                    pt.gt(filling_factor_min, 1.0),
                    1.0,
                    filling_factor_norm * (1.0 - filling_factor_min)
                    + filling_factor_min,
                ),
                dims="cloud",
            )

            # (1 - exp(-tau_peak))
            exp_tau_peak = filling_factor_min / filling_factor
            exp_tau_peak = pt.clip(exp_tau_peak, 0.0, 0.9999)
            tau_peak = -pt.log(1.0 - exp_tau_peak)

            # total optical depth (km s-1; shape: clouds)
            const = np.sqrt(2.0 * np.pi) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            _ = pm.Deterministic(
                "tau_total",
                tau_peak * pt.sqrt(self.model["fwhm2"]) * const,
                dims="cloud",
            )

            if prior_sigma_log10_NHI is None:
                _ = pm.Data("absorption_weight", np.ones(self.n_clouds), dims="cloud")
            else:
                # Absorption weight / filling factor / tspin (K-1; shape: clouds)
                # Mean assuming underlying column density distribution is log-normal
                mu = -0.5 * (np.log(10.0) * prior_sigma_log10_NHI) ** 2.0
                wt_ff_tspin = pm.LogNormal(
                    "wt_ff_tspin",
                    mu=mu - pt.log(tspin),
                    sigma=np.log(10.0) * prior_sigma_log10_NHI,
                    dims="cloud",
                )
                wt_ff = wt_ff_tspin * tspin

                # Absorption weight (shape: clouds)
                _ = pm.Deterministic(
                    "absorption_weight",
                    wt_ff * filling_factor,
                    dims="cloud",
                )

        # Add HIModel deterministics
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
