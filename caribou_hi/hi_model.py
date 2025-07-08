"""
hi_model.py
HIModel definition

Copyright(C) 2024-2025 by
Trey V. Wenger; tvwenger@gmail.com
This code is licensed under MIT license (see LICENSE for details)
"""

from typing import Iterable, Optional
from abc import abstractmethod

import pymc as pm
import pytensor.tensor as pt

from bayes_spec import BaseModel

from caribou_hi import physics


class HIModel(BaseModel):
    """Definition of the HIModel model. This model is further extended by other models."""

    def __init__(self, *args, **kwargs):
        """Initialize a new HIModel instance"""
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Select features used for posterior clustering
        self._cluster_features += [
            "velocity",
            "fwhm2",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "fwhm2": r"$\Delta V^2$ (km$^{2}$ s$^{-2}$)",
                "log10_nHI": r"log$_{10}$ $n_{\rm HI}$ (cm$^{-3}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "n_alpha": r"$n_\alpha$ (cm$^{-3}$)",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
                "tau_total": r"$\int \tau(v) dv$ (km s$^{-1}$)",
                "TB_fwhm": r"$T_B \Delta V$ (K km s$^{-1}$)",
                "tkin_factor": r"$T_K/T_{K, \rm max}$",
                "tkin": r"$T_K$ (K)",
                "tspin": r"$T_S$ (K)",
                "filling_factor": r"$f$",
                "wt_ff_tspin": r"$w_\tau/(f T_s)$ (K$^{-1}$)",
                "absorption_weight": r"$w_\tau$",
                "log10_Pth": r"log$_{10}$ $P_{\rm th}$ (K cm$^{-3}$)",
                "log10_NHI": r"log$_{10}$ $N_{\rm HI}$ (cm$^{-2}$)",
                "log10_depth": r"log$_{10}$ $d$ (pc)",
            }
        )

    def add_priors(
        self,
        prior_fwhm2: float = 200.0,
        prior_log10_nHI: Iterable[float] = [0.0, 1.5],
        prior_velocity: Iterable[float] = [-10.0, 10.0],
        prior_n_alpha: float = 1.0e-6,
        prior_fwhm_L: Optional[float] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_fwhm2 : float, optional
            Prior distribution on FWHM^2  (km2 s-2), by default 200.0, where
            fwhm2 ~ prior * ChiSquared(nu=1)
            i.e., half-normal on FWHM
        prior_log10_nHI : Iterable[float], optional
            Prior distribution on log10 volume density (cm-3), by default [0.0, 1.5], where
            log10_nHI ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [-10.0, 10.0], where
            velocity_norm ~ Beta(alpha=2.0, beta=2.0)
            velocity ~ prior[0] + (prior[1] - prior[0]) * velocity_norm
        prior_n_alpha : Iterable[float], optional
            Prior distribution on n_alpha (cm-3), by default 1.0e-6, where
            n_alpha ~ HalfNormal(sigma=prior)
        prior_fwhm_L : Optional[float], optional
            Prior distribution on the pseudo-Voight Lorentzian profile line width (km/s),
            by default None, where
            fwhm_L ~ HalfNormal(sigma=prior_fwhm_L)
            If None, the line profile is assumed Gaussian.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # FWHM^2 (km2 s-2; shape: clouds)
            fwhm2_norm = pm.ChiSquared("fwhm2_norm", nu=1, dims="cloud")
            _ = pm.Deterministic("fwhm2", prior_fwhm2 * fwhm2_norm, dims="cloud")

            # density (cm-3; shape: clouds)
            log10_nHI_norm = pm.Normal(
                "log10_nHI_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            _ = pm.Deterministic(
                "log10_nHI",
                prior_log10_nHI[0] + prior_log10_nHI[1] * log10_nHI_norm,
                dims="cloud",
            )

            # Velocity (km/s; shape: clouds)
            velocity_norm = pm.Beta("velocity_norm", alpha=2.0, beta=2.0, dims="cloud")
            _ = pm.Deterministic(
                "velocity",
                prior_velocity[0]
                + (prior_velocity[1] - prior_velocity[0]) * velocity_norm,
                dims="cloud",
            )

            # Lyman-alpha photon density (cm-3)
            n_alpha_norm = pm.HalfNormal("n_alpha_norm", sigma=1.0, dims="cloud")
            _ = pm.Deterministic("n_alpha", prior_n_alpha * n_alpha_norm, dims="cloud")

            # Pseudo-Voigt profile latent variable (km/s)
            if prior_fwhm_L is not None:
                fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0, dims="cloud")
                _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm, dims="cloud")
            else:
                _ = pm.Data("fwhm_L", 0.0)

    def add_deterministics(self):
        """Add additional deterministic quantities to the model."""
        with self.model:
            # thermal pressure (K cm-3; shape: clouds)
            _ = pm.Deterministic(
                "log10_Pth",
                pt.log10(self.model["tkin"]) + self.model["log10_nHI"],
                dims="cloud",
            )

            # Column density (cm-2; shape: clouds)
            log10_NHI = pm.Deterministic(
                "log10_NHI",
                physics.calc_log10_column_density(
                    self.model["tau_total"], self.model["tspin"]
                ),
                dims="cloud",
            )

            # Depth (pc; shape: clouds)
            _ = pm.Deterministic(
                "log10_depth",
                log10_NHI - self.model["log10_nHI"] - 18.489351,
                dims="cloud",
            )

    @abstractmethod
    def add_likelihood(self, *args, **kwargs):  # pragma: no cover
        """Must be defined in inhereted class."""
        pass
