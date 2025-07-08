"""
hi_physical_model.py
HIPhysicalModel definition

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


class HIPhysicalModel(BaseModel):
    """Definition of the HIPhysicalModel model. This model is further extended by other models."""

    def __init__(self, *args, depth_nth_fwhm_power: float = 1 / 3, **kwargs):
        """Initialize a new HIPhysicalModel instance

        Parameters
        ----------
        depth_nth_fwhm_power : float, optional
            Assumed nonthermal FWHM vs. depth power law index, by default 1/3
        """
        # Initialize BaseModel
        super().__init__(*args, **kwargs)

        # Save inputs
        self.depth_nth_fwhm_power = depth_nth_fwhm_power

        # Select features used for posterior clustering
        self._cluster_features += [
            "velocity",
            "fwhm2",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "fwhm2": r"$\Delta V^2$ (km$^{2}$ s$^{-2}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "n_alpha": r"$n_\alpha$ (cm$^{-3}$)",
                "nth_fwhm_1pc": r"$\Delta V_{\rm nth, 1 pc}$ (km s$^{-1}$)",
                "fwhm_L": r"$\Delta V_L$ (km s$^{-1}$)",
                "fwhm2_thermal_fraction": r"$\Delta V_{\rm th}^2/\Delta V^2$",
                "fwhm2_thermal": r"$\Delta V_{\rm th}^2$ (km$^{2}$ s$^{-2}$)",
                "tkin": r"$T_K$ (K)",
                "filling_factor": r"$f$",
                "log10_NHI": r"log$_{10}$ $N_{\rm HI}$ (cm$^{-2}$)",
                "wt_ff_fwhm2_thermal": r"$w_\tau/(f \Delta V_{\rm th}^2)$ (km$^{-2}$ s$^{2}$)",
                "absorption_weight": r"$w_\tau$",
                "fwhm2_nonthermal": r"$\Delta V_{\rm nth}^2$ (km$^{2}$ s$^{-2}$)",
                "depth": r"$d$ (pc)",
                "log10_nHI": r"log$_{10}$ $n_{\rm HI}$ (cm$^{-3}$)",
                "tspin": r"$T_S$ (K)",
                "tau_total": r"$\int \tau(v) dv$ (km s$^{-1}$)",
                "log10_Pth": r"log$_{10}$ $P_{\rm th}$ (K cm$^{-3}$)",
                "log10_Pnth": r"log$_{10}$ $P_{\rm nth}$ (K cm$^{-3}$)",
                "log10_Ptot": r"log$_{10}$ $P_{\rm tot}$ (K cm$^{-3}$)",
            }
        )

    def add_priors(
        self,
        prior_fwhm2: float = 200.0,
        prior_velocity: Iterable[float] = [-10.0, 10.0],
        prior_n_alpha: float = 1.0e-6,
        prior_nth_fwhm_1pc: Iterable[float] = [1.75, 0.25],
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
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [-10.0, 10.0], where
            velocity_norm ~ Beta(alpha=2.0, beta=2.0)
            velocity ~ prior[0] + (prior[1] - prior[0]) * velocity_norm
        prior_n_alpha : Iterable[float], optional
            Prior distribution on n_alpha (cm-3), by default 1.0e-6, where
            n_alpha ~ HalfNormal(sigma=prior)
        prior_nth_fwhm_1pc : float, optional
            Prior distribution on non-thermal line width at 1 pc (km/s), by default [1.75, 0.25], where
            nth_fwhm_1pc ~ TruncatedNormal(mu=prior[0], sigma=prior[1], lower=0.0)
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

            # Non-thermal FWHM at 1 pc (km s-1; shape: clouds)
            _ = pm.TruncatedNormal(
                "nth_fwhm_1pc",
                mu=prior_nth_fwhm_1pc[0],
                sigma=prior_nth_fwhm_1pc[1],
                lower=0.0,
                dims="cloud",
            )

            # Pseudo-Voigt profile latent variable (km/s)
            if prior_fwhm_L is not None:
                fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
                _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)
            else:
                _ = pm.Data("fwhm_L", 0.0)

    def add_deterministics(self):
        """Add deterministic quantities to the model."""
        with self.model:
            # Thermal pressure (K cm-3; shape: clouds)
            log10_Pth = pm.Deterministic(
                "log10_Pth",
                pt.log10(self.model["tkin"]) + self.model["log10_nHI"],
                dims="cloud",
            )

            # Non-thermal pressure (K cm-3; shape: clouds)
            log10_Pnth = pm.Deterministic(
                "log10_Pnth",
                physics.calc_log10_nonthermal_pressure(
                    self.model["log10_nHI"], self.model["fwhm2_nonthermal"]
                ),
                dims="cloud",
            )

            # Total pressure (K cm-3; shape: clouds)
            _ = pm.Deterministic(
                "log10_Ptot",
                pt.log10(10.0**log10_Pth + 10.0**log10_Pnth),
                dims="cloud",
            )

    @abstractmethod
    def add_likelihood(self, *args, **kwargs):  # pragma: no cover
        """Must be defined in inhereted class."""
        pass
