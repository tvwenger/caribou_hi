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
            "fwhm",
        ]

        # Define TeX representation of each parameter
        self.var_name_map.update(
            {
                "log10_NHI": r"log$_{10}$ $N_{\rm HI}$ (cm$^{-2}$)",
                "log10_depth": r"log$_{10}$ $d$ (pc)",
                "log10_pressure": r"log$_{10}$ $P_{\rm th}/k_B$ (K cm$^{-3}$)",
                "velocity": r"$V_{\rm LSR}$ (km s$^{-1}$)",
                "log10_n_alpha": r"log$_{10}$ $n_\alpha$ (cm$^{-3}$)",
                "log10_nth_fwhm_1pc": r"log$_{10}$ $\Delta V_{\rm nth, 1 pc}$ (km s$^{-1}$)",
                "depth_nth_fwhm_power": r"$\alpha$",
                "log10_nHI": r"log$_{10}$ $n_{\rm HI}$ (cm$^{-3}$)",
                "log10_tkin": r"log$_{10}$ $T_K$ (K)",
                "tspin": r"$T_S$ (K)",
                "depth": r"$d$ (pc)",
                "fwhm_thermal": r"$\Delta V_{\rm th}$ (km s$^{-1}$)",
                "fwhm_nonthermal": r"$\Delta V_{\rm nth}$ (km s$^{-1}$)",
                "fwhm": r"$\Delta V$ (km s$^{-1}$)",
            }
        )

    def add_priors(
        self,
        prior_log10_NHI: Iterable[float] = [20.0, 1.0],
        prior_log10_depth: Iterable[float] = [1.0, 1.0],
        prior_log10_pressure: Iterable[float] = [3.0, 1.0],
        prior_velocity: Iterable[float] = [0.0, 10.0],
        prior_log10_n_alpha: Iterable[float] = [-6.0, 1.0],
        prior_log10_nth_fwhm_1pc: Iterable[float] = [0.2, 0.1],
        prior_depth_nth_fwhm_power: Iterable[float] = [0.3, 0.1],
        prior_fwhm_L: Optional[float] = None,
        prior_baseline_coeffs: Optional[dict[str, Iterable[float]]] = None,
        ordered: bool = False,
        hyper_depth_linewidth: bool = False,
    ):
        """Add priors and deterministics to the model

        Parameters
        ----------
        prior_log10_NHI : Iterable[float], optional
            Prior distribution on log10 HI column density (cm-2), by default [20.0, 1.0], where
            log10_NHI ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_depth : Iterable[float], optional
            Prior distribution on log10 depth (pc), by default [1.0, 1.0], where
            log10_depth ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_pressure : Iterable[float], optional
            Prior distribution on log10 pressure/k_B (K cm-3), by default [3.0, 1.0], where
            log10_pressure ~ Normal(mu=prior[0], sigma=prior[1])
        prior_velocity : Iterable[float], optional
            Prior distribution on centroid velocity (km s-1), by default [0.0, 10.0], where
            velocity ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_n_alpha : Iterable[float], optional
            Prior distribution on log10 n_alpha (cm-3), by default [-6.0, 1.0], where
            n_alpha ~ Normal(mu=prior[0], sigma=prior[1])
        prior_log10_nth_fwhm_1pc : Iterable[float], optional
            Prior distribution on non-thermal line width at 1 pc, by default [0.2, 0.1], where
            log10_nth_fwhm_1pc ~ Normal(mu=prior[0], sigma=prior[1])
        prior_depth_nth_fwhm_power : Iterable[float], optional
            Prior distribution on depth vs. non-thermal line width power law index, by default [0.3, 0.1], where
            depth_nth_fwhm_power ~ Gamma(mu=prior[0], sigma=prior[1])
        prior_fwhm_L : Optional[float], optional
            Prior distribution on the pseudo-Voight Lorentzian profile line width (km/s),
            by default None, where
            fwhm_L ~ HalfNormal(sigma=prior_fwhm_L)
            If None, the line profile is assumed Gaussian.
        prior_baseline_coeffs : Optional[dict[str, Iterable[float]]], optional
            Width of normal prior distribution on the normalized baseline polynomial coefficients.
            Keys are dataset names and values are lists of length `baseline_degree+1`. If None, use
            `[1.0]*(baseline_degree+1)` for each dataset, by default None
        ordered : bool, optional
            If True, assume ordered velocities (optically thin assumption), by default False. If True, the prior
            distribution on the velocity becomes
            velocity(cloud = n) ~ prior[0] + sum_i(velocity[i < n]) + Gamma(alpha=2.0, beta=1.0/prior[1])
        hyper_depth_linewidth : bool, optional
            If True, assume that the depth-linewidth relationship is the same for every cloud, by default
            False.
        """
        # add polynomial baseline priors
        super().add_baseline_priors(prior_baseline_coeffs=prior_baseline_coeffs)

        with self.model:
            # HI column density (cm-2; shape: clouds)
            log10_NHI_norm = pm.Normal(
                "log10_NHI_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_NHI = pm.Deterministic(
                "log10_NHI",
                prior_log10_NHI[0] + prior_log10_NHI[1] * log10_NHI_norm,
                dims="cloud",
            )

            # depth (pc; shape: clouds)
            log10_depth_norm = pm.Normal(
                "log10_depth_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_depth = pm.Deterministic(
                "log10_depth",
                prior_log10_depth[0] + prior_log10_depth[1] * log10_depth_norm,
                dims="cloud",
            )

            # pressure (K cm-3; shape: clouds)
            log10_pressure_norm = pm.Normal(
                "log10_pressure_norm", mu=0.0, sigma=1.0, dims="cloud"
            )
            log10_pressure = pm.Deterministic(
                "log10_pressure",
                prior_log10_pressure[0] + prior_log10_pressure[1] * log10_pressure_norm,
                dims="cloud",
            )

            # Velocity (km/s; shape: clouds)
            if ordered:
                velocity_offset_norm = pm.Gamma(
                    "velocity_norm", alpha=2.0, beta=1.0, dims="cloud"
                )
                velocity_offset = velocity_offset_norm * prior_velocity[1]
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + pm.math.cumsum(velocity_offset),
                    dims="cloud",
                )
            else:
                velocity_norm = pm.Normal(
                    "velocity_norm",
                    mu=0.0,
                    sigma=1.0,
                    dims="cloud",
                )
                _ = pm.Deterministic(
                    "velocity",
                    prior_velocity[0] + prior_velocity[1] * velocity_norm,
                    dims="cloud",
                )

            # Lyman-alpha photon density (cm-3)
            log10_n_alpha_norm = pm.Normal("log10_n_alpha_norm", mu=0.0, sigma=1.0)
            log10_n_alpha = pm.Deterministic(
                "log10_n_alpha",
                prior_log10_n_alpha[0] + prior_log10_n_alpha[1] * log10_n_alpha_norm,
            )

            # Non-thermal FWHM at 1 pc (km s-1; shape: clouds)
            log10_nth_fwhm_1pc_norm = pm.Normal(
                "log10_nth_fwhm_1pc_norm",
                mu=0.0,
                sigma=1.0,
                dims=None if hyper_depth_linewidth else "cloud",
            )
            log10_nth_fwhm_1pc = pm.Deterministic(
                "log10_nth_fwhm_1pc",
                prior_log10_nth_fwhm_1pc[0]
                + prior_log10_nth_fwhm_1pc[1] * log10_nth_fwhm_1pc_norm,
                dims=None if hyper_depth_linewidth else "cloud",
            )

            # Non-thermal FWHM vs. depth power law index (shape: clouds)
            depth_nth_fwhm_power = pm.Gamma(
                "depth_nth_fwhm_power",
                mu=prior_depth_nth_fwhm_power[0],
                sigma=prior_depth_nth_fwhm_power[1],
                dims=None if hyper_depth_linewidth else "cloud",
            )

            # Volume density (cm-3; shape: clouds)
            log10_nHI = pm.Deterministic(
                "log10_nHI", log10_NHI - log10_depth - 18.48935, dims="cloud"
            )

            # Kinetic temperature (K; shape: clouds)
            log10_tkin = pm.Deterministic(
                "log10_tkin", log10_pressure - log10_nHI - pt.log10(1.1), dims="cloud"
            )

            # Spin temperature (K; shape: clouds)
            _ = pm.Deterministic(
                "tspin",
                physics.calc_spin_temp(
                    10.0**log10_tkin, 10.0**log10_nHI, 10.0**log10_n_alpha
                ),
                dims="cloud",
            )

            # Thermal line width (km/s; shape: clouds)
            fwhm_thermal = pm.Deterministic(
                "fwhm_thermal",
                physics.calc_thermal_fwhm(10.0**log10_tkin),
                dims="cloud",
            )

            # Non-thermal line width (km/s; shape: clouds)
            fwhm_nonthermal = pm.Deterministic(
                "fwhm_nonthermal",
                physics.calc_nonthermal_fwhm(
                    10.0**log10_depth, 10.0**log10_nth_fwhm_1pc, depth_nth_fwhm_power
                ),
                dims="cloud",
            )

            # Total line width (km/s; shape: clouds)
            _ = pm.Deterministic(
                "fwhm", pt.sqrt(fwhm_thermal**2.0 + fwhm_nonthermal**2.0), dims="cloud"
            )

            # Pseudo-Voigt profile latent variable (km/s)
            if prior_fwhm_L is not None:
                fwhm_L_norm = pm.HalfNormal("fwhm_L_norm", sigma=1.0)
                _ = pm.Deterministic("fwhm_L", prior_fwhm_L * fwhm_L_norm)
            else:
                _ = pm.Data("fwhm_L", 0.0)

    @abstractmethod
    def add_likelihood(self, *args, **kwargs):  # pragma: no cover
        """Must be defined in inhereted class."""
        pass
