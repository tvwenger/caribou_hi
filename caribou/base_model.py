"""
base_model.py - BaseModel definition

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

Changelog:
Trey Wenger - March 2024
"""

import os
import warnings
from collections.abc import Iterable

import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
import pytensor.tensor as pt
import arviz as az
import arviz.labels as azl
import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import norm
import matplotlib.pyplot as plt
import graphviz

from caribou.cluster_posterior import cluster_posterior
from caribou import utils, plots
from caribou.nuts import init_nuts


class BaseModel:
    """
    BaseModel defines functions and attributes common to all model definitions.
    """

    def __init__(
        self,
        data: dict,
        n_clouds: int,
        baseline_degree: int = 0,
        seed: int = 1234,
        verbose: bool = False,
    ):
        """
        Initialize a new model

        Inputs:
            data :: dictionary
                Dictionary with keys
                "emission_velocity", "emission_spectrum", "emission_noise",
                "absorption_velocity", "absorption_spectrum", "absorption_noise"
                where
                data['emission_velocity'] contains the emission spectral axis (km/s)
                data['emission_spectrum'] contains the emission spectrum (brightness temp, K)
                data['emission_noise'] contains the emission rms noise (brightness temp, K)
                data['absorption_velocity'] contains the absorption spectral axis (km/s)
                data['absorption_spectrum'] contains the absorption spectrum (optical depth)
                data['absorption_noise'] contains the absorption rms noise (optical depth)
            n_clouds :: integer
                Number of cloud components
            baseline_degree :: integer
                Degree of the polynomial baseline used for both the
                emission and absorption spectra.
            seed :: integer
                Random seed
            verbose :: boolean
                Print extra info
        """
        self.n_clouds = n_clouds
        self.baseline_degree = baseline_degree
        self.seed = seed
        self.verbose = verbose
        self.data = data

        # center and normalize velocity
        self.data["emission_velocity_norm"] = (
            self.data["emission_velocity"] - np.mean(self.data["emission_velocity"])
        ) / np.std(self.data["emission_velocity"])
        self.data["absorption_velocity_norm"] = (
            self.data["absorption_velocity"] - np.mean(self.data["absorption_velocity"])
        ) / np.std(self.data["absorption_velocity"])

        # normalize data by the noise
        self.data["emission_spectrum_norm"] = (
            self.data["emission_spectrum"] / self.data["emission_noise"]
        )
        self.data["absorption_spectrum_norm"] = (
            self.data["absorption_spectrum"] / self.data["absorption_noise"]
        )

        # Initialize the model
        self.model = pm.Model(
            coords={
                "v_em": self.data["emission_velocity"],
                "v_abs": self.data["absorption_velocity"],
                "coeff": range(self.baseline_degree + 1),
                "cloud": range(self.n_clouds),
            }
        )
        with self.model:
            for key, value in self.data.items():
                if "emission" in key:
                    dims = "v_em"
                else:
                    dims = "v_abs"
                _ = pm.ConstantData(key, value, dims=dims)
        self._n_data = len(self.data["emission_velocity"]) + len(
            self.data["absorption_velocity"]
        )

        # Model parameters
        self.baseline_params = ["emission_coeffs", "absorption_coeffs"]
        self.hyper_params = []
        self.cloud_params = []
        self.deterministics = []

        # Parameters used for posterior clustering
        self._cluster_features = []

        # Arviz labeller map
        self.var_name_map = {
            "emission_coeffs": r"$\beta_{T}$",
            "absorption_coeffs": r"$\beta_{\tau}$",
        }

        # set results and convergence checks
        self.reset_results()

    @property
    def _n_params(self):
        """
        Determine the number of model parameters.
        """
        return (
            len(self.cloud_params) * self.n_clouds
            + len(self.baseline_params) * (self.baseline_degree + 1)
            + len(self.hyper_params)
        )

    @property
    def _get_unique_solution(self):
        """
        Return the unique solution index (0) if there is a unique
        solution, otherwise raise an exception.
        """
        if not self.unique_solution:
            raise ValueError("There is not a unique solution. Must supply solution.")
        return 0

    @property
    def labeller(self):
        """
        Get the arviz labeller.
        """
        return azl.MapLabeller(var_name_map=self.var_name_map)

    @property
    def unique_solution(self):
        """
        Check if posterior samples suggest a unique solution
        """
        if self.solutions is None:
            raise ValueError("No solutions. Try solve()")
        return len(self.solutions) == 1

    def _add_spectral_likelihood(self):
        """
        Evaluate the baseline model, perform radiative transfer,
        and add spectral likelihood to the model.

        Inputs: None

        Returns: Nothing
        """
        with self.model:
            # Spin temperature (K)
            log10_spin_temp = pm.Deterministic(
                "log10_spin_temp",
                utils.calc_log10_spin_temp(
                    self.model["log10_kinetic_temp"],
                    self.model["log10_density"],
                    self.model["log10_n_alpha"],
                ),
                dims="cloud",
            )
            spin_temp = 10.0**log10_spin_temp

            # Thermal line width (km/s)
            log10_thermal_fwhm = pm.Deterministic(
                "log10_thermal_fwhm",
                1.33 + 0.5 * (self.model["log10_kinetic_temp"] - 4.0),
                dims="cloud",
            )

            # Depth (pc)
            log10_depth = pm.Deterministic(
                "log10_depth",
                self.model["log10_NHI"] - self.model["log10_density"] - 18.48935,
                dims="cloud",
            )

            # Nonthermal line width from Larson's Law (km/s)
            log10_nonthermal_fwhm = pm.Deterministic(
                "log10_nonthermal_fwhm",
                self.model["log10_larson_linewidth"]
                + self.model["larson_power"] * log10_depth,
                dims="cloud",
            )

            # Total line width (km/s)
            log10_fwhm = pm.Deterministic(
                "log10_fwhm",
                0.5
                * pt.log10(
                    10.0 ** (2.0 * log10_thermal_fwhm)
                    + 10.0 ** (2.0 * log10_nonthermal_fwhm)
                ),
                dims="cloud",
            )
            fwhm = 10.0**log10_fwhm
            # 0.37196 = pt.log10(2.0 * pt.sqrt(2.0 * pt.log(2.0)))
            log10_sigma = log10_fwhm - 0.37196

            # Peak optical depth
            # const = pt.log10(1.82243e18 * pt.sqrt(2.0 * np.pi))  # cm-2 (K km s-1)-1
            const = 18.6597408
            log10_peak_tau = pm.Deterministic(
                "log10_peak_tau",
                self.model["log10_NHI"] - log10_spin_temp - const - log10_sigma,
                dims="cloud",
            )
            peak_tau = 10.0**log10_peak_tau

            # spectra
            pred_emission, pred_absorption, _, _ = utils.radiative_transfer(
                self.model["emission_velocity"],
                self.model["absorption_velocity"],
                peak_tau,
                spin_temp,
                self.model["velocity"],
                fwhm,
            )

            # baseline
            emission_baseline_norm = pt.sum(
                [
                    self.model["emission_coeffs"][i]
                    * self.model["emission_velocity_norm"] ** i
                    for i in range(self.baseline_degree + 1)
                ],
                axis=0,
            )
            absorption_baseline_norm = pt.sum(
                [
                    self.model["absorption_coeffs"][i]
                    * self.model["absorption_velocity_norm"] ** i
                    for i in range(self.baseline_degree + 1)
                ],
                axis=0,
            )

            # Normalized spectral likelihood
            emission_mu = (
                pred_emission / self.model["emission_noise"] + emission_baseline_norm
            )
            _ = pm.Normal(
                "emission_norm",
                mu=emission_mu,
                sigma=1.0,
                observed=self.model["emission_spectrum_norm"],
                dims="v_em",
            )
            absorption_mu = (
                pred_absorption / self.model["absorption_noise"]
                + absorption_baseline_norm
            )
            _ = pm.Normal(
                "absorption_norm",
                mu=absorption_mu,
                sigma=1.0,
                observed=self.model["absorption_spectrum_norm"],
                dims="v_abs",
            )

    def _add_thermal_balance_likelihood(
        self, prior_log10_thermal_ratio: Iterable[float]
    ):
        """
        Evaluate the net cooling model and add the thermal
        balance likelihood.

        Inputs:
            prior_log10_thermal_ratio :: 2-length array of scalars
                Prior distribution on log10(cooling/heating), where:
                log10_thermal_ratio ~ Normal(mu=prior[0], sigma=prior[1])

        Returns: Nothing
        """
        with self.model:
            # Assume half path-length for effective radiation field
            log_NHI_eff = self.model["log10_NHI"] - 0.30103
            log10_heating, log10_cooling = utils.calc_log10_heating_cooling(
                self.model["log10_density"],
                self.model["log10_G0"],
                log_NHI_eff,
                self.model["log10_kinetic_temp"],
                self.model["log10_cr_ion_rate"],
                self.model["log10_xCII"],
                self.model["log10_xO"],
                self.model["log10_inv_pah_recomb"],
            )

            # Heating and cooling rates (erg s-1 cm-3)
            log10_cooling = pm.Deterministic(
                "log10_cooling",
                log10_cooling,
                dims="cloud",
            )
            log10_heating = pm.Deterministic(
                "log10_heating",
                log10_heating,
                dims="cloud",
            )
            log10_thermal_ratio = pm.Deterministic(
                "log10_thermal_ratio", log10_cooling - log10_heating, dims="cloud"
            )

            # Thermal equilibrium likelihood
            _ = pm.Normal(
                "log10_thermal_ratio_offset",
                mu=prior_log10_thermal_ratio[0] - log10_thermal_ratio,
                sigma=prior_log10_thermal_ratio[1],
                observed=self.model["log10_equal_thermal_ratio"],
                dims="cloud",
            )

    def reset_results(self):
        """
        Reset results and convergence checks.

        Inputs: None
        Returns: Nothing
        """
        self.approx: pm.Approximation = None
        self.trace: az.InferenceData = None
        self.solutions = None
        self._good_chains = None
        self._chains_converged: bool = None

    def null_bic(self):
        """
        Evaluate the BIC for the null hypothesis (baseline only, no clouds)

        Inputs: None
        Returns: Nothing
        """
        # fit polynomial baseline
        emission_baseline = Polynomial.fit(
            self.data["emission_velocity"],
            self.data["emission_spectrum"],
            self.baseline_degree,
        )(self.data["emission_velocity"])
        absorption_baseline = Polynomial.fit(
            self.data["absorption_velocity"],
            self.data["absorption_spectrum"],
            self.baseline_degree,
        )(self.data["absorption_velocity"])

        # evaluate likelihood
        emission = self.data["emission_spectrum"] - emission_baseline
        absorption = self.data["absorption_spectrum"] - absorption_baseline
        lnlike = (
            norm.logpdf(emission, scale=self.data["emission_noise"]).sum()
            + norm.logpdf(absorption, scale=self.data["absorption_noise"]).sum()
        )

        n_params = 2 * (self.baseline_degree + 1)
        return n_params * np.log(self._n_data) - 2.0 * lnlike

    def lnlike_mean_point_estimate(self, chain: int = None, solution: int = None):
        """
        Evaluate model log-likelihood at the mean point estimate of posterior samples.

        Inputs:
            chain :: None or integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only using un-clustered posterior samples.
            solution :: None or integer
                Solution index
                If chain is None and solution is None:
                    If there is a unique solution, use that
                    Otherwise, raise an exception
                If chain is None and solution is not None:
                    Use this solution index
                If chain is not None:
                    This parameter has no effect

        Returns: lnlike
            lnlike :: scalar
                Log likelihood at point
        """
        if chain is None and solution is None:
            solution = self._get_unique_solution

        # mean point estimate
        if chain is None:
            point = self.trace[f"solution_{solution}"].mean(dim=["chain", "draw"])
        else:
            point = self.trace.posterior.sel(chain=chain).mean(dim=["draw"])

        # RV names and transformations
        params = {}
        for rv in self.model.free_RVs:
            name = rv.name
            param = self.model.rvs_to_values[rv]
            transform = self.model.rvs_to_transforms[rv]
            if transform is None:
                params[param] = point[name].data
            else:
                params[param] = transform.forward(
                    point[name].data, *rv.owner.inputs
                ).eval()

        return float(self.model.logp().eval(params))

    def bic(self, chain: int = None, solution: int = None):
        """
        Calculate the Bayesian information criterion at the mean point estimate.

        Inputs:
            chain :: None or integer
                If None (default), evaluate BIC across all chains using
                clustered posterior samples. Otherwise, evaluate BIC for
                this chain only using un-clustered posterior samples.
            solution :: None or integer
                Solution index
                If chain is None and solution is None:
                    If there is a unique solution, use that
                    Otherwise, raise an exception
                If chain is None and solution is not None:
                    Use this solution index
                If chain is not None:
                    This parameter has no effect

        Returns: bic
            bic :: scalar
                Bayesian information criterion
        """
        lnlike = self.lnlike_mean_point_estimate(chain=chain, solution=solution)
        return self._n_params * np.log(self._n_data) - 2.0 * lnlike

    def good_chains(self, mad_threshold: float = 5.0):
        """
        Identify bad chains as those with deviant BICs.

        Inputs:
            mad_threshold :: scalar
                Chains are good if they have BICs within {mad_threshold} * MAD of the median BIC.

        Returns: good_chains
            good_chains :: 1-D array of integers
                Chains that appear converged
        """
        if self.trace is None:
            raise ValueError("Model has no posterior samples. Try fit() or sample().")

        # check if already determined
        if self._good_chains is not None:
            return self._good_chains

        # if the trace has fewer than 2 chains, we assume they're both ok so we can run
        # convergence diagnostics
        if len(self.trace.posterior.chain) < 3:
            self._good_chains = self.trace.posterior.chain.data
            return self._good_chains

        # per-chain BIC
        bics = np.array(
            [self.bic(chain=chain) for chain in self.trace.posterior.chain.data]
        )
        mad = np.median(np.abs(bics - np.median(bics)))
        good = np.abs(bics - np.median(bics)) < mad_threshold * mad

        self._good_chains = self.trace.posterior.chain.data[good]
        return self._good_chains

    def set_priors(self):
        """
        Add normalized baseline priors to model.

        Inputs: None
        Returns: Nothing
        """
        with self.model:
            # polynomial baseline coefficients
            _ = pm.Normal(
                "emission_coeffs",
                mu=0.0,
                sigma=1.0,
                dims="coeff",
            )
            _ = pm.Normal(
                "absorption_coeffs",
                mu=0.0,
                sigma=1.0,
                dims="coeff",
            )

    def prior_predictive_check(self, samples: int = 50, plot_fname: str = None):
        """
        Generate prior predictive samples, and optionally plot the outcomes.

        Inputs:
            samples :: integer
                Number of prior predictive samples to generate
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing prior and prior predictive samples
        """
        with self.model:
            trace = pm.sample_prior_predictive(samples=samples, random_seed=self.seed)
            # add un-normalized predictive
            trace.prior_predictive["emission"] = (
                trace.prior_predictive["emission_norm"] * self.data["emission_noise"]
            )
            trace.prior_predictive["absorption"] = (
                trace.prior_predictive["absorption_norm"]
                * self.data["absorption_noise"]
            )

        if plot_fname is not None:
            plots.plot_predictive(self.data, trace.prior_predictive, plot_fname)

        return trace

    def posterior_predictive_check(self, thin: int = 100, plot_fname: str = None):
        """
        Generate posterior predictive samples, and optionally plot the outcomes.

        Inputs:
            thin :: integer
                Thin posterior samples by keeping one in {thin}
            plot_fname :: string
                If not None, generate a plot of the outcomes over
                the data, and save to this filename.

        Returns: predictive
            predictive :: InferenceData
                Object containing posterior and posterior predictive samples
        """
        with self.model:
            trace = pm.sample_posterior_predictive(
                self.trace.sel(chain=self.good_chains(), draw=slice(None, None, thin)),
                extend_inferencedata=True,
                random_seed=self.seed,
            )
            # add un-normalized predictive
            trace.posterior_predictive["emission"] = (
                trace.posterior_predictive["emission_norm"]
                * self.data["emission_noise"]
            )
            trace.posterior_predictive["absorption"] = (
                trace.posterior_predictive["absorption_norm"]
                * self.data["absorption_noise"]
            )

        if plot_fname is not None:
            plots.plot_predictive(
                self.data,
                trace.posterior_predictive,
                plot_fname,
                posterior=trace.posterior,
            )

        return trace

    def fit(
        self,
        n: int = 500_000,
        draws: int = 1_000,
        rel_tolerance: float = 0.01,
        abs_tolerance: float = 0.01,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Fit posterior using variational inference (VI). If you get NaNs
        during optimization, try increasing the learning rate.

        Inputs:
            n :: integer
                Number of VI iterations
            draws :: integer
                Number of samples to draw from fitted posterior
            rel_tolerance :: scalar
                Relative parameter tolerance for VI convergence
            abs_tolerance :: scalar
                Absolute parameter tolerance for VI convergence
            learning_rate :: scalar
                adagrad_window learning rate. Try increasing if you get NaNs
            **kwargs :: additional keyword arguments
                Additional arguments passed to pymc.fit
                (method)

        Returns: Nothing
        """
        # reset convergence checks
        self.reset_results()

        with self.model:
            callbacks = [
                CheckParametersConvergence(tolerance=rel_tolerance, diff="relative"),
                CheckParametersConvergence(tolerance=abs_tolerance, diff="absolute"),
            ]
            self.approx = pm.fit(
                n=n,
                random_seed=self.seed,
                progressbar=self.verbose,
                callbacks=callbacks,
                obj_optimizer=pm.adagrad_window(learning_rate=learning_rate),
                **kwargs,
            )
            self.trace = self.approx.sample(draws)

    def sample(
        self,
        init: str = "advi+adapt_diag",
        n_init: int = 500_000,
        chains: int = 4,
        init_kwargs: dict = None,
        nuts_kwargs: dict = None,
        **kwargs,
    ):
        """
        Sample posterior distribution using MCMC.

        Inputs:
            init :: string
                Initialization strategy
            n_init :: integer
                Number of initialization iterations
            chains :: integer
                Number of chains
            init_kwargs :: dictionary
                Keyword arguments passed to init_nuts
                (tolerance, learning_rate)
            nuts_kwargs :: dictionary
                Keyword arguments passed to pm.NUTS
                (target_accept)
            **kwargs :: additional keyword arguments
                Keyword arguments passed to pm.sample
                (cores, tune, draws)

        Returns: Nothing
        """
        # reset convergence checks
        self.reset_results()

        if init == "auto":
            init = "jitter+adapt_diag"

        if init_kwargs is None:
            init_kwargs = {}
        if nuts_kwargs is None:
            nuts_kwargs = {}

        # attempt custom initialization
        initial_points, step = init_nuts(
            self.model,
            init=init,
            n_init=n_init,
            chains=chains,
            nuts_kwargs=nuts_kwargs,
            seed=self.seed,
            verbose=self.verbose,
            **init_kwargs,
        )

        # if we're using custom initialization, then drop nuts
        # arguments from pm.sample
        if initial_points is not None:
            nuts_kwargs = {}

        with self.model:
            self.trace = pm.sample(
                init=init,
                initvals=initial_points,
                step=step,
                chains=chains,
                progressbar=self.verbose,
                discard_tuned_samples=False,
                compute_convergence_checks=False,
                random_seed=self.seed,
                **nuts_kwargs,
                **kwargs,
            )

        # diagnostics
        if self.verbose:
            # converged chains
            good_chains = self.good_chains()
            if len(good_chains) < len(self.trace.posterior.chain):
                print(f"Only {len(good_chains)} chains appear converged.")

            # divergences
            num_divergences = self.trace.sample_stats.diverging.sel(
                chain=self.good_chains()
            ).data.sum()
            if num_divergences > 0:
                print(f"There were {num_divergences} divergences in converged chains.")

    def solve(self, p_threshold=0.9):
        """
        Cluster posterior samples and determine unique solutions. Adds
        new groups to self.trace called "solution_{idx}" for the posterior
        samples of each unique solution.

        Inputs:
            p_threshold :: scalar
                p-value threshold for considering a unique solution

        Returns: Nothing
        """
        # Drop solutions if they already exist in trace
        for group in list(self.trace.groups()):
            if "solution" in group:
                del self.trace[group]

        self.solutions = []
        solutions = cluster_posterior(
            self.trace.posterior.sel(chain=self.good_chains()),
            self.n_clouds,
            self._cluster_features,
            p_threshold=p_threshold,
            seed=self.seed,
        )

        # convergence check
        unique_solution = len(solutions) == 1
        if self.verbose:
            if unique_solution:
                print("GMM converged to unique solution")
            else:
                print(f"GMM found {len(solutions)} unique solutions")
                for solution_idx, solution in enumerate(solutions):
                    print(
                        f"Solution {solution_idx}: chains {list(solution['chains'].keys())}"
                    )

        # labeling degeneracy check
        for solution_idx, solution in enumerate(solutions):
            chain_order = np.array(
                [chain["label_order"] for chain in solution["chains"].values()]
            )
            if self.verbose and not np.all(chain_order == solution["label_order"]):
                print(f"Chain label order mismatch in solution {solution_idx}")
                for chain, order in solution["chains"].items():
                    print(f"Chain {chain} order: {order['label_order']}")
                print(f"Adopting (first) most common order: {solution['label_order']}")

            # Add solution to the trace
            with warnings.catch_warnings(action="ignore"):
                self.trace.add_groups(
                    **{
                        f"solution_{solution_idx}": solution["posterior_clustered"],
                        "coords": solution["coords"],
                        "dims": solution["dims"],
                    }
                )
                self.solutions.append(solution_idx)

    def plot_graph(self, dotfile: str, ext: str):
        """
        Generate dot plot of model graph.

        Inputs:
            dotfile :: string
                Where graphviz source is saved
            ext :: string
                Rendered image is {dotfile}.{ext}

        Returns: Nothing
        """
        gviz = pm.model_to_graphviz(self.model)
        gviz.graph_attr["rankdir"] = "TB"
        gviz.graph_attr["splines"] = "ortho"
        gviz.graph_attr["newrank"] = "false"
        unflat = gviz.unflatten(stagger=3)

        # clean up
        source = []
        for line in unflat.source.splitlines():
            # rename normalized data/likelihood vars
            line = line.replace("emission_velocity_norm", "emission_velocity")
            line = line.replace("emission_norm", "emission")
            line = line.replace("emission_spectrum_norm", "emission_spectrum")
            line = line.replace("absorption_velocity_norm", "absorption_velocity")
            line = line.replace("absorption_norm", "absorption")
            line = line.replace("absorption_spectrum_norm", "absorption_spectrum")
            source.append(line)

        # save and render
        with open(dotfile, "w", encoding="ascii") as f:
            f.write("\n".join(source))
        graphviz.render("dot", ext, dotfile)

    def plot_traces(self, plot_fname: str, warmup: bool = False):
        """
        Plot traces for all chains.

        Inputs:
            plot_fname :: string
                Plot filename
            warmup :: boolean
                If True, plot warmup samples instead

        Returns: Nothing
        """
        posterior = self.trace.warmup_posterior if warmup else self.trace.posterior
        with az.rc_context(rc={"plot.max_subplots": None}):
            var_names = [rv.name for rv in self.model.free_RVs]
            axes = az.plot_trace(
                posterior.sel(chain=self.good_chains()),
                var_names=var_names,
            )
            fig = axes.ravel()[0].figure
            fig.tight_layout()
            fig.savefig(plot_fname, bbox_inches="tight")
            plt.close(fig)

    def plot_pair(self, plot_fname: str, solution: int = None):
        """
        Generate pair plots from clustered posterior samples.

        Inputs:
            plot_fname :: string
                Figure filename with the format: {basename}.{ext}
                Several plots are generated:
                {basename}.{ext}
                    Pair plot of non-clustered cloud parameters
                {basename}_determ.{ext}
                    Pair plot of non-clustered cloud deterministic parameters
                {basename}_{cloud}.{ext}
                    Pair plot of clustered cloud with index {cloud} parameters
                {basename}_{num}_determ.{ext}
                    Pair plot of clustered cloud with index {cloud} deterministic parameters
                {basename}_other.{ext}
                    Pair plot of baseline and hyper parameters
            solution :: None or integer
                Plot the posterior samples associated with this solution index. If
                solution is None and there is a unique solution, use that.
                Otherwise, raise an exception.

        Returns: Nothing
        """
        if solution is None:
            solution = self._get_unique_solution
        trace = self.trace[f"solution_{solution}"]

        basename, ext = os.path.splitext(plot_fname)

        # All cloud free parameters
        plots.plot_pair(
            trace,
            self.cloud_params,
            "All Clouds\nFree Parameters",
            plot_fname,
        )
        # All cloud deterministic parameters
        plots.plot_pair(
            trace,
            self.deterministics,
            "All Clouds\nDerived Quantities",
            basename + "_determ" + ext,
        )
        # Baseline & hyper parameters
        plots.plot_pair(
            trace,
            self.baseline_params + self.hyper_params,
            "All Clouds\nDerived Quantities",
            basename + "_other" + ext,
        )
        # Cloud quantities
        for cloud in range(self.n_clouds):
            plots.plot_pair(
                trace.sel(cloud=cloud),
                self.cloud_params,
                f"Cloud {cloud}\nFree Parameters",
                basename + f"_{cloud}" + ext,
            )
            plots.plot_pair(
                trace.sel(cloud=cloud),
                self.deterministics,
                f"Cloud {cloud}\nDerived Quantities",
                basename + f"_{cloud}_determ" + ext,
            )
