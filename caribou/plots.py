"""
plots.py - Plotting helper utilities.

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

import warnings

import arviz as az
import arviz.labels as azl

import matplotlib.pyplot as plt
import numpy as np

from caribou.utils import radiative_transfer


def plot_predictive(
    data: dict,
    predictive: az.InferenceData,
    plot_fname: str,
    posterior: az.InferenceData = None,
):
    """
    Generate plots of predictive checks.

    Inputs:
        data :: dictionary
            Model data dictionary
        predictive :: az.InferenceData
            Predictive samples
        plot_fname :: string
            Plot filename
        posterior :: az.InferenceData
            If not None, plot individual posterior predictions
            for individual clouds

    Returns: Nothing
    """
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    num_chains = len(predictive.chain)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_chains)))

    for chain in predictive.chain:
        c = next(color)
        if posterior is not None:
            for draw in posterior.draw:
                # baseline
                emission_coeffs = (
                    posterior["emission_coeffs"].sel(chain=chain, draw=draw).data
                )
                emission_baseline_norm = np.sum(
                    [
                        emission_coeff * data["emission_velocity_norm"] ** i
                        for i, emission_coeff in enumerate(emission_coeffs)
                    ],
                    axis=0,
                )
                absorption_coeffs = (
                    posterior["absorption_coeffs"].sel(chain=chain, draw=draw).data
                )
                absorption_baseline_norm = np.sum(
                    [
                        absorption_coeff * data["absorption_velocity_norm"] ** i
                        for i, absorption_coeff in enumerate(absorption_coeffs)
                    ],
                    axis=0,
                )

                # radiative transfer
                peak_tau = (
                    10.0 ** posterior["log10_peak_tau"].sel(chain=chain, draw=draw).data
                )
                spin_temp = (
                    10.0
                    ** posterior["log10_spin_temp"].sel(chain=chain, draw=draw).data
                )
                velocity = posterior["velocity"].sel(chain=chain, draw=draw).data
                fwhm = 10.0 ** posterior["log10_fwhm"].sel(chain=chain, draw=draw).data
                _, _, cloud_emission, cloud_absorption = radiative_transfer(
                    data["emission_velocity"],
                    data["absorption_velocity"],
                    peak_tau,
                    spin_temp,
                    velocity,
                    fwhm,
                )

                # un-normalize
                emission_baseline = emission_baseline_norm * data["emission_noise"]
                cloud_emission_mu = cloud_emission.eval() + emission_baseline[:, None]
                absorption_baseline = (
                    absorption_baseline_norm * data["absorption_noise"]
                )
                cloud_absorption_mu = (
                    cloud_absorption.eval() + absorption_baseline[:, None]
                )

                ax1.plot(
                    data["emission_velocity"],
                    cloud_emission_mu,
                    linestyle="-",
                    color=c,
                    alpha=0.1,
                    linewidth=1,
                )
                ax2.plot(
                    data["absorption_velocity"],
                    cloud_absorption_mu,
                    linestyle="-",
                    color=c,
                    alpha=0.1,
                    linewidth=1,
                )

        # plot predictives
        outcomes = predictive["emission"].sel(chain=chain).data
        ax1.plot(
            data["emission_velocity"],
            outcomes.T,
            linestyle="-",
            color=c,
            alpha=0.5,
            linewidth=2,
        )
        outcomes = predictive["absorption"].sel(chain=chain).data
        ax2.plot(
            data["absorption_velocity"],
            outcomes.T,
            linestyle="-",
            color=c,
            alpha=0.5,
            linewidth=2,
        )

    # plot data
    ax1.plot(
        data["emission_velocity"],
        data["emission_spectrum"],
        "k-",
    )
    ax2.plot(
        data["absorption_velocity"],
        data["absorption_spectrum"],
        "k-",
    )
    ax1.set_ylabel("Emission")
    ax2.set_ylabel("Absorption")
    ax2.set_xlabel("Velocity")
    fig.savefig(plot_fname, bbox_inches="tight")
    plt.close(fig)


def plot_pair(trace, var_names, label, fname, labeller: azl.MapLabeller = None):
    """
    Pair plot helper.

    Inputs:
        trace :: az.InferenceData
            Samples to plot
        var_names :: list of strings
            variables from trace to plot
        label :: string
            Label for plot
        fname :: string
            Save plot to this filename
        cloud :: integer or None
            If None, combine all clouds into one. Otherwise, plot only
            this cloud.
        labeller :: azl.MapLabeller or None
            If not None, use apply these labels

    Returns: Nothing
    """
    size = int(2.0 * (len(var_names) + 1))
    textsize = int(np.sqrt(size)) + 8
    fontsize = 2 * size
    with az.rc_context(rc={"plot.max_subplots": None}):
        with warnings.catch_warnings(action="ignore"):
            axes = az.plot_pair(
                trace,
                var_names=var_names,
                combine_dims={"cloud"},
                kind="kde",
                figsize=(size, size),
                labeller=labeller,
                marginals=True,
                marginal_kwargs={"color": "k"},
                textsize=textsize,
                kde_kwargs={
                    "hdi_probs": [
                        0.3,
                        0.6,
                        0.9,
                    ],  # Plot 30%, 60% and 90% HDI contours
                    "contourf_kwargs": {"cmap": "Grays"},
                    "contour_kwargs": {"colors": "k"},
                },
                backend_kwargs={"layout": "constrained"},
            )
    # drop y-label of top left marginal
    axes[0][0].set_ylabel("")
    for ax in axes.flatten():
        ax.grid(False)
    fig = axes.ravel()[0].figure
    fig.text(0.7, 0.8, label, ha="center", va="center", fontsize=fontsize)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
