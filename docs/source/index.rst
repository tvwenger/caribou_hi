caribou_hi
==========

``caribou_hi`` is a Bayesian model of the diffuse neutral interstellar medium. Written in the ``bayes_spec`` framework, ``caribou`` implements models to predict 21-cm observations of neutral hydrogen in emission, absorption, or both. The ``bayes_spec`` framework provides methods to fit these models to data using Monte Carlo Markov Chain techniques.

Useful information can be found in the `caribou_hi Github repository <https://github.com/tvwenger/caribou>`_, the `bayes_spec Github repository <https://github.com/tvwenger/bayes_spec>`_, and in the tutorials below.

============
Installation
============
.. code-block::

    conda create --name caribou_hi -c conda-forge pytensor pymc pip
    conda activate caribou_hi
    # Due to a bug in arviz, this fork is temporarily necessary
    # See: https://github.com/arviz-devs/arviz/issues/2437
    pip install git+https://github.com/tvwenger/arviz.git@plot_pair_reference_labels
    pip install caribou_hi

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   notebooks/absorption_model
   notebooks/emission_model
   notebooks/emission_absorption_model
   notebooks/optimization

.. toctree::
   :maxdepth: 2
   :caption: API:

   modules
