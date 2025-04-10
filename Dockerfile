FROM continuumio/miniconda3:latest

RUN conda install conda --yes
RUN conda install -c conda-forge pymc cxx-compiler pip --yes
RUN pip cache purge
RUN pip install git+https://github.com/tvwenger/arviz.git@plot_pair_reference_labels
RUN pip install --no-cache-dir caribou_hi
