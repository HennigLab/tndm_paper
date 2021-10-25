![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# TNDM - Targeted Neural Dynamical Modeling

Note: This code is no longer being updated. The official re-implementation can be found at: https://github.com/HennigLab/tndm.

The code in this repository implements the models used in the Neurips 2021 paper, "Targeted Neural Dynamical Modeling". It also houses code from the baseline model, "Latent Factor Analysis via Dynamical Systems" (borrowed from https://github.com/lfads/models/tree/master/research/lfads). Latent dynamics models have emerged as powerful tools for modeling and interpreting neural population activity. Recently, there has been a focus on incorporating simultaneously measured behaviour into these models to further disentangle sources of neural variability in their latent space. These approaches, however, are limited in their ability to capture the underlying neural dynamics (e.g. linear) and in their ability to relate the learned dynamics back to the observed behaviour (e.g. no time lag). To this end, we introduce Targeted Neural Dynamical Modeling (TNDM), a nonlinear state-space model that jointly models the neural activity and external behavioural variables. TNDM decomposes neural dynamics into behaviourally relevant and behaviourally irrelevant dynamics; the relevant dynamics are used to reconstruct the behaviour through a flexible linear decoder and both sets of dynamics are used to reconstruct the neural activity through a linear decoder with no time lag. We implement TNDM as a sequential variational autoencoder and validate it on recordings taken from the premotor and motor cortex of a monkey performing a center-out reaching task. We show that TNDM is able to learn low-dimensional latent dynamics that are highly predictive of behaviour without sacrificing its fit to the neural data.


## Prerequisites

The code is written in Python 2.7.6. The other prerequisites are:

* **TensorFlow** version 1.5 ([install](https://www.tensorflow.org/install/)) -
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them)
* **h5py** ([install](https://pypi.python.org/pypi/h5py))


## Getting started

Before starting, run the following:

<pre>
$ export PYTHONPATH=$PYTHONPATH:/<b>path/to/your/directory</b>/tndm_paper/
</pre>

where "path/to/your/directory" is replaced with the path to the tndm_paper repository (you can get this path by using the `pwd` command). This allows the nested directories to access modules from their parent directory.

## Train an TNDM model

For a full list of flags, their descriptions, and their default values, refer to the top of `run_tndm_double.py`. We trained all of our models using the `run_tndm_double.sh` bash script which allows for modifying important values.

Finally, you can view the results in the ```tndm_eval_matt_data-M1.ipynb``` file.
