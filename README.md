![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# PLFADS - Preferential Latent Factor Analysis via Dynamical Systems

This code implements the model from the paper, "PLFADS - Preferential Latent Factor Analysis via Dynamical Systems". PLFADS is an unsupervised method to decompose time series data into various factors, such as an initial condition, a generative dynamical system, control inputs to that generator, and a low dimensional description of the observed data, called the factors. With PLFADS, there are two types of observed data: the neural activity and the associated behaviour of interest. These observations have a noise model (Poisson or Gaussian), so a denoised version of the observations is also created (e.g. underlying rates of a Poisson distribution given the observed spike counts).


## Prerequisites

The code is written in Python 2.7.6. You will also need:

* **TensorFlow** version 1.5 ([install](https://www.tensorflow.org/install/)) -
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them)
* **h5py** ([install](https://pypi.python.org/pypi/h5py))


## Getting started

Before starting, run the following:

<pre>
$ export PYTHONPATH=$PYTHONPATH:/<b>path/to/your/directory</b>/plfads/
</pre>

where "path/to/your/directory" is replaced with the path to the PLFADS repository (you can get this path by using the `pwd` command). This allows the nested directories to access modules from their parent directory.

## Generate synthetic data

In order to generate the synthetic lorenz datasets from the top-level plfads directory, run:

```sh
$./generate_lorenz_data.sh
```

## Train an PLFADS model

Now that we have our example datasets, we can train some models! To spin up an PLFADS model on the synthetic data, run any of the following commands. For the examples that are in the paper, the important hyperparameters are roughly replicated. Most hyperparameters are insensitive to small changes or won't ever be changed unless you want a very fine level of control. In the first example, all hyperparameter flags are enumerated for easy copy-pasting, but for the rest of the examples only the most important flags (~the first 9) are specified for brevity. For a full list of flags, their descriptions, and their default values, refer to the top of `run_plfads.py`.  Please see Table 1 in the Online Methods of the associated paper for definitions of the most important hyperparameters.

```sh
# Run PLFADS on lorenz data
$ ./run_plfads.sh
```
You have to change the parameters and data paths in the ```run_plfads.sh``` file

Finally, you can view the results in the ```plfads_eval_example.ipynb``` file.
