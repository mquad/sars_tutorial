
# RecSys 18 Tutorial on Sequence-Aware Recommenders

This is the repository for the hands-on session of the Tutorial on Sequence-Aware Recommenders to be held at ACM RecSys 2018 in Vancouver.

You have two options to run the code contained in this repository:
1. Setup a new environment on your local machine and run the code locally (_highly recommended_).
2. Launch a new Binder instance by clicking on this badge [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/mquad/sars_tutorial/master). 

While we all know that setting up a new local environment is a slightly tedious process, Binder instances have strict resource limits (1-2GB of memory, max 100 concurrent users per repository).
So we *highly recommend* to set up a new local environment in advance by following the [Setup instructions](#setup-instructions).

## Setup instructions

First of all, clone this project to your local machine.

Now you need to set up a new python3 environment. We will use Anaconda/Miniconda for doing so.
If you don't have Anaconda/Minicoda already installed on your machine, download it [here](https://conda.io/miniconda.html) (**Python 3 version**).

After that, install the environment for this hands-on by running:
```bash
conda env create --file environment.yml
```

Then activate the environment with `source activate srs` or `conda activate srs`, and install a new `iptyhon` kernel by running:

```bash
python -m ipykernel install --name srs
``` 

Finally, launch the Jupyter Notebook with
```bash
jupyter notebook --port=8888
```

and open it your browser at the address `localhost:8888`. 
(Beware, if port `8888` is already taken by another service, jupyter notebook will automatically open on a different one. Check out the startup log!).

## Running the notebooks

The notebooks used in this hands-on are listed in the main directory of this project, as shown below:

<img src="images/running_notebooks_1.png" width="300" >

Click on the name of the notebook to open it in a new window. The name of each running notebook is highlighted in green 
(in the screen above, the notebook `00_TopPopular` is the only one running).

Before starting to execute the notebook cells, you have to ensure that the kernel is properly set to `srs`, like in the screen below:

![](images/running_notebooks_2.png)

If it's not your case, change the kernel to `srs` by clicking on `Kernel > Change kernel > srs` in the menu bar, as shown below:

![](images/running_notebooks_3.png)

NOTE: this requires the installation of the `srs` kernel, as explained in the [Setup instructions](#setup-instructions).

You can now start running the cells in the notebook! Yay!


# Acknowledgments

We want to sincerely thank [Umberto Di Fabrizio](https://www.linkedin.com/in/umbertodifabrizio) for developing large sections of this repository back when he was a MSc student at Politecnico di Milano. Good luck Umberto!