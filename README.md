# Lacan paper experiments

These files are provided to allow rerunning the experiments presented in the paper "LACAN: Leveraging Adjacent Co-occurrence of Atomic Neighborhoods for Molecular Scoring and Generation", Dehaen, W. (2026). Most of the experiments are provided as notebooks, in case of the GuacaMol benchmarks, the tasks are ran as python scripts.

## Installing the proper environments

Both MolSkill and GuacaMol require some effort to install. Thus, the recommendation is to make a specific environment to run the Guacamol experiments, another specific environment to run the GuacaMol experiments. The other needed packages will pip install in a clean environment without too much troubleshooting needed. Using a Python 3.10 environment:

```bash
conda create -n lacan-experiments -c conda-forge "python=3.10"
conda activate lacan-experiments
```

When running the MolSkill experiment, install MolSkill as per instructions on https://github.com/microsoft/molskill

When running the GuacaMol experiment, install GuacaMol as per instructions on https://github.com/BenevolentAI/guacamol

Then, install the remaining needed packages with:

```bash
pip install jupyterlab rdkit lacan numpy pandas matplotlib seaborn scipy dockstring chembl_downloader scikit-learn
```

## Data

Some of the data is too large to be included. This includes the entirety of ChEMBL35. This version was created according to the following instructions: https://github.com/dehaenw/lacan/issues/1#issuecomment-2571403057. The resulting files should be placed in the data folder. A notebook has been included for generating the ChEMBL36 time split data. For COCONUT data, the SDF can be downloaded from https://coconut.naturalproducts.net/download and conversion to a SMILES file in the right format can be done using the provided cell in the `profiles_compare.ipynb` notebook.
