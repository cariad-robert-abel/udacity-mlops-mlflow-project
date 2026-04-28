# Building a Reproducible Model Workflow Project

This repository contains the project associated with "Building a Reproducible Model Workflow"
Udacity course. It's a fork of Udacity's [Starter Kit](https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices).

This GitHub.com project is located at [github.com/cariad-robert-abel/udacity-mlops-mlflow-project](https://github.com/cariad-robert-abel/udacity-mlops-mlflow-project).
The associated Weights & Biases project is located at [wandb.ai/cariad-robert-abel-cariad-se/nyc_airbnb](https://wandb.ai/cariad-robert-abel-cariad-se/nyc_airbnb).

## Goal

The goal of this project is to build a reproducible machine learning pipeline that predicts short-term rental prices in
New York City.
The model needs to estimate the typical price for a given property based on the price of similar properties trained off
of AirBnB data.
Training, testing, and validation are performed on bulk data. Our example includes one update to show-case the ability
to catch data quality errors early on.

## Setup

This project builds a reproductible machine learning pipeline using [MLflow](https://mlflow.org/), which uses [Python](https://python.org)
environment managed through [conda](https://conda-forge.org/) environments.

### Python Requirement

This project requires **Python 3.13**. Please ensure that you have Python 3.13 installed and set as the default version
in your environment to avoid any runtime issues.

### Create Environment
Make sure to have conda installed and ready, then create a new environment using the ``environment.yml``
file provided in the root of the repository and activate it:

```bash
> conda env create -f environment.yml
> conda activate nyc-airbnb-dev
```

### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

## Usage

Use [MLflow](https://mlflow.org/) to run the pipeline from the root of the repository:

    mlflow run ./exercise_N [-P steps=download -P hydra-options=main.project=<project-name>]

You may also pass a comma-separated list of steps (without whitespace) to the `steps` parameter in
order to only run individual steps or `all` to run all steps (default).
Configuration options may be overwritten at run time using the `hydra-options` parameter, see
[Hydra](https://hydra.cc/).

## License

Original files Copyright 2012–2022 Udacity, Inc.
My additions to documentation and code are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [LICENSE-Udacity](LICENSE-Udacity) resp. [LICENSE](LICENSE).
