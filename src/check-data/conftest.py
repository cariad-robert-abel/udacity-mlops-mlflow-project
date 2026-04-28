#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Data Testing with pytest and W&B artifacts."""
import pytest
import warnings
# filter deprecation warnings at import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import wandb
import wandb_utils # noqa: F401 # we need the Artifact.file work-around


run : wandb.Run = wandb.init(job_type='check-data')
"""Global Run for Session"""


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--csv", type=str, required=True, help="Input CSV file to be tested")
    parser.addoption("--ref", type=str, required=True, help="Reference CSV file to compare the new csv to")
    parser.addoption("--kl-threshold", type=float, required=True, help="Threshold for the KL divergence test on the neighborhood group column")
    parser.addoption("--min-price", type=int, required=True, help="Minimum accepted price")
    parser.addoption("--max-price", type=int, required=True, help="Maximum accepted price")


def pytest_configure(config: pytest.Config):
    """Add configuration to filter deprecation warnings at run-time"""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning"
    )


def pytest_sessionfinish(
        session: pytest.Session,
        exitstatus: int):
    """Hook at the end of a Session

    Args:
        session: pytest Session object (unused)
        exitstatus: pytest exit status code
    """
    # silence pylint
    _ = session

    # actuall fail the Weight & Biases run if any test failed
    run.finish(exitstatus)


@pytest.fixture(scope='session')
def data(request: pytest.FixtureRequest) -> pd.DataFrame:
    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()
    return pd.read_csv(data_path)


@pytest.fixture(scope='session')
def ref_data(request: pytest.FixtureRequest) -> pd.DataFrame:
    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()
    return  pd.read_csv(data_path)


@pytest.fixture(scope='session')
def kl_threshold(request: pytest.FixtureRequest) -> float:
    return request.config.option.kl_threshold

@pytest.fixture(scope='session')
def min_price(request: pytest.FixtureRequest) -> int:
    return request.config.option.min_price

@pytest.fixture(scope='session')
def max_price(request: pytest.FixtureRequest) -> int:
    return request.config.option.max_price
