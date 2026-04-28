#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main Executable for MLflow Project Pipeline"""

import os
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

if TYPE_CHECKING:
    import mlflow.projects

@dataclass
class PipelineStep:
    name:      str
    """Command-Line Name of the Pipeline Step"""
    directory: str
    """Directory of the Pipeline Step"""
    cfgs:    list[str] = field(repr=False, default_factory=list)
    """Key(s) in Hydra Configuration"""
    args:      dict[str, str] = field(repr=False, default_factory=dict)
    """Additional Arguments"""
    skip:      bool = field(repr=False, default=False)
    """Skip Step by default in `all` mode"""
    component: bool = field(repr=False, default=False)
    """Component or Source Directory"""

    def _get_mlflow_parameters(self, config: DictConfig) -> dict[str, str]:
        """Get MLflow arguments for Pipeline Step
        
        Args:
            config: Hydra Configuration
        
        Returns:
            Dictionary of arguments to pass to MLflow.
        """
        def _make_param(key: str) -> str:
            """Convert Hydra configuration key to MLflow parameter name
            
            Args:
                key: Key in dot notation (e.g. `etl.sample`)
            Returns:
                MLflow parameter name (e.g. `sample`).
            """
            return key.rsplit('.', maxsplit=1)[-1]

        def _get_value(key: str) -> str | DictConfig:
            """Extract value from Hydra configuration using dot notation key

            Args:
                key: Key in dot notation (e.g. `etl.sample`)
            Returns:
                Value from Hydra configuration (e.g. `1000`).
            """
            value = config
            for key in key.split('.'):
                value = value[key]
            return value
        
        def _extract_json_key(key: str, arg: str, outdir: Path) -> str:
            """Extract value from Hydra configuration and write to JSON file for MLflow
            
            Args:
                key: Key in dot notation (e.g. `modeling.random-forest`)
                arg: Argument name for the JSON file (e.g. `rt-config`)
                outdir: Output directory for the JSON file
            Returns:
                Absolute path to JSON file.
            """
            value = _get_value(key)
            filename = (outdir / f'{arg}.json').absolute()
            with open(filename, 'w') as f:
                json.dump(dict(value.items()), f)
            return str(filename)

        def _esc_sp(value):
            """MLflow on Windows invokes cmd /k with mangled arguments"""
            if (type(value) is str and ' ' in value):
                return value.replace(' ', '_')
            return value

        # grab hydra output directory
        outdir = Path(HydraConfig.get().runtime.output_dir) / '.json-inputs'
        outdir.mkdir(parents=True, exist_ok=True)
        # collect configuration arguments
        cfg_args = tuple((*cfg.rsplit(':', maxsplit=1), None)[:2] for cfg in self.cfgs)
        # regular config arguments
        args = {_make_param(key): _esc_sp(_get_value(key)) for key, json_arg in cfg_args if json_arg is None}
        # json config arguments
        args.update({_make_param(json_arg): _extract_json_key(key, json_arg, outdir) for key, json_arg in cfg_args if json_arg is not None})
        # additional arguments
        args.update({k: _esc_sp(v) for k, v in self.args.items()})
        return args

    def run(self, config: DictConfig) -> 'mlflow.projects.SubmittedRun':
        """Run the Pipeline Step as an MLflow Project
        
        Args:
            config: Hydra Configuration
        
        Returns:
            SubmittedRun exposing information (e.g. run ID) about the launched run.
        """
        prefix = Path(__file__).parent / (config['main']['components'] if (self.component) else 'src')
        # pass +main.environment=local to MLflow to re-use local environment for each step
        env_manager = config['main'].get('environment', 'conda')

        # execute MLflow
        mlflow.set_tracking_uri
        return mlflow.run(
            uri=f'{prefix}/{self.directory}',
            entry_point='main',
            env_manager=env_manager,
            parameters=self._get_mlflow_parameters(config),
        )

PIPELINE_STEPS = [
    PipelineStep('eda', 'eda', skip=True),
    PipelineStep('download', 'get-data', component=True,
                 cfgs=[
                   'etl.sample'
                   ],
                args={
                    "artifact-name": "sample.csv",
                    "artifact-type": "raw_data",
                    "artifact-description": "Raw file as downloaded"
                }
    ),
    PipelineStep('basic_cleaning', 'clean-data',
                 cfgs=[
                     'etl.min-price',
                     'etl.max-price',
                     ],
                 args={
                     'input-artifact': 'sample.csv:latest',
                     'output-artifact': 'cleaned_sample.csv',
                     'output-type': 'clean_sample',
                     'output-description': 'Data with outliers and null values removed',
                     },
    ),
    PipelineStep('data_check', 'check-data',
                 cfgs=[
                     'etl.min-price',
                     'etl.max-price',
                     'data-check.kl-threshold'
                     ],
                 args={
                     'csv': 'cleaned_sample.csv:latest',
                     'ref': 'cleaned_sample.csv:reference',
                     },
    ),
    PipelineStep('data_split', 'train-val-test-split', component=True,
                 cfgs=[
                     'modeling.test-size',
                     'modeling.random-seed',
                     'modeling.stratify-by',
                     ],
                 args={
                     'input': 'cleaned_sample.csv:latest',
                      },
    ),
    PipelineStep('train_random_forest', 'train-random-forest',
                 cfgs=[
                     # special syntax to write config to JSON
                     'modeling.random-forest:rf-config',
                     'modeling.val-size',
                     'modeling.random-seed',
                     'modeling.stratify-by',
                     'modeling.max-tfidf-features',
                     ],
                 args={
                     'trainval-artifact': 'trainval_data.csv:latest',
                     'output-artifact': 'random_forest_export',
                     }
    ),
    PipelineStep('test_regression_model', 'test-regression-model', component=True, skip=True,
                 args={
                     'mlflow-model': 'random_forest_export:prod',
                     'test-dataset': 'test_data.csv:latest',
                     }
    ),
]
"""List of Pipeline Steps in order"""

# convert list of steps to dictionary, order is preserved
PIPELINE_STEPS_MAP = {step.name: step for step in PIPELINE_STEPS}
"""Map of Pipeline Steps by command-line name"""


# This automatically reads in the configuration
@hydra.main(config_name='config', config_path='conf', version_base=None)
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["group"]

    # Steps to execute
    steps_par = config['main']['steps']
    # explicit steps are not skipped
    steps_expl = steps_par != "all"
    active_steps = steps_par.split(",") if steps_expl else list(PIPELINE_STEPS_MAP.keys())

    # make sure all steps are actually defined
    missing_steps = tuple(step for step in active_steps if step not in PIPELINE_STEPS_MAP)
    if missing_steps:
        raise RuntimeError(f'Step(s) {", ".join(missing_steps)} are not defined! Check configuration or command-line arguments.')

    # run pipeline steps
    for step in (PIPELINE_STEPS_MAP[s] for s in active_steps):
        if (not steps_expl and step.skip):
            # skip 'all' steps that are marked as skip by default
            continue
        # run explicitly requested or unskipped steps
        step.run(config)

if __name__ == "__main__":
    go()
