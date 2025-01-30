"""Defines locations where project data are stored."""
from pathlib import Path
import sys
import os

from pkg_resources import resource_filename

import src.records as records

# An example IRB number
IRB_NUMBER = '2021C000001'


# Standard location for share mounting
# This will vary based on the platform
if sys.platform == 'darwin':
    # 'darwin' just means MacOS

    # The actual project directory will look something like this
    # project_dir = Path(f'/Volumes/{IRB_NUMBER}')

    # For testing purposes, we just use the /tmp directory
    project_dir = Path('/tmp')
else:
    # Linux servers and containers

    # The actual project directory will look something like this
    # project_dir = Path.home().joinpath(f'mnt/{IRB_NUMBER}/')

    # For testing purposes, we just use the /tmp directory
    if os.environ.get('IN_DOCKER_CONTAINER') == 'True':
        project_dir = Path('/mnt/tmp')
    else:
        project_dir = Path('/tmp')


# Issue an error here if the project share doesn't exist
if not project_dir.exists() and not records.is_slurm_job():
    raise FileNotFoundError(
        f'The project share is not mounted at the expected location {project_dir}'
    )

# Location to store data
kits21_data_dir = '/data/kits/'
visceral_data_dir = '/data/visceral/'

# Location to store checkpoints
checkpoints_dir = project_dir / 'checkpoints'

# Location to store inference outputs
inferences_dir = project_dir / 'inferences'

# Location where train configs are kept
train_configs_dir = Path('configs/train')

# Locations where inference configs are kept
infer_configs_dir = Path('configs/inference')