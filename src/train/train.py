"""Main training functions."""
import json
import datetime

from src import locations
from src.records import get_job_record, save_job_record, is_slurm_job
from src.train.train_seg import train_segmentation

import click


@click.command()
@click.option('-config_file', required=True)
@click.option('-description', required=False, default='')
def train(config_file: str, description: str) -> None:
    """Train a model using a config file.

    This will create a new directory in the checkpoints directory containing
    model weights.

    Parameters
    ----------
    config_file: str
        Name of the config file within the repo's training config directory
        containing all parameters for the training job.
    description: str
        Description of experiment. Saved to train_record for reference.
    """
    # Read in the config file
    if not config_file.lower().endswith('.json'):
        config_file += '.json'
    config_path = locations.train_configs_dir / config_file
    with config_path.open('r') as jf:
        config = json.load(jf)

    # Get a record of this job's parameters
    seed = config['random_seed'] if 'random_seed' in config else None
    job_record = get_job_record(seed)
    job_record['DESCRIPTION'] = description

    # Determine the output directory for the trained model and associated files
    if is_slurm_job():
        # Create the training job's output directory from the config file name and
        # the job ID, and place in the scratchdir
        output_dir = (
            locations.checkpoints_dir / f'{config_path.stem}_ID{job_record["SLURM_JOB_ID"]}'
        )
    else:
        date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = (
            locations.checkpoints_dir / f'{config_path.stem}_{date_str}'
        )
    output_dir.mkdir(parents=True)

    # Stash the job record and a copy of the config file in the output directory
    save_job_record(output_dir, record=job_record, name='train_record.json')
    config_copy_path = output_dir / 'config.json'
    with config_copy_path.open('w') as jf:
        json.dump(config, jf, indent=4)
    # Now for actual training code...
    task = config['task']
    print(f"----training task: {task}----")
    if task == "segmentation":
        _, epoch_loss_values, val_metric_values = train_segmentation(config, output_dir)
    else:
        raise NotImplementedError(f"{task} task not defined!")

    # save training outputs to checkpoints_dir
    training_outputs_record = {
        'epoch_loss_values': epoch_loss_values,
        'val_metric_values': val_metric_values
    }
    save_job_record(output_dir, "training_outputs.json", training_outputs_record)