import datetime
import json
from pathlib import Path
from src import locations
import click
from src.records import get_job_record, get_latest_checkpoint, save_job_record
from src.inference.infer_seg import infer_segmentation


@click.command()
@click.option('-config_file', required=True)
@click.option('-training_checkpoint', required=False, default='')
@click.option('-description', required=False, default='')
def infer(config_file: str, training_checkpoint: str, description: str) -> None:
    """Performs inference over a trained model.

    Loads model weights from $checkpoint and performs inference based on parameters
    of $config_file.

    Parameters
    ----------
    config_file: str
        Name of the config file within the repo's inference config directory
        containing all parameters for the inference job.
    training_checkpoint: str
        Path to training checkpoint that contains model weights, training config file, etc.
        Defaults to latest training checkpoint if nothing is passed.
    description: str
        Description of experiment. Saved to infer_record for reference.
    """
    # load inference config file
    if not config_file.lower().endswith('.json'):
        config_file += '.json'
    infer_config_path = locations.infer_configs_dir / config_file
    with infer_config_path.open('r') as jf:
        infer_config = json.load(jf)

    if training_checkpoint == '':
        training_checkpoint_path = get_latest_checkpoint()
    else:
        training_checkpoint_path = Path(training_checkpoint)

    # load training config file
    training_config_path = training_checkpoint_path / "config.json"
    with training_config_path.open('r') as jf:
        training_config = json.load(jf)

    # make output directory for inference outputs
    date_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = (
        locations.inferences_dir / f'{training_checkpoint_path.stem}--{infer_config_path.stem}_{date_str}'
    )
    output_dir.mkdir(parents=True)

    # save inference job record
    seed = infer_config['random_seed'] if 'random_seed' in infer_config else None
    job_record = get_job_record(seed)
    job_record['DESCRIPTION'] = description
    save_job_record(output_dir, "infer_record.json", job_record)
    # save copy of inference config to output dir
    save_job_record(output_dir, "config.json", infer_config)

    # execute inference
    infer_task = infer_config['task']
    training_task = training_config['task']
    assert infer_task == training_config["task"], \
        f"Error: cannot perform {infer_task} inference on {training_task} model"

    print(f"----inference task: {infer_task}----")
    if infer_task == "segmentation":
        infer_segmentation(infer_config, training_config, training_checkpoint_path, output_dir)
    else:
        raise NotImplementedError(f"{infer_task} task not defined!")