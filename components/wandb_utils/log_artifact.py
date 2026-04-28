import wandb


def log_artifact(artifact_name: str, artifact_type: str, artifact_description: str, filename: str, run: wandb.Run):
    """Log file artifact in Weights & Biases.

    Associate a local file as a single-file artifact in the current Weights & Biases run.
    Execution will wait for the artifact to be uploaded.

    Args:
        artifact_name: Name for the artifact.
        artifact_type: Type for the artifact (for example, "raw-data" or "clean-data").
        artifact_description: Brief description of the artifact.
        filename: Local filename to attach to the artifact.
        run: Current Weights & Biases run.
    """
    # Log to W&B
    artifact = wandb.Artifact(
        artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()
