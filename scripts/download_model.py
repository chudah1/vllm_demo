"""
Helper script to download models from HuggingFace.
Pre-downloads models to avoid delays during server startup.
"""

import os
import sys
import click
from huggingface_hub import snapshot_download
from pathlib import Path


# Popular open-source models
POPULAR_MODELS = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
}


def download_model_impl(model_name: str, cache_dir: str = None, token: str = None):
    """
    Download a model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name or short name from POPULAR_MODELS
        cache_dir: Directory to cache the model (defaults to ~/.cache/huggingface)
        token: HuggingFace API token (required for gated models like Llama)
    """
    # Check if it's a short name
    if model_name in POPULAR_MODELS:
        full_model_name = POPULAR_MODELS[model_name]
        click.echo(f"Resolved '{model_name}' to '{full_model_name}'")
    else:
        full_model_name = model_name

    # Use default cache directory if not specified
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")

    click.echo(f"\nDownloading model: {full_model_name}")
    click.echo(f"Cache directory: {cache_dir}")

    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token:
        click.echo("Using HuggingFace token for authentication")
    else:
        click.echo("No HuggingFace token provided (required for gated models)")

    try:
        # Download the model
        click.echo("\nDownloading... This may take several minutes depending on model size.")

        local_path = snapshot_download(
            repo_id=full_model_name,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
        )

        click.secho(f"\n✓ Model downloaded successfully!", fg="green", bold=True)
        click.echo(f"  Location: {local_path}")
        click.echo(f"\nYou can now use this model by setting:")
        click.secho(f"  MODEL_NAME={full_model_name}", fg="cyan")

    except Exception as e:
        click.secho(f"\n✗ Error downloading model: {e}", fg="red", bold=True)
        if "gated" in str(e).lower() or "access" in str(e).lower():
            click.echo("\nThis model may require:")
            click.echo("  1. Accepting the license agreement on HuggingFace")
            click.echo("  2. Providing a valid HuggingFace token with --token")
        sys.exit(1)


def list_popular_models():
    """Display list of popular models with short names."""
    click.echo("\nPopular Models (use short name for convenience):")
    click.echo("=" * 70)
    for short_name, full_name in POPULAR_MODELS.items():
        click.secho(f"  {short_name:<20}", fg="yellow", nl=False)
        click.echo(f" -> {full_name}")
    click.echo("=" * 70)
    click.echo("\nUsage: python scripts/download_model.py <model-name>")
    click.echo("   or: python scripts/download_model.py meta-llama/Llama-3.2-3B-Instruct")


@click.command()
@click.argument('model', required=False)
@click.option(
    '--list',
    is_flag=True,
    help='List popular models with short names'
)
@click.option(
    '--cache-dir',
    type=click.Path(),
    default=None,
    help='Cache directory for downloaded models'
)
@click.option(
    '--token',
    type=str,
    default=None,
    envvar='HUGGING_FACE_HUB_TOKEN',
    help='HuggingFace API token (or set HUGGING_FACE_HUB_TOKEN env var)'
)
def main(model, list, cache_dir, token):
    """
    Download models from HuggingFace Hub for vLLM.

    MODEL: Model name or short name (use --list to see available short names)
    """
    if list or model is None:
        list_popular_models()
        if model is None:
            sys.exit(0)

    download_model_impl(model, cache_dir, token)


if __name__ == "__main__":
    main()
