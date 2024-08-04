from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import wandb
import streamlit as st

from utilities import ContextUnet, setup_ddpm
from constants import WANDB_API_KEY


def load_model():
    """
    This function loads the model from the model registry.
    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # login to wandb
    # wandb.login(key=WANDB_API_KEY)

    api = wandb.Api(api_key=WANDB_API_KEY)
    artifact = api.artifact("teamaditya/model-registry/Feature2Sprite:v1", type="model")
    model_path = Path(artifact.download())

    # recover model info from the registry
    producer_run = artifact.logged_by()

    # load the weights dictionary
    model_weights = torch.load(model_path / "context_model.pth", map_location="cpu")

    # create the model
    model = ContextUnet(
        in_channels=3,
        n_feat=producer_run.config["n_feat"],
        n_cfeat=producer_run.config["n_cfeat"],
        height=producer_run.config["height"],
    )

    # load the weights into the model
    model.load_state_dict(model_weights)

    # set the model to eval mode
    model.eval()
    return model.to(DEVICE)


def show_image(img):
    """
    This function shows the image in the streamlit app.
    """
    img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
    st.image(img, clamp=True)
    return img


def generate_sprites(feature_vector):
    """
    This function generates sprites from a given feature vector.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    config = SimpleNamespace(
        # hyperparameters
        num_samples=30,
        # ddpm sampler hyperparameters
        timesteps=500,
        beta1=1e-4,
        beta2=0.02,
        # network hyperparameters
        height=16,
    )
    nn_model = load_model()

    _, sample_ddpm_context = setup_ddpm(
        config.beta1, config.beta2, config.timesteps, DEVICE
    )

    noises = torch.randn(config.num_samples, 3, config.height, config.height).to(DEVICE)

    feature_vector = torch.tensor([feature_vector]).to(DEVICE).float()
    ddpm_samples, _ = sample_ddpm_context(nn_model, noises, feature_vector)

    # upscale the 16*16 images to 256*256
    ddpm_samples = F.interpolate(ddpm_samples, size=(256, 256), mode="bilinear")
    # show the images
    img = show_image(ddpm_samples[0])
    return img
