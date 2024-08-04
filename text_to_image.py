""" 
    This file contains the code for the text to image generation using the Stable Diffusion model.
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from constants import MODEL_ID


def generate_image(prompt: str) -> torch.Tensor:
    """
    Generates an image from a given prompt.

    Args:
        prompt (str): The prompt to generate the image from.

    Returns:
        torch.Tensor: The generated image.
    """
    # load model

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID
    )  # torch_dtype=torch.float16
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")  # move model to GPU if available
        print("Model loaded successfully and moved to GPU.")
    else:
        print("Model loaded successfully on CPU.")

    image = pipe(prompt).images[0]

    return image
