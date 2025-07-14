import argparse
import logging
import os
import random
from pathlib import Path
from time import perf_counter

import dotenv
import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from PIL.Image import Image
from transformers import T5EncoderModel

from prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

dotenv.load_dotenv()
token = os.getenv("HF_TOKEN")


def create_pipeline(model_id="stabilityai/stable-diffusion-3.5-large"):

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        token=token,
    )

    t5_nf4 = T5EncoderModel.from_pretrained(
        "diffusers/t5-nf4", torch_dtype=torch.bfloat16, token=token
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16,
    )

    # Apply optimization for SD3
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    # Enable xFormers optimization only if available
    try:
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xFormers memory efficient attention enabled.")

    except Exception as e:
        logger.warning(f"Failed to enable xFormers: {e}")
        logger.warning("Using default attention mechanism.")

    # Use model offload for memory efficiency (if needed)
    # pipeline.enable_model_cpu_offload()
    pipeline = pipeline.to("cuda")

    # Note: torch.compile is not compatible with 4-bit quantized models, so it is disabled
    # The following code is commented out because it will cause errors
    # if torch.__version__ >= "2":
    #     try:
    #         pipeline.transformer = torch.compile(
    #             pipeline.transformer, mode="reduce-overhead", fullgraph=True
    #         )
    #         logger.info("Successfully compiled transformer model")
    #     except Exception as e:
    #         logger.warning(f"Failed to compile: {e}")

    return pipeline


def generate_prompts(prompt_generator: PromptGenerator, num_prompts=4):
    """Generate multiple prompts with balanced gender."""
    prompts = []
    for _ in range(num_prompts):
        prompt = (
            "Head and shoulders portrait of a character, all facial features. "
            "Follow specific characteristics below.\n"
        )
        prompt += prompt_generator.generate()
        prompts.append(prompt)
    return prompts


def generate_batch(
    pipeline,
    generator,
    batch_size=4,
    num_total=10,
    output_dir="dataset",
    num_inference_steps=4,
    guidance_scale=10.0,
):
    """Generate images in batches."""
    output_dir = Path(output_dir)
    images_dir = output_dir.joinpath("images")
    prompts_dir = output_dir.joinpath("prompts")
    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Process in batches
    num_batches = (num_total + batch_size - 1) // batch_size  # Ceiling division

    exist_images = sorted(images_dir.glob("*.png"))
    if exist_images:
        last_index = int(exist_images[-1].stem) + 1
    else:
        last_index = 0
    # 2. Forward embeddings and negative embeddings through text encoder
    for batch_idx in range(num_batches):
        # For the last batch, generate only the remaining number
        current_batch_size = min(batch_size, num_total - batch_idx * batch_size)
        prompts = generate_prompts(
            prompt_generator=generator, num_prompts=current_batch_size
        )

        try:
            images: list[Image] = pipeline(
                prompt=prompts,
                # prompt_embeds=prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images  # list[PIL.Image.Image]
            logger.info(f"{len(images)} images generated in batch {batch_idx + 1}")
            for image, prompt in zip(images, prompts):
                global_idx = batch_idx * batch_size + images.index(image) + last_index
                image.save(images_dir.joinpath(f"{global_idx:06d}.png"))
                with open(prompts_dir.joinpath(f"{global_idx:06d}.txt"), "w") as f:
                    f.write(prompt)

        except Exception as e:
            logger.error(f"Error occurred during batch {batch_idx+1}: {e}")
            logger.warning("Falling back to single image mode...")

            # Fallback to single image generation mode
            for i in range(current_batch_size):
                try:
                    sub_start = perf_counter()
                    image = pipeline(
                        prompt=prompts[i],
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    ).images[0]
                    sub_end = perf_counter()

                    global_idx = batch_idx * batch_size + i + last_index

                    image.save(images_dir.joinpath(f"{global_idx:06d}.png"))
                    with open(prompts_dir.joinpath(f"{global_idx:06d}.txt"), "w") as f:
                        f.write(prompts[i])
                    logger.info(
                        f"  Image {i+1}/{current_batch_size}: {sub_end - sub_start:.2f}s"
                    )

                    # Clear memory after each image generation
                    torch.cuda.empty_cache()

                except Exception as sub_err:
                    logger.error(f"  Failed to generate image {i+1}: {sub_err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate character face dataset in batches"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-3.5-large",
        help="Model ID for the Stable Diffusion pipeline",
    )
    parser.add_argument(
        "--num", type=int, default=10, help="Total number of images to generate"
    )
    parser.add_argument(
        "--output", type=str, default="dataset3", help="Output directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=8,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=28,
        help="Number of inference steps for generation",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger.info("Initializing pipeline...")
    pipeline = create_pipeline(model_id=args.model_id)
    prompt_generator = PromptGenerator()
    logger.info(f"Generating {args.num} images in batches of {args.batch_size}...")
    generate_batch(
        pipeline,
        prompt_generator,
        batch_size=args.batch_size,
        num_total=args.num,
        output_dir=args.output,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
