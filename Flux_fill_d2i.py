import torch
from diffusers.pipelines.flux.pipeline_flux_fill_unmasked_image_condition_version import FluxFillPipeline_token12_depth as FluxFillPipeline
from diffusers.utils import load_image
import os, glob
import numpy as np
import cv2
from PIL import Image

image_path = ["example_data/I-210618_I01001_W01_I-210618_I01001_W01_F0153_img.jpg"]

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("stage2/checkpoint-20000")
for image_ep in image_path:
    image = Image.open(image_ep)
    mask = Image.new("L", image.size, 0)  # place_hold
    depth_path = image_ep.replace("_img.jpg", "_depth_img.png")
    depth_image = Image.open(depth_path)
    depth = Image.open(depth_path.replace("_img", "_img_fill_in"))
    image_name = os.path.basename(image_ep)

    orig_w, orig_h = image.size
    w, h = image.size
    MAX_SIZE = 1024
    if max(w, h) > MAX_SIZE:
            factor = MAX_SIZE / max(w, h)
            w = int(factor * w)
            h = int(factor * h)
    width, height = map(lambda x: x - x % 64, (w, h))
    # # Resize to 1024 × 1024
    target_size = (width, height)
    # target_size = (1024, 1024)
    # image_resized = image.resize(target_size, Image.BICUBIC)
    # mask_resized = mask.resize(target_size, Image.NEAREST)
    # depth_resized = depth.resize(target_size, Image.BICUBIC)
    # depth_image_resized = depth_image.resize(target_size, Image.BICUBIC)

    image = pipe(
        prompt="A beautiful scene",
        image=image,
        mask_image=mask,
        width=target_size[0],
        height=target_size[1],
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        depth=depth,
        depth_image=depth_image,
    ).images[0]
    image_final = image.resize((orig_w * 3, orig_h), Image.BICUBIC)
    output_dir = "./test_images/"
    os.makedirs(output_dir, exist_ok=True)
    image_final.save(os.path.join(output_dir,image_name))
