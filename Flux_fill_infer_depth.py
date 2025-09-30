import torch
from diffusers.pipelines.flux.pipeline_flux_fill_unmasked_image_condition_version import FluxFillPipeline_token12_depth_only as FluxFillPipeline
from diffusers.utils import load_image
import os, glob
import numpy as np
import cv2
from PIL import Image, ImageOps

image_path = ["example_data/I-210618_I01001_W01_I-210618_I01001_W01_F0153_img.jpg"]
pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("stage1/checkpoint-4800")
for image_ep in image_path:
    mask_path = image_ep.replace("_img.jpg","_mask.png")
    image = Image.open(image_ep)  # place_hold
    depth = Image.open(image_ep.replace("_img.jpg", 
                                                "_depth_img.png"))
    image_name = os.path.basename(image_ep)
    mask = Image.open(mask_path).convert("L")
    mask = ImageOps.invert(mask)    # inverse rord_mask

    # mask_np = np.array(mask) 

    # # mask dilation
    # dilation_px = 32
    # kernel = np.ones((3, 3), np.uint8)
    # iterations = dilation_px // 2  
    # dilated_mask = cv2.dilate(mask_np, kernel, iterations=iterations)
    # mask = Image.fromarray(dilated_mask)

    orig_w, orig_h = image.size

    # Resize to 1024 × 1024
    # target_size = (1024, 1024)
    # image_resized = image.resize(target_size, Image.BICUBIC)
    # mask_resized = mask.resize(target_size, Image.NEAREST)
    # depth_resized = depth.resize(target_size, Image.BICUBIC)

    w, h = image.size
    MAX_SIZE = 1024
    if max(w, h) > MAX_SIZE:
            factor = MAX_SIZE / max(w, h)
            w = int(factor * w)
            h = int(factor * h)
    width, height = map(lambda x: x - x % 64, (w, h))
    image_out = pipe(
        prompt="A beautiful scene",
        image=image,
        mask_image=mask,
        width=width,
        height=height,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        depth=depth
    ).images[0]


    image_final = image_out.resize((orig_w, orig_h), Image.BICUBIC)

    output_dir = "./depth_fillin_results"
    os.makedirs(output_dir, exist_ok=True)
    image_final.save(os.path.join(output_dir, image_name))

