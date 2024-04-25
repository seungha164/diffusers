import torch
import torch.nn as nn
from diffusers.schedulers import PNDMScheduler
import onnx
import onnxruntime
from torch.onnx import export
from pathlib import Path
from packaging import version
from diffusers import DiffusionPipeline
import os
import shutil
from typing import List, Optional, Tuple, Union
import numpy as np
import inspect
from tqdm import tqdm

import gc
gc.collect()

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

device = "cpu"
dtype = torch.float32

class TextEncoder(nn.Module):
    def __init__(self, tokenizer, textencoder, device='cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.text_encoder = textencoder.to(device = device)
    
    def forward(self, text_input):
        # 1. uncond-input 준비
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device=self.device, dtype=torch.int32)
        negative_prompt_embeds  = self.text_encoder(uncond_input).last_hidden_state
        
        # 2. text-input 준비
        text_embed  = self.text_encoder(text_input.to(device=self.device, dtype=torch.int32)).last_hidden_state
        prompt_embeds = torch.cat([negative_prompt_embeds, text_embed])

        return prompt_embeds

textEncoder = TextEncoder(pipeline.tokenizer, pipeline.text_encoder, device)
text_input = pipeline.tokenizer(
    ["A sample prompt"],
    padding="max_length",
    max_length=pipeline.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
    ).input_ids.to(device=device, dtype=torch.int32)

te = pipeline.text_encoder.to(device)
prompt_embedding = textEncoder(text_input)
prompt_embedding.shape

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class Denoising(nn.Module):
    def __init__(self, unet, scheduler, device, num_inference_steps=50):
        super().__init__()
        self._execution_device = device #'cuda:4'
        self.scheduler = scheduler
        self.unet = unet
        self.unet = self.unet.to(device = device)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, 64, 64)
        #print(shape)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def forward(
        self, 
        prompt_embeds
    ):
        guidance_scale, guidance_rescale = 7.5, 0.0
        num_inference_steps = 50
        height, width = 512, 512    # 0. Default height and width to unet
        batch_size, num_images_per_prompt = 1, 1              # 2. Define call parameters
        device = self._execution_device

        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 5. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
            None
        )
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(None, 0.0)
        
        # 7. Denoising loop
        #with torch.no_grad():
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) #if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(device)
            prompt_embeds = prompt_embeds.to(device)
                
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
                
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            #break

        return latents
    
#model =  Denoising(pipeline.unet, pipeline.scheduler, 'cpu')
#out = model(prompt_embedding.to(device=device, dtype=torch.float32))
#out.shape

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")

def onnx_export(
    
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    print('ONNX export Start🚗')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    if is_torch_less_than_1_11:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            use_external_data_format=use_external_data_format,
            enable_onnx_checker=True,
            opset_version=opset,
        )
    else:
        export(
            model,
            model_args,
            f=output_path.as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset,
        )
    print('ONNX export Finish🍷')

onnx_export(
    Denoising(pipeline.unet, pipeline.scheduler, 'cpu'),
    model_args=(
        prompt_embedding.to(device=device, dtype=torch.float32)
    ),
    output_path = Path('./Denoising-CPU/model.onnx'),
    ordered_input_names=["encoder_hidden_states"],
    output_names=["out_sample"],  # has to be different from "sample" for correct tracing
    dynamic_axes={
        "encoder_hidden_states": {0: "batch", 1: "sequence"},
    },
    opset=14,
    use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
)