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

device = "cpu"
dtype = torch.float32

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)

text_input = pipeline.tokenizer(
    ["A smile cat"],
    padding="max_length",
    max_length=pipeline.tokenizer.model_max_length,     # 77
    truncation=True,
    return_tensors="pt",
    ).input_ids.to(device=device, dtype=torch.int32)

print(text_input.shape)