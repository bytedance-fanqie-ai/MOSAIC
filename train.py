#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import json
import argparse
import copy
import gc
import itertools
import logging
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from typing import Union, Callable
import random
import shutil
import warnings
from os import path as osp
from contextlib import nullcontext
from pathlib import Path
from datetime import timedelta
import numpy as np
from copy import deepcopy
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.utils import InitProcessGroupKwargs
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
from safetensors.torch import save_file
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from datasets import load_dataset
from safetensors.torch import save_file, load_file

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, cast_training_params, free_memory
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    make_image_grid,
    convert_state_dict_to_diffusers
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


# omini Modules
from src.flux_omini import (
    generate,
    Condition
)

from src.flux_omini_mosaic import transformer_forward

# My Modules
from my_datasets.Subject200k_dataset import Subjects200K, make_collate_fn, make_collate_fn_w_coord

from utils import (
    count_parameters_in_M, 
    prepare_batched_data, 
    tensor2pil, 
    convert_png_to_rgb_with_white_bg, 
    special_sigmoid_smoothstep,
    subsample_dataset,
    process_image
)


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0")

logger = get_logger(__name__)



def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 / CLIP text encoder",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ema_interval",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--dev2pro",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--get_attn_maps",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--de_distill",
        action="store_true",
        default=False,
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--ref_size",type=int,default=512)
    parser.add_argument("--height",type=int,default=1024)
    parser.add_argument("--width",type=int,default=1024)
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--max_num_refs",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--ref_info_mask_prob",
        type=float,
        default=0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--caption_mask_prob",
        type=float,
        default=0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_prompt(tokenizer, prompt, max_sequence_length=512):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.unsqueeze(1)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
    )

    # text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    # text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def time_shift(mu: Union[float, torch.Tensor], sigma: float, t: torch.Tensor):
    if isinstance(mu, torch.Tensor):
        exp = torch.exp
    else:
        exp = math.exp
    return exp(mu) / (exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def sample_t_with_resolution(bsz, height, width, device, dtype):
    # 1. logistic-normal 基础采样
    t = torch.sigmoid(torch.randn((bsz,), device=device, dtype=dtype))

    # 2. resolution -> mu（分辨率越大，mu 越大）
    resolution = height * width
    mu = get_lin_function(y1=0.5, y2=1.15)(resolution)

    # 3. 分辨率相关的 time_shift
    t = time_shift(mu, 1.0, t)

    return t


def jsd_loss(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl_pm = (p * (p.add(eps).log() - m.add(eps).log())).sum()
    kl_qm = (q * (q.add(eps).log() - m.add(eps).log())).sum()
    return 0.5 * (kl_pm + kl_qm)


def sym_kl_divergence(p, q, eps=1e-8):
    """
    对称 KL 散度: 0.5 * KL(p‖q) + 0.5 * KL(q‖p)
    p, q: [tgt_len] 已经归一化的概率分布
    """
    p = p + eps
    q = q + eps
    kl_pq = (p * (p.log() - q.log())).sum()
    kl_qp = (q * (q.log() - p.log())).sum()
    return 0.5 * (kl_pq + kl_qp)


def compute_align_sep_losses(attn_map, coords, ref_len):
    """
    attn_map: Tensor [num_refs*ref_len, tgt_len]
    coords: list of dicts, len=num_refs
            each dict {ref_token_idx: tgt_token_idx}
    ref_len: number of tokens per reference
    """
    # num_refs = len(coords)
    num_refs = attn_map.shape[0] // ref_len
    tgt_len = attn_map.shape[1]

    # --- 1. token-level 对齐监督 ---
    losses_align = []
    for ref_idx in range(num_refs):
        for local_ref_idx, tgt_idx in coords[ref_idx].items():
            global_ref_idx = ref_idx * ref_len + local_ref_idx
            pred_dist = attn_map[global_ref_idx]  # [tgt_len]
            target = torch.tensor([tgt_idx], device=pred_dist.device)
            # loss = F.cross_entropy(pred_dist.unsqueeze(0), target)  # (1, tgt_len) vs (1,)  attn_map 已经做过一次 softmax，所以这里用 nll_loss 等价于 cross_entropy
            pred_log_prob = torch.log(pred_dist + 1e-9)  # [tgt_len]
            loss = F.nll_loss(pred_log_prob.unsqueeze(0), target)
            losses_align.append(loss)

    loss_align = torch.stack(losses_align).mean() if losses_align else torch.tensor(0.0, device=attn_map.device)

    # --- 2. ref-image level 分布分离 ---
    if num_refs >= 2:
        attn_map_reshaped = attn_map.view(num_refs, ref_len, tgt_len)  # [num_refs, ref_len, tgt_len]
        avg_tokens = attn_map_reshaped.mean(dim=1)  # [num_refs, tgt_len]
        # avg_tokens = F.normalize(avg_tokens, p=2, dim=-1)  # L2 归一化
        
        # JSD
        loss_sep = 0.0
        for i in range(num_refs):
            for j in range(i+1, num_refs):
                loss_sep += jsd_loss(avg_tokens[i], avg_tokens[j])
        loss_sep /= (num_refs * (num_refs-1) / 2)

        # 对称 KL 散度
        # loss_sep = 0.0
        # for i in range(num_refs):
        #     for j in range(i+1, num_refs):
        #         loss_sep += sym_kl_divergence(avg_tokens[i], avg_tokens[j])
        # loss_sep = loss_sep / (num_refs * (num_refs - 1) / 2)

    else:
        loss_sep = torch.tensor(0.0, device=attn_map.device)

    return loss_align, - loss_sep


@torch.no_grad()
def log_validation(args, pipe, device, weight_dtype, img_log_dir, step=0, num_images_per_prompt=1, postfix=''):

    position_deltas = []
    for i in range(args.max_num_refs):
        position_deltas.append([0, -(args.ref_size * (i + 1)) // 16])

    # 定义测试用的 cases
    test_cases = [
        {
            "prompt": "A toy car on a wooden floor.",
            "image_paths": [
                "/mnt/bn/shedong/consistent_image_generation/benchmarks/dreambench/rc_car/03_white_bg.jpg"
            ]
        }
    ]

    for case_idx, case in enumerate(test_cases):
        # 处理参考图像
        appearance_imgs = [
            process_image(p, target_size=args.ref_size, pad_color=(255, 255, 255), scale=0.9)
            for p in case["image_paths"]
        ]
        prompts = [case["prompt"]]

        conditions = [
            Condition(appearance, "default", position_deltas[i]) 
            for i, appearance in enumerate(appearance_imgs)
        ]

        num_refs = len(conditions)
        bsz = len(prompts)

        result_imgs = generate(
            pipe,
            prompt=prompts,
            conditions=conditions,
            num_inference_steps=28,
            num_images_per_prompt=num_images_per_prompt,
            height=args.height,
            width=args.width,
            guidance_scale=3.5,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )[0]

        # 拼图展示: 参考图像 + 生成结果
        image_list = []
        for row in range(bsz):
            for i in range(num_refs):
                image_list.append(appearance_imgs[i].resize((args.width, args.height)))
            for col in range(num_images_per_prompt):
                image_list.append(result_imgs[(row * num_images_per_prompt) + col])

        image_grid = make_image_grid(image_list, rows=bsz, cols=num_images_per_prompt + num_refs)
        image_grid.save(f"{img_log_dir}/global_step_{step:06d}{postfix}_{case_idx}.jpg")

    return


def main(args):

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    img_log_dir = osp.join(logging_dir, "images") # logs/images/ 前面不能有 / !!!
    ckpt_dir = osp.join(args.output_dir, "checkpoints") # logs/images/ 前面不能有 / !!!

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if img_log_dir is not None:
            os.makedirs(img_log_dir, exist_ok=True)
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path( # CLIP 
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path( # T5
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        # "/mnt/workspace/shedong/hf_model/black-forest-labs/FLUX-xhspro-sailu",
        subfolder="transformer",
        revision=args.revision, 
        variant=args.variant
    )

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    transformer.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    print(f"------------------------  offload: {args.offload} --------------------------------")
    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    vae.to(**to_kwargs)
    text_encoder_one.to(**to_kwargs)
    text_encoder_two.to(**to_kwargs)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
        
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        # target_modules=["to_q", "to_k", "to_v", "norm.linear", "norm1.linear"],
        target_modules="(.*x_embedder|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2|.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp|.*single_transformer_blocks\\.[0-9]+\\.proj_out|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)",
    )
    transformer.add_adapter(transformer_lora_config)

    # ema
    transformer_ema_dict = {
        f"module.{k}": deepcopy(v).requires_grad_(False) for k, v in transformer.named_parameters() if v.requires_grad
    } if args.ema else None

    # only upcast trainable parameters (LoRA) into fp32
    # cast_training_params(transformer, dtype=torch.float32)

    params_to_optimize = filter(lambda p: p.requires_grad, transformer.parameters())

    total_params_in_m, trainable_params_in_m = count_parameters_in_M(transformer)
    print(f"Total params: {total_params_in_m:.2f}M, trainable params: {trainable_params_in_m:.2f}M")

    # Optimizer creation
    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3, # None
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple, # True
            use_bias_correction=args.prodigy_use_bias_correction, # True
            safeguard_warmup=args.prodigy_safeguard_warmup, # True
        )

    # Dataset and DataLoaders creation ===============================
    # subject200k
    data_files = {"train":os.listdir("ByteDance-FanQie/SemAlign-MS-Subjects200K/data")}
    dataset_subject200k = load_dataset("parquet", data_dir="ByteDance-FanQie/SemAlign-MS-Subjects200K/data", data_files=data_files)
    def filter_func(item):
        if item.get("collection") != "collection_2":
            return False
        if not item.get("quality_assessment"):
            return False
        return all(
            item["quality_assessment"].get(key, 0) >= 5
            for key in ["compositeStructure", "objectConsistency", "imageQuality"]
        )
    dataset_subject200k_valid = dataset_subject200k["train"].filter( # 过滤出高质量
        filter_func,
        num_proc=16,
        cache_file_name="./cache/dataset/data_valid.arrow", # Optional
    )
    # 使用自定义类
    subject_dataset = Subjects200K(
        original_dataset=dataset_subject200k_valid,
        ref_size=args.ref_size,
        tgt_size=args.height,
        grounding_dir="ByteDance-FanQie/SemAlign-MS-Subjects200K/mask",
        coord_folder="ByteDance-FanQie/SemAlign-MS-Subjects200K/coord",
        mode="train", t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, # dropout for cfg
        add_postfix=False,
    )

    dataset_list = [
        subject_dataset, # 11w
    ]

    train_dataset = ConcatDataset(dataset_list)
    # train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset) if args.shuffle else None
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        # sampler=train_sampler,
        num_workers=args.dataloader_num_workers, 
        pin_memory=True, drop_last=True,
        collate_fn=make_collate_fn(num_refs=args.max_num_refs) \
            if not args.get_attn_maps else make_collate_fn_w_coord(num_refs=args.max_num_refs)
    )
    # ==============================================================

    tokenizers = [tokenizer_one, tokenizer_two]
    # text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "omini-multi"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            if global_step == 0 and accelerator.is_main_process:
                pipe = FluxPipeline(
                    vae=vae,
                    tokenizer=tokenizer_one,
                    tokenizer_2=tokenizer_two,
                    text_encoder=text_encoder_one.to(accelerator.device),
                    text_encoder_2=text_encoder_two.to(accelerator.device),
                    transformer=accelerator.unwrap_model(transformer),
                    scheduler=noise_scheduler
                ).to(accelerator.device)
                log_validation(args, pipe, accelerator.device, weight_dtype, img_log_dir, step=global_step, num_images_per_prompt=1)
                del pipe
                torch.cuda.empty_cache()
            # ============================================================
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                prepare_batched_data(batch, device=accelerator.device, dtype=weight_dtype) # batch data 转换 float32 --> weight_dtype
                
                num_refs = batch['ref_imgs'].shape[1]
                appearance_imgs = [] # [bs, 3, 512, 512]
                for i in range(num_refs):
                    appearance_imgs.append(batch['ref_imgs'][:, i])
                pixel_values = batch['tgt_imgs'] # [bs, 3, 512, 512]
                captions = batch['captions']
                if args.get_attn_maps:
                    coords = batch['coords'][0]

                drop_image = batch['drop_images'].bool()
                drop_text = batch['drop_texts'].bool()
                bsz = pixel_values.shape[0]

                captions = ["" if drop_text[i] == True else captions[i] for i in range(len(captions))]
                # Convert images to latent space
                with torch.no_grad():
                    
                    if args.offload:
                        vae = vae.to(accelerator.device)

                    appearance_latents = []
                    for i in range(num_refs):
                        appearance_latent = vae.encode(appearance_imgs[i]).latent_dist.mode()
                        appearance_latent = (appearance_latent - vae.config.shift_factor) * vae.config.scaling_factor
                        appearance_latent[drop_image] = 0
                        appearance_latents.append(appearance_latent)
                    
                    pixel_values = vae.encode(pixel_values).latent_dist.sample()
                    pixel_values = (pixel_values - vae.config.shift_factor) * vae.config.scaling_factor

                    if args.offload:
                        vae = vae.to("cpu")

                if args.offload:
                    text_encoder_one = text_encoder_one.to(accelerator.device)
                    text_encoder_two = text_encoder_two.to(accelerator.device)
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    captions, [text_encoder_one, text_encoder_two], tokenizers
                )
                if args.offload:
                    text_encoder_one = text_encoder_one.to("cpu")
                    text_encoder_two = text_encoder_two.to("cpu")

                model_input = pixel_values
                model_input = model_input.to(dtype=weight_dtype)

                packed_model_input = FluxPipeline._pack_latents(model_input, *model_input.shape)
                
                latent_image_ids = FluxPipeline._prepare_latent_image_ids( # [4096, 3]
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                cond_image_ids = FluxPipeline._prepare_latent_image_ids( # [1024, 3]
                    appearance_latents[0].shape[0],
                    appearance_latents[0].shape[2] // 2,
                    appearance_latents[0].shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                appearances_image_ids = []
                for i in range(num_refs):
                    appearances_image_ids.append(cond_image_ids.clone())
                    
                for i in range(num_refs):
                    appearances_image_ids[i][:, 2] -= (args.ref_size * (i + 1)) // 16 # 左移参考图的宽度

                # Sample noise that we'll add to the latents
                packed_noise = torch.randn_like(packed_model_input) # [1, 16, 64, 64]
                bsz = packed_model_input.shape[0]
                
                # t = torch.sigmoid(torch.randn((bsz,), device=accelerator.device, dtype=weight_dtype)) # logistic-normal distribution
                t = sample_t_with_resolution(bsz, args.height // 16, args.width // 16, accelerator.device, weight_dtype)
                t_ = t.unsqueeze(1).unsqueeze(1)
                packed_noisy_model_input = (1.0 - t_) * packed_model_input + t_ * packed_noise

                packed_appearances_latents = []
                for i in range(num_refs):
                    packed_appearances_latents.append(FluxPipeline._pack_latents(appearance_latents[i], *appearance_latents[i].shape))
                packed_condition_latents = []
                cond_image_ids = []

                for i in range(num_refs):
                    packed_condition_latents.append(packed_appearances_latents[i])
                    cond_image_ids.append(appearances_image_ids[i])
                
                # handle guidance
                guidance = torch.ones_like(t).to(accelerator.device) if model_config.guidance_embeds else None

                branch_n = 2 + len(packed_condition_latents) # txt + tgt + num_refs
                group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(accelerator.device)
                # Disable the attention cross different condition branches
                group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(packed_condition_latents)))

                # Predict the noise residual
                transformer_out = transformer_forward(
                    transformer,
                    image_features=[packed_noisy_model_input, *(packed_condition_latents)], # 解包 [x_t, c1, c2, c3]
                    text_features=[prompt_embeds],
                    img_ids=[latent_image_ids, *(cond_image_ids)],
                    txt_ids=[text_ids],
                    # There are three timesteps for the three branches
                    # (text, image, and the condition)
                    timesteps=[t, t] + [torch.zeros_like(t)] * len(packed_condition_latents),
                    # Same as above
                    pooled_projections=[pooled_prompt_embeds] * branch_n,
                    guidances=[guidance] * branch_n,
                    # The LoRA adapter names of each branch
                    adapters= [None, None] + ["default"] * num_refs,
                    return_dict=False,
                    group_mask=group_mask,
                    get_attn_maps=args.get_attn_maps,
                    get_attn_maps_single=False,
                )

                if args.get_attn_maps:
                    packed_pred, attn_maps = transformer_out
                else:
                    packed_pred = transformer_out[0]
                
                # Compute regular loss.
                diffusion_loss = torch.nn.functional.mse_loss(packed_pred, (packed_noise - packed_model_input), reduction="mean")
                
                if args.get_attn_maps:
                    # block_num = len(attn_maps)
                    # first_third_heads = list(attn_maps.values())[: block_num // 3]
                    # attn_maps_avg = torch.stack(first_third_heads, dim=0).mean(dim=0)
                    
                    attn_maps_avg = torch.stack(attn_maps, dim=0).mean(dim=0)
                    loss_align, loss_sep = compute_align_sep_losses(attn_maps_avg.squeeze(0), coords, ref_len=(args.ref_size // 16)**2)

                if global_step < 10000:
                    sep_weight = 0
                elif global_step < 11000:
                    sep_weight = (global_step - 10000) / 1000
                else:
                    sep_weight = 1.0

                if args.get_attn_maps:
                    loss = diffusion_loss + loss_align * 0.1 + loss_sep * sep_weight
                else:
                    loss = diffusion_loss

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if args.train_text_encoder
                        # else transformer.parameters()
                        else params_to_optimize
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.sync_gradients and transformer_ema_dict is not None and global_step % args.ema_interval == 0:
                    src_dict = transformer.state_dict()
                    for tgt_name in transformer_ema_dict:
                        transformer_ema_dict[tgt_name].data.lerp_(src_dict[tgt_name].to(transformer_ema_dict[tgt_name]), 1 - args.ema_decay)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if args.get_attn_maps:
                logs = {
                    "loss": loss.detach().item(),
                    "diffusion_loss": diffusion_loss.detach().item(),
                    "align_loss": loss_align.detach().item() * 0.1,
                    "sep_loss": loss_sep.detach().item() * sep_weight,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "timestep": t.mean().item(),
                }
            else:
                logs = {
                    "loss": loss.detach().item(),
                    "diffusion_loss": diffusion_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "timestep": t.mean().item(),
                }
                
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                transformer.eval()
                torch.set_grad_enabled(False)
                
                if accelerator.is_main_process:
                    
                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        pipe = FluxPipeline(
                            vae=vae,
                            tokenizer=tokenizer_one,
                            tokenizer_2=tokenizer_two,
                            text_encoder=text_encoder_one.to(accelerator.device),
                            text_encoder_2=text_encoder_two.to(accelerator.device),
                            transformer=accelerator.unwrap_model(transformer),
                            scheduler=noise_scheduler
                        ).to(accelerator.device)
                        log_validation(args, pipe, accelerator.device, weight_dtype, img_log_dir, step=global_step, num_images_per_prompt=1)
                        
                        del pipe
                        torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        save_path = os.path.join(ckpt_dir, f"checkpoint-{global_step}")
                        FluxPipeline.save_lora_weights(
                            save_directory=save_path,
                            transformer_lora_layers=get_peft_model_state_dict(unwrap_model(transformer)),
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

                    if args.ema and global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        original_state_dict = transformer.state_dict()
                        transformer.load_state_dict(transformer_ema_dict, strict=False)

                        pipe = FluxPipeline(
                            vae=vae,
                            tokenizer=tokenizer_one,
                            tokenizer_2=tokenizer_two,
                            text_encoder=text_encoder_one.to(accelerator.device),
                            text_encoder_2=text_encoder_two.to(accelerator.device),
                            transformer=accelerator.unwrap_model(transformer),
                            scheduler=noise_scheduler
                        ).to(accelerator.device)
                        log_validation(args, pipe, accelerator.device, weight_dtype, img_log_dir, step=global_step, num_images_per_prompt=1, postfix='_ema')
                        
                        transformer.load_state_dict(original_state_dict, strict=False)

                        del pipe
                        torch.cuda.empty_cache()

                    if args.ema and global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        accelerator.save(
                            {"transformer." + k.split("module.")[-1].replace("default.", ""): v for k, v in transformer_ema_dict.items()},
                            os.path.join(save_path, 'pytorch_lora_weights_ema.safetensors'),
                            safe_serialization=True
                        )

                torch.cuda.empty_cache()
                gc.collect()
                torch.set_grad_enabled(True)
                transformer.train()
                accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)