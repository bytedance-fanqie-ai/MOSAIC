import os
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
from PIL import Image, ImageOps
import gc

import random
from torch.utils.data import Subset


def subsample_dataset(dataset, ratio=0.1, seed=12345):
    """随机抽取 ratio 比例的数据"""
    n = len(dataset)
    k = int(n * ratio)
    random.seed(seed)
    indices = random.sample(range(n), k)
    return Subset(dataset, indices)


def special_sigmoid_smoothstep(x, step):
    def smoothstep(edge0, edge1, x):
        """
        平滑阶梯函数（SmoothStep）:
        计算 \( t = \text{clamp}((x - edge0)/(edge1 - edge0), 0, 1) \)
        返回 \( t * t * (3 - 2 * t) \)
        """
        t = torch.clamp((x - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)

    return torch.where(
        x <= 0,
        torch.zeros_like(x),
        torch.where(
            x < step,
            smoothstep(0, step, x),
            torch.ones_like(x)
        )
    )

def special_sigmoid_smoothstep_v2(x, start_step, end_step):
    def smoothstep(edge0, edge1, x):
        """
        平滑阶梯函数（SmoothStep）:
        计算 \( t = \text{clamp}((x - edge0)/(edge1 - edge0), 0, 1) \)
        返回 \( t * t * (3 - 2 * t) \)
        """
        t = torch.clamp((x - edge0) / (edge1 - edge0), 0, 1)
        return t * t * (3 - 2 * t)

    return torch.where(
        x <= start_step,
        torch.zeros_like(x),
        torch.where(
            x < end_step,
            smoothstep(start_step, end_step, x),
            torch.ones_like(x)
        )
    )

def prepare_batched_data(batch, device, dtype):
        """Recursively move tensors in a nested dict to the device and cast to weight_dtype."""
        for key, value in batch.items():
            if isinstance(value, dict):
                batch[key] = prepare_batched_data(value, device, dtype)
            elif isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    batch[key] = value.unsqueeze(0).to(device).to(dtype)
                else:
                    batch[key] = value.to(device).to(dtype)
        return batch


def count_parameters_in_M(model: nn.Module):
    """
    计算模型的总参数量和可训练参数量，并以百万 (M) 为单位输出.
    
    Args:
        model (nn.Module): 要计算的 PyTorch 模型.
    Returns:
        tuple: 总参数量和可训练参数量，以百万 (M) 为单位
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params_in_m = total_params / 1e6
    trainable_params_in_m = trainable_params / 1e6
    
    return total_params_in_m, trainable_params_in_m
    # print(f"{model.__class__.__name__} total params: {total_params_in_m:.2f}M, trainable params: {trainable_params_in_m:.2f}M")


def tensor2pil(tensor, width=512, height=512):
    target_size = (width, height)
    # 确保 tensor 是 PyTorch 张量
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('tensor must be a PyTorch tensor.')

    # 移除多余的维度 (例如，从 [1, H, W] 变为 [H, W])
    if tensor.ndim > 2:
        tensor = tensor.squeeze()
    
    # 转换为NumPy数组
    array_img = tensor.cpu().to(dtype=torch.float32).numpy()
    
    # 如果维度为 [C, H, W]，转换为 [H, W, C]
    if array_img.ndim == 3 and array_img.shape[0] in [1, 3]:
        array_img = array_img.transpose(1, 2, 0)
    
    # 检查并处理单通道图像
    if array_img.ndim == 2 or array_img.shape[2] == 1:
        array_img = np.expand_dims(array_img, axis=-1)
        array_img = np.repeat(array_img, 3, axis=-1)

    # 确保数据范围在 [0, 1] 之间
    array_img = np.clip(array_img, 0, 1)
    
    # 转换为 uint8
    array_img = (array_img * 255).astype(np.uint8)

    # 使用 PIL.Image.fromarray 创建 PIL 图片
    pil_image = Image.fromarray(array_img)
    # 将图像 resize 到目标大小
    pil_image = pil_image.resize(target_size, Image.LANCZOS)
    
    return pil_image


def convert_png_to_rgb_with_white_bg(input_path, pad_color=(255, 255, 255)):
    """
    将带透明背景的 PNG 图像转换为带白色背景的 RGB 图像。

    参数:
        input_path (str): 输入 PNG 图像的路径。
        pad_color (tuple): 填充颜色，默认为白色。
    
    返回:
        PIL.Image: 转换后的 RGB 图像。
    """
    image = Image.open(input_path)
    
    # 检查图像是否有透明通道
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert("RGBA")
        # 创建白色背景
        white_bg = Image.new("RGB", image.size, pad_color)
        # 合成图像
        white_bg.paste(image, mask=image.split()[3])  # 3 是 alpha 通道
        final_image = white_bg
    else:
        final_image = image.convert("RGB")
    
    return final_image

def get_bounding_box(image, bg_color=(255, 255, 255)):
    """
    获取图像中非背景部分的最小边界框。

    参数:
        image (PIL.Image): 输入的 RGB 图像。
        bg_color (tuple): 背景颜色，默认为白色。
    
    返回:
        tuple: (left, upper, right, lower) 边界框坐标
    """
    # 将图像转换为 numpy 数组
    np_image = np.array(image)
    
    # 如果图像有透明度，忽略 alpha 通道
    if np_image.shape[2] == 4:
        np_image = np_image[:, :, :3]
    
    # 找到非背景像素
    mask = (np_image != bg_color).any(axis=2)
    
    if not mask.any():
        # 如果整个图像都是背景，返回全图
        return (0, 0, image.width, image.height)
    
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # slices are exclusive at the top
    
    return (x_min, y_min, x_max, y_max)

def resize_with_bbox(image, target_size, pad_color=(255, 255, 255), scale=0.8):
    """
    根据物体的最小边界框调整图像大小并填充到目标尺寸。

    参数:
        image (PIL.Image): 输入的 RGB 图像。
        target_size (int): 目标图像的宽度和高度（假设为正方形）。
        pad_color (tuple): 填充颜色，默认为白色。
        scale (float): 物体在目标图像中所占比例，默认为 0.9。
    
    返回:
        PIL.Image: 处理后的图像。
    """
    # 获取物体的边界框
    bbox = get_bounding_box(image, bg_color=pad_color)
    cropped = image.crop(bbox)
    cropped_width, cropped_height = cropped.size

    # 确定缩放比例
    long_side = max(cropped_width, cropped_height)
    scale_factor = (target_size * scale) / long_side
    new_width = int(cropped_width * scale_factor)
    new_height = int(cropped_height * scale_factor)
    
    # 缩放图像
    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 创建白色背景
    final_image = Image.new("RGB", (target_size, target_size), pad_color)
    
    # 计算填充位置
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # 粘贴缩放后的图像到背景
    final_image.paste(resized, (paste_x, paste_y))
    
    return final_image

def process_image(image_path, target_size, pad_color=(255, 255, 255), scale=0.8):
    """
    完整的图像处理流程：转换背景、调整大小和填充。

    参数:
        image_path (str): 输入图像的路径。
        target_size (int): 目标图像的宽度和高度。
        pad_color (tuple): 填充颜色，默认为白色。
        scale (float): 物体在目标图像中所占比例，默认为 0.9。
    
    返回:
        PIL.Image: 处理后的图像。
    """
    # 转换背景为白色
    image_pil = convert_png_to_rgb_with_white_bg(image_path, pad_color=pad_color)
    
    # 根据边界框调整大小并填充
    image_pil = resize_with_bbox(image_pil, target_size=target_size, pad_color=pad_color, scale=scale)
    
    return image_pil