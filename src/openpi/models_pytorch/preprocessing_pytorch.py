from collections.abc import Sequence
import logging

import torch

from openpi.shared import image_tools   # 自定义的图像处理工具

logger = logging.getLogger("openpi")

# Constants moved from model.py
# 定义默认的图像名，表示不同相机视角
IMAGE_KEYS = (
    "base_0_rgb",           # 基座相机RGB图像
    "left_wrist_0_rgb",     # 左手腕相机RGB图像
    "right_wrist_0_rgb",    # 右手腕相机RGB图像
)

IMAGE_RESOLUTION = (224, 224)   # 标准图像分辨率 224 * 224


def preprocess_observation_pytorch(
    observation,    # 输入的观测图像
    *,
    train: bool = False,    # 是否处于训练模式，默认False
    image_keys: Sequence[str] = IMAGE_KEYS,     # 要处理的图像列表
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,   # 目标分辨率
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """

    # 检查观测数据中是否有要处理的图像名
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # 获得batch_shape，排除最后一个特征维度
    batch_shape = observation.state.shape[:-1]

    # 输出图像列表
    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        # 处理不同图像格式，根据第二个维度是否是3（通道数）来判断格式
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        # 通道优先格式转化为通道末尾格式
        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        # 如果分辨率不符合，进行调整
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            # 使用带padding的调整大小方法，来保持纵横比
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        # 如果训练模式，应用数据增强
        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            # [-1, 1] 转化为 [0, 1]
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            # 对手腕相机应用几何增强
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                # 随机裁剪和调整大小的参数
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height    # 高度的最大偏移量
                max_w = width - crop_width      # 宽度的最大偏移量
                # 确保有足够空间进行裁剪
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    # 随机生成裁剪起始位置
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    # 执行裁剪
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                # 调整回原大小
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),   # 目标大小为原始尺寸
                    mode="bilinear",        # 使用双线性插值
                    align_corners=False,    # 不对齐角点，防止边缘伪影
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]   # 转回原格式

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                # 随机小角度旋转
                angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    # 角度转化为弧度
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    # 计算正弦余弦
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    # 创建网格坐标
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    # 创建网格
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    # 扩展到批次处理
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    # 应用旋转变换
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    # 堆叠并重塑网格以用于grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    # 应用网格采样进行旋转
                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations for all cameras
            # Random brightness
            # Use tensor operations instead of .item() for torch.compile compatibility
            # 对所有图像颜色增强，随机亮度调整
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
            image = image * brightness_factor

            # Random contrast
            # Use tensor operations instead of .item() for torch.compile compatibility
            # 随机对比度
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # Random saturation (convert to HSV, modify S, convert back)
            # For simplicity, we'll just apply a random scaling to the color channels
            # Use tensor operations instead of .item() for torch.compile compatibility
            # 随机饱和度
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            # 限制到 [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    # 获取图像掩码
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 默认不适用掩码，返回全1
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            # 使用提供掩码
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    # 创建简单对象包含所需属性
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # 返回包含处理后数据的简化观测对象
    return SimpleProcessedObservation(
        images=out_images,                                          # 处理后的图像dict
        image_masks=out_masks,                                      # 处理后的图像mask
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
