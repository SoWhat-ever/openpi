"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

"""
OpenPI项目中使用Pytorch训练 PI0 / PI05 模型的统一入口
支持单机单卡、单机多卡、多机多卡分布式训练
目标是与JAX版本的 scripts/train.py 行为保持一致,但完全基于Pytorch实现
复用统一的配置和数据加载管线 (src/openpi/training/config.py / src/openpi/training/data_loader.py)
"""

import dataclasses
import gc           # 垃圾回收，主动释放内存
import logging
import os
import platform
import shutil       # 高级文件操作
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist    # 分布式训练支持
import torch.nn.parallel            # 并行计算
import tqdm                         # 进度条
import wandb                        # 实验跟踪与可视化

# 导入内部模块
import openpi.models.pi0_config             # PI0 模型配置
import openpi.models_pytorch.pi0_pytorch    # Pytorch版PI0模型实现
import openpi.shared.normalize as _normalize    # 数据归一化
import openpi.training.config as _config        # 训练配置
import openpi.training.data_loader as _data     # 数据加载器


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    # 初始化 Weights & Biases （wandb）日志
    # config: 训练配置 resuming：是否为断点续训 enables：是否禁用
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled") # 禁用模式不会连接wandb服务器
        return

    # checkpoint目录，用于存储wandb_id或读取已有id
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:    # 续训
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()    # 从文件读取之前保存的wandb运行id
        wandb.init(id=run_id, resume="must", project=config.project_name)   # 使用id初始化wandb，设置resume=must确保必须成功恢复
    else:
        # 新的训练，创建wandb运行，记录完整配置
        wandb.init(
            name=config.exp_name,   # 实验名称
            config=dataclasses.asdict(config),  # 将参数dataclass配置转化为字典，便于wandb记录和可视化
            project=config.project_name,    # 项目名称
        )
        # 新生成的wandb ID保存到文件，用于后续可能的续训
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    # 初始化分布式训练环境 DDP（Distributed Data Parallel）

    # WORLD_SIZE：由 torchrun 自动设置，表示总进程数（总 GPU 数）
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # 如果 WORLD_SIZE > 1，说明在使用多进程训练 → 需要启用 DDP
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        # 选择后端，初始化进程组：GPU → NCCL，CPU → gloo
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://") # 多进程间此刻已能互相通信（all-reduce、broadcast 等）

        # PyTorch 的 DDP debug 模式，在开发阶段用来排查问题，很有用！！
        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    # LOCAL_RANK：本机的 GPU ID（例如单机 4 卡时分别是 0,1,2,3）
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    # 让每个进程绑定到：cuda: local_rank （GPU） 或 CPU（训练很慢，但兼容）
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # 设置当前 CUDA 设备
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    # 返回：use_ddp：是否启用 DDP；local_rank：当前进程绑定的 GPU；device：给 .to(device) 使用
    return use_ddp, local_rank, device


def cleanup_ddp():
    # 清理 DDP 资源
    # 在进程间做一次barrier，随后销毁进程组
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    # 让训练在多 GPU（DDP）环境中保持可控的、确定性的随机行为，避免所有 GPU 使用相同的随机数序列
    # 如果使用同一个 seed：所有 GPU 的 dropout、数据扰动、初始化都会完全一致
    # 这会导致模型训练的某些操作不再独立（例如 dropout mask 在多卡是一样的）
    # 导致某些 bug，如 loss 完全一模一样，影响训练真实随机性
    # 所以业界常用 pattern，都使用类似逻辑：
    # 模型初始化应该在 rank 0 控制，并在 DDP sync 后保持一致
    # 数据增强、dropout 的随机性则需要每 GPU 不同
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    # 使用统一的数据加载器接口，指定框架为Pytorch，shuffle=True启用数据随机打乱，提高训练稳定性并防止模型记住数据顺序
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    # 返回数据加载器和数据配置
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    # state_dict 是一个字典 [名字->张量]，包含：
    # 可训练参数（weights/bias）
    # buffers（如 BatchNorm 的 running_mean、running_var）
    # 其他不训练但需要保存的状态（如 LayerNorm 的 eps）

    # 参数model可能是被 DDP 包装过的 Pytorch 模型实例
    """Get state dict from model, handling DDP wrapper."""
    return (
        # 如果是DDP，从model.module获取 state_dict
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        # 否则直接从model获取
        else model.state_dict()
    )


def get_model_parameters(model):
    # parameters:只包含“可学习参数”的 iterator
    # 不能直接看到名字，只能拿到 Tensor 对象
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""

    # 分布式训练中只有主进程（rank 0）负责保存checkpoint
    # 避免多进程同时写文件
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    # 满足保存条件：当前步数是保存间隔的整数倍且大于0，或者是当前训练的最后一步
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        # 使用临时目录写入
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        # 清理旧的临时目录，创建新的
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        # 使用safetensors保存模型权重（比Pytorch安全，且支持内存映射加载，共享张量）
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        # Pytorch原生格式保存优化器状态
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        # 保存训练元数据，没有直接序列化完整config，避免跨框架不兼容
        metadata = {
            "global_step": global_step,             # 当前训练步数
            "config": dataclasses.asdict(config),   # 参数字典
            "timestamp": time.time(),               # 时间戳
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        # 保存数据归一化统计（若存在），路径对齐 assets 目录
        # 归一化统计对正确处理输入数据很重要，确保推理时使用与训练相同的归一化参数
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        # 原子删除再重命名
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        # 保存步数记录到wandb，用来可视化
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""

    # 查找所有有效目录：是目录；名称为纯数字；不是临时目录
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()   # 目录名转为int，方便找最大值
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    # 没找到任何checkpoint，抛异常
    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # 找到最大步数，构建最近checkpoint完整路径
    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    # 加载前尽可能释放显存，便于大模型加载，减少OOM风险
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    # 清空cuda显存
        gc.collect()    # 出发python垃圾回收
        log_memory_usage(device, latest_step, "before_loading_checkpoint")  # 记录加载前内存使用

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        # 加载模型权重（safetensors）
        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model # 处理 DDP
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))   # 加载到指定设备
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}") # 记录加载模型后内存使用

        # 加载完成后立刻清理内存，为加载优化器腾出空间
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # 加载优化器状态
        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            # 加载优化器状态字典，用map_location确保加载到正确设备
            # weights_only=False表示加载完整状态而非权重
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        # 将加载的状态应用到优化器
        optimizer.load_state_dict(optimizer_state_dict)
        # 删除临时变量释放内存
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")    # 记录加载优化器后内存使用

        # Load metadata
        # 加载训练元数据，与上面save_checkpoint对应，包含当前步数，配置参数，时间戳
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        # 获取全局步数，如果元数据中没有则使用目录名
        global_step = metadata.get("global_step", latest_step)
        # 删除临时变量释放内存
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")  # 加载元数据后内存使用
        return global_step  # 返回应继续训练的步数

    except RuntimeError as e:
        # 特殊处理 OOM 错误，提供错误信息
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    # 获取checkpoint目录中最新的步数编号
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""

    # 不是cuda，直接返回，避免后续代码出错
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9    # 当前已分配的GPU显存(实际被张量使用的显存)
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9  # 当前保留的GPU显存（Pytorch 申请的总显存）
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)  # 差值：空闲显存
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    # 更详细的显存统计信息
    memory_stats = torch.cuda.memory_stats(device)  # memory_stats包含更全面的内存使用统计
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9    # 历史峰值分配显存
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9  # 历史峰值使用显存

    # Get DDP info if available
    # 如果分布式环境，拼接rank / world_size信息
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    # 初始化分布式环境，设置DDP，获取本地进程排名，当前设备
    use_ddp, local_rank, device = setup_ddp()
    # 确定当前是否为主进程，分布式训练只有rank=0负责日志记录，checkpoint保存，wandb初始化
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    # 设置随机种子，分布式中以避免数据重复
    set_seed(config.seed, local_rank) # seed + local_rank

    # Initialize checkpoint directory and wandb
    resuming = False
    # 处于续训模式：在已有目录中选择最新可用checkpoint
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            # 获取最新步数
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    # 处于覆盖模式：删除checkpoint目录，确保新实验起点干净
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    # 新实验，创建新的checkpoint目录
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)   # parents=True确保父目录也被创建
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # 续训模式：复用已存在checkpoint目录
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    # 主进程上初始化wandb
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1   # 全局进程数
    effective_batch_size = config.batch_size // world_size  # 总batch_size划分到各个GPU进程
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)    # 划分数据

    # Log sample images to wandb on first batch
    # 新实验：采样若干张图像上传wandb
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    # 兼容 dataclass 和 Pi0Config 两种输入
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True        # 若模型支持，开启梯度checkpoint以降低显存占用（有的激活不保存，反向如果用到了再重新算）
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # 大规模训练：开启一系列性能/显存优化
    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        # 封装DDP，支持静态图/梯度桶共享等优化
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    # 若指定权重路径，则先加载进行微调
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps  # 预热步数
    peak_lr = config.lr_schedule.peak_lr            # 预热后达到的最大lr
    decay_steps = config.lr_schedule.decay_steps    # 衰减步数
    end_lr = config.lr_schedule.decay_lr            # 衰减后的最小lr

    # Create optimizer with config parameters
    # 使用Adam，与配置中超参对齐
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    # 如果续训，冲checkpoint中恢复模型和优化器，并获得起始步数
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        # warm up中线性上升，衰退中使用余弦退火
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()   # 切换model为训练模式
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        # 与DDP sample 对齐 epoch，保证各进程切分一致
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            # 将observation 和 action 迁移到目标设备
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            # 反向传播计算梯度
            loss.backward()

            # Log memory usage after backward pass
            # 前几步记录显存
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            # 梯度裁剪防止梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()    # 参数更新
            optim.zero_grad(set_to_none=True) # 置空梯度节省显存

            # Clear gradients more aggressively
            # 更彻底的释放grad引用，进一步减少显存
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            # 主进程收集日志数据
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):    # 按间隔上报
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    # 主函数入口
    # 初始化日志，解析CLI配置（复用 training / config.py 的 CLI）
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
