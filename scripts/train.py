# JAX: 高性能数据计算库（类似PyTorch的底层）
# Flax：基于JAX的神经网络库（类似PyTorch的 nn.Model）
# 直接看 train_pytorch.py 即可
import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    # 初始化日志系统，设置自定义日志格式
    """Custom logging format for better readability."""
    # 日志级别的简化映射
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    # 日志格式化器，设置日志格式
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    # 获取根日志器，应用配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):

    # 禁用，直接返回
    if not enabled:
        wandb.init(mode="disabled")
        return

    # 检查checkpoint目录是否存在
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    # 恢复训练：从文件读取之前的wandb运行id并恢复
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    # 新训练：创建新的wandb运行，将id保存到checkpoint目录中文件，用于后续恢复
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    # 如果启用代码日志：将源码上传到wandb
    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)  # 当前文件的父目录的父目录（根目录）


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    # 加载并验证模型权重
    # loader：权重加载器    params_shape：预期的参数shape，用于验证
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    # 检查参数shape和type符合预期
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    # 移除 jax.ShapeDtypeStruct 占位符
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    # 初始化训练状态：包括模型、参数、优化器；支持预训练权重初始化和恢复训练

    # 创建优化器
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        # 内部初始化函数，创建模型和训练状态
        
        # 生成随机数
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        # 初始化模型及其参数
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        # 如果提供了部分参数，合并到模型
        if partial_params is not None:
            # 分割模型的图定义和状态
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            # 更新模型的状态
            state.replace_by_pure_dict(partial_params)
            # 重新何必
            model = nnx.merge(graphdef, state)

        # 提取模型参数
        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        # 将冻结参数转化为 fp16 节省内存
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # 创建并返回训练状态对象
        return training_utils.TrainState(
            step=0,                                                     # 初始步数为 0
            params=params,                                              # 模型参数
            model_def=nnx.graphdef(model),                              # 模型图定义
            tx=tx,                                                      # 优化器
            opt_state=tx.init(params.filter(config.trainable_filter)),  # 优化器状态（只包含可训练参数）
            ema_decay=config.ema_decay,                                 # 指数移动平均衰减率
            ema_params=None if config.ema_decay is None else params,    # 指数移动平均参数
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # 执行单个的完整训练步骤：前向传播，损失计算，梯度计算，参数更新
    # config-训练配置对象 rng-随机数生成器密钥 state-当前训练状态 batch-训练批次数据（obs + action）

    # 合并模型定义和参数，重建完整对象
    model = nnx.merge(state.model_def, state.params)
    model.train()   # 训练模式

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # 损失计算函数，模型预测动作和真实动作之间损失
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        # 返回平均损失
        return jnp.mean(chunked_loss)
    
    # 为当前步骤创建一个唯一的随即数
    train_rng = jax.random.fold_in(rng, state.step)
    # 解包batch数据
    observation, actions = batch

    # Filter out frozen params.
    # 创建差分状态，只包含可训练参数（模型里只包含可训练参数的“状态视图”）
    diff_state = nnx.DiffState(0, config.trainable_filter)
    # 计算损失和梯度
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 过滤出可训练参数
    params = state.params.filter(config.trainable_filter)
    # 使用优化器更新参数
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    # 应用更新到参数
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    # 更新模型中的参数
    nnx.update(model, new_params)
    # 获取更新后的完整参数
    new_params = nnx.state(model)

    # 创建新的训练状态，步数+1，更新参数和优化器状态
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    # 如果启用了指数移动平均（EMA）,更新EMA参数
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    # 过滤出核心参数
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,  # 所有参数
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")), # 排除偏置、缩放、位置嵌入、输入嵌入
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # 收集训练信息
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),          # 梯度的全局范数
        "param_norm": optax.global_norm(kernel_params), # 核心参数的全局范数
    }
    return new_state, info


def main(config: _config.TrainConfig):
    # 主训练函数

    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 验证 batch 大小能否被设备数量整除（分布式训练要求）
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 设置JAX编译缓存目录，加速后续编译
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 创建随机数生成器，分割为训练和初始化用
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 创建分布式训练的网络和分片策略
    mesh = sharding.make_mesh(config.fsdp_devices)
    # 数据分片：沿数据轴分片
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # 复制分片策略：用于不需要分片的数据
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化checkpoint目录和管理器
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,          # checkpoint 目录
        keep_period=config.keep_period, # 保留检查点的周期
        overwrite=config.overwrite,     # 是否覆盖现有checkpoint
        resume=config.resume,           # 是否从检查点恢复
    )
    # 初始化wandb跟踪实验
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 创建数据加载器
    data_loader = _data_loader.create_data_loader(
        config,                     # 训练配置
        sharding=data_sharding,     # 数据分片策略
        shuffle=True,               # 是否打乱数据
    )
    # 创建数据迭代器
    data_iter = iter(data_loader)
    # 获取第一个批次的数据
    batch = next(data_iter)
    # 记录数据加载器的初始化信息
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 记录第一个batch的图像
    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练状态
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    # 等待初始化完成
    jax.block_until_ready(train_state)
    # 记录训练状态初始化信息
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果是恢复，从checkpoint恢复
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 使用JIT编译train_step函数，提高执行效率
    ptrain_step = jax.jit(
        functools.partial(train_step, config),      # 固定 config 参数并移除
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding), # 输入分片策略
        out_shardings=(train_state_sharding, replicated_sharding),  # 输出分片策略
        donate_argnums=(1,), # 因为config被移除了，现在的 1 指的是 TrainState;donate:在函数执行后不再使用，可以把它的内存直接拿去放新的 TrainState，极大节省内存
    )

    # 获取起始步数
    start_step = int(train_state.step)
    # 创建进度条
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # 收集训练信息的list
    infos = []
    # 训练主循环
    for step in pbar:
        # 设置当前线程的网络，用于分布式训练
        with sharding.set_mesh(mesh):
            # 执行一个训练步骤
            train_state, info = ptrain_step(train_rng, train_state, batch)
        # 收集训练信息
        infos.append(info)
        # 定期记录训练信息
        if step % config.log_interval == 0:
            # 将收集信息堆叠成树结构
            stacked_infos = common_utils.stack_forest(infos)
            # 计算平均值并从设备获取
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # 格式化信息字符串
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            # 进度条显示
            pbar.write(f"Step {step}: {info_str}")
            # 记录到wandb
            wandb.log(reduced_info, step=step)
            # 情况信息list
            infos = []
        # 获取下一个批次
        batch = next(data_iter)

        # 定期保存 checkpoint 或在训练结束时
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # 等待manager完成全部保存操作
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
