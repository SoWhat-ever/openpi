import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # eval_libero: 在 LIBERO benchmark 上对机器人策略进行评估，并保存回放视频

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict() # 返回所有可用任务套件字典
    task_suite = benchmark_dict[args.task_suite_name]() # 初始化具体任务套件
    num_tasks_in_suite = task_suite.n_tasks # 任务数量，用于循环评估
    logging.info(f"Task suite: {args.task_suite_name}")

    # 确保回放视频保存目录存在
    # arents=True → 如果上级目录不存在自动创建
    # exist_ok=True → 如果目录已经存在不报错
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True) 

    # 根据任务套件设置最大步数， max_steps 控制单次评估的最大步数
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # 初始化策略客户端
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    # 循环遍历任务套件， tqdm.tqdm 显示进度条
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # 获取每个任务对象和初始状态
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # 调用 _get_libero_env 初始化 LIBERO 环境 并获取任务描述
        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        # 循环执行每个任务的多个试验,用于统计成功率
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque() # 用来缓存动作序列

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []  # 保存回放视频帧

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    # 前几个时间步用 dummy 动作等待物体落地，防止仿真对象漂浮,args.num_steps_wait 控制等待时间
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    # 获取 主视角 和 手腕相机视角, 翻转 180° 以匹配训练数据
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    # 调整大小并转换为 uint8
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # 保存帧用于视频回放
                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        # 核心代码，准备字典，包含所有信息
                        element = {
                            "observation/image": img,   # 主视角
                            "observation/wrist_image": wrist_img, # 手腕相机
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],  # 末端执行器位置
                                    _quat2axisangle(obs["robot0_eef_quat"]),    # 末端执行器姿态
                                    obs["robot0_gripper_qpos"], # 机械臂 gripper 状态
                                )
                            ),
                            "prompt": str(task_description),    # 任务描述
                        }

                        # Query model to get action
                        # 调用策略服务器获取动作序列
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        # 每次规划 args.replan_steps 步，延迟执行
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # --------------- 分块规划，逐步执行 ---------------
                    action = action_plan.popleft()

                    # Execute action in environment
                    # 环境返回新的观察、奖励、done 标志
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            # 用 imageio 保存回放视频（主视角）
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    # 获取任务描述
    task_description = task.language
    # 构建 BDDL 文件路径： LIBERO数据集中BDDL文件根目录/任务文件夹/具体任务的 BDDL 文件
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # 构建环境参数字典: 指定任务定义文件,设置摄像机图像的分辨率
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    # OffScreenRenderEnv是LIBERO 提供的环境类，传入环境参数
    env = OffScreenRenderEnv(**env_args)
    # 物体布局也可能随种子变化， 通过 seed 固定随机数， 对于训练/评估可重复性非常重要
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    # 返回初始化好的 LIBERO 环境，任务的自然语言描述
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # 四元数（quaternion）转换成轴角（axis-angle）表示
    # 形式为 [x, y, z, w]，其中 (x, y, z) 是旋转轴向量，w 是旋转角度
    # clip quaternion

    # 限制四元数 w 分量在 [-1, 1],由于浮点误差，w 可能略大于 1 或小于 -1
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    # 轴角公式计算
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)
    # 返回 numpy.ndarray，维度为(3,)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero) # 把 eval_libero 包装成命令行程序，python eval.py --config_file=config.yaml --other_option=val
