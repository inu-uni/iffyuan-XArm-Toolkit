# LeRobot 集成：XArm6 + SpaceMouse 数据采集
## 本文档经过测试，代码无误，但是太卡了，不好用

> 本文档覆盖：LeRobot 插件安装、SpaceMouse 遥操作采集、CLI 参数详解、数据格式说明、与原有 Zarr 采集的对比。

## 整体架构

```
LeRobot CLI (lerobot-record / lerobot-teleoperate / lerobot-replay)
       ↓ 内置 xarm6 + spacemouse_xarm6 适配器
lerobot/src/lerobot/robots/xarm6/           →  包装 XArmEnv + RealsenseEnv
lerobot/src/lerobot/teleoperators/spacemouse_xarm6/  →  包装 SpacemouseAgent
       ↓ 适配层调用（不改动 xarm_toolkit/）
xarm_toolkit/env/xarm_env.py      (机械臂控制)
xarm_toolkit/env/realsense_env.py  (双相机)
xarm_toolkit/teleop/spacemouse.py  (SpaceMouse 6DOF)
       ↓
采集数据 → LeRobot Dataset v3.0 (Parquet + MP4)
```

**与原有方案的关系**：这是一套**独立的**数据采集方案。原有的 `scripts/data/collect_data.py` → Zarr 方案完全不受影响，两者可以共存。

---

## 一、安装

### 1.1 前置条件

确保以下已安装：

```bash
# 1. xarm-toolkit（本项目主包）
cd /path/to/iffyuan-XArm-Toolkit
pip install -e .

# 2. LeRobot（本地源码，已内置 XArm6 + SpaceMouse 适配器）
cd lerobot && pip install -e . && cd ..
```

### 1.2 验证安装

```bash
# 检查 xarm6 是否被 LeRobot 识别
python -c "
from lerobot.robots.xarm6 import Xarm6Config
from lerobot.teleoperators.spacemouse_xarm6 import SpacemouseXarm6Config
print('✅ Robot:      xarm6')
print('✅ Teleop:     spacemouse_xarm6')
print('   Default IP:       ', Xarm6Config().ip_address)
print('   Default cameras:  ', Xarm6Config().cam_arm_serial, '+', Xarm6Config().cam_fix_serial)
print('   Default image:    ', f'{Xarm6Config().image_width}x{Xarm6Config().image_height}')
"
```

---

## 二、快速开始

### 2.1 遥操作测试（不保存数据） ✅ 已测试通过

先测试 SpaceMouse + 机械臂是否正常工作：

```bash
lerobot-teleoperate \
  --robot.type=xarm6 \
  --teleop.type=spacemouse_xarm6
```

移动 SpaceMouse 应该能看到机械臂跟随运动，左键切换夹爪开/关。

### 2.2 数据采集 ⚠️ 功能已实现，未经完整测试

```bash
lerobot-record \
  --robot.type=xarm6 \
  --teleop.type=spacemouse_xarm6 \
  --dataset.repo_id=iffyuan/xarm6_pick_place \
  --dataset.single_task="Pick the object and place it in the box" \
  --dataset.root=data/xarm6_pick_place \
  --dataset.fps=10 \
  --dataset.num_episodes=10 \
  --dataset.video=false \
  --dataset.push_to_hub=false \
  --display_data=true
```

> 加 `--display_data=true` 会弹出 Rerun 可视化窗口，实时显示相机画面和机器人状态。不需要可视化时去掉此参数。

采集完成后数据自动保存为 **LeRobot Dataset v3.0** 格式。

### 2.3 回放验证 ⚠️ 功能已实现，未经测试

```bash
lerobot-replay \
  --robot.type=xarm6 \
  --dataset.repo_id=iffyuan/xarm6_pick_place \
  --dataset.root=data/xarm6_pick_place \
  --dataset.episode=0
```

---

## 三、CLI 参数详解

### 3.1 Robot 参数 (`--robot.*`)

所有 Robot 参数通过 `--robot.xxx` 传入，对应 `Xarm6Config` 的字段：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--robot.type` | — | **必填**，固定填 `xarm6` |
| `--robot.ip_address` | `192.168.31.232` | XArm 控制器 IP 地址 |
| `--robot.action_mode` | `delta_eef` | 控制模式：`delta_eef` / `absolute_eef` / `absolute_joint` |
| `--robot.initial_gripper_position` | `840` | 初始夹爪位置（0=闭，840=全开） |
| `--robot.cam_arm_serial` | `327122075644` | 臂上相机序列号（D435i） |
| `--robot.cam_fix_serial` | `f1271506` | 固定相机序列号（L515） |
| `--robot.image_width` | `320` | 采集图像宽度（像素） |
| `--robot.image_height` | `240` | 采集图像高度（像素） |

### 3.2 Teleoperator 参数 (`--teleop.*`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teleop.type` | — | **必填**，固定填 `spacemouse_xarm6` |
| `--teleop.translation_scale` | `5.0` | 平移灵敏度（mm/tick） |
| `--teleop.z_scale` | `None` | Z 轴灵敏度，None 时同 translation_scale |
| `--teleop.rotation_scale` | `0.004` | 旋转灵敏度（rad/tick） |
| `--teleop.deadzone` | `0.0` | 死区阈值，低于此值视为 0 |
| `--teleop.gripper_open_pos` | `840` | 夹爪打开位置 |
| `--teleop.gripper_close_pos` | `0` | 夹爪关闭位置 |

### 3.3 Record 参数 (`--dataset.*`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset.repo_id` | — | **必填**，数据集 ID，格式 `用户名/数据集名` |
| `--dataset.single_task` | — | **必填**，任务描述（如 "Pick the object and place it"） |
| `--dataset.root` | `None` | 本地存储路径，None 则存到 `$HF_LEROBOT_HOME/repo_id` |
| `--dataset.fps` | `30` | 采集帧率（推荐 VLA 训练用 10~15） |
| `--dataset.episode_time_s` | `60` | 每个 episode 时长（秒） |
| `--dataset.reset_time_s` | `60` | episode 间重置等待时间（秒） |
| `--dataset.num_episodes` | `50` | 采集 episode 数量 |
| `--dataset.video` | `true` | 是否将图像编码为 MP4 视频 |
| `--dataset.push_to_hub` | `true` | 采集完成后是否上传 HuggingFace Hub |
| `--dataset.private` | `false` | Hub 上是否设为私有仓库 |

### 3.4 Replay 参数 (`--dataset.*`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset.repo_id` | — | **必填**，数据集 ID |
| `--dataset.episode` | — | **必填**，回放的 episode 编号 |
| `--dataset.root` | `None` | 本地数据集路径 |
| `--dataset.fps` | `30` | 回放帧率 |

---

## 四、常用采集场景

### 4.1 精细操作（降低灵敏度）

```bash
lerobot-record \
  --robot.type=xarm6 \
  --teleop.type=spacemouse_xarm6 \
  --teleop.translation_scale=2.5 \
  --teleop.rotation_scale=0.002 \
  --teleop.deadzone=0.1 \
  --dataset.repo_id=iffyuan/xarm6_plug \
  --dataset.single_task="Insert the plug into the socket" \
  --dataset.root=data/xarm6_plug \
  --dataset.num_episodes=20 \
  --dataset.fps=10 \
  --dataset.video=false \
  --dataset.push_to_hub=false
```

### 4.2 更换机械臂 IP

```bash
lerobot-record \
  --robot.type=xarm6 \
  --robot.ip_address=192.168.1.100 \
  --teleop.type=spacemouse_xarm6 \
  --dataset.repo_id=iffyuan/xarm6_demo \
  --dataset.single_task="Demo task" \
  --dataset.root=data/xarm6_demo \
  --dataset.num_episodes=5 \
  --dataset.video=false \
  --dataset.push_to_hub=false
```

### 4.3 更高分辨率图像

```bash
lerobot-record \
  --robot.type=xarm6 \
  --robot.image_width=640 \
  --robot.image_height=480 \
  --teleop.type=spacemouse_xarm6 \
  --dataset.repo_id=iffyuan/xarm6_hires \
  --dataset.single_task="Pick and place with high resolution" \
  --dataset.root=data/xarm6_hires \
  --dataset.num_episodes=10 \
  --dataset.video=false \
  --dataset.push_to_hub=false
```

### 4.4 关节空间控制模式

```bash
lerobot-record \
  --robot.type=xarm6 \
  --robot.action_mode=absolute_joint \
  --teleop.type=spacemouse_xarm6 \
  --dataset.repo_id=iffyuan/xarm6_joint \
  --dataset.single_task="Joint space control demo" \
  --dataset.root=data/xarm6_joint \
  --dataset.num_episodes=10 \
  --dataset.video=false \
  --dataset.push_to_hub=false
```

> **注意**：SpaceMouse 输出的是 delta EEF，如果 action_mode 设为 `absolute_joint`，则 Robot 内部仍然通过 EEF delta 控制，只是观测特征中的 action 定义不同。一般遥操作推荐用默认的 `delta_eef`。

---

## 五、数据格式

LeRobot 采集的数据为 **LeRobot Dataset v3.0** 格式（LeRobot ≥ 0.5），和原有的 Zarr 格式不同。

### 5.1 目录结构

```
data/xarm6_pick_place/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet              # 数值数据（state, action, 帧索引）
│       └── ...
├── videos/
│   └── observation.images.image/         # 固定相机视频（主视角）
│       └── chunk-000/
│           ├── file-000.mp4
│           └── ...
│   └── observation.images.wrist_image/   # 臂上相机视频（手腕视角）
│       └── chunk-000/
│           ├── file-000.mp4
│           └── ...
└── meta/
    ├── info.json                         # 数据集配置（robot_type, fps, features, codebase_version）
    ├── stats.json                        # 各特征的统计量（均值、方差、min、max）
    ├── tasks.parquet                     # 任务描述
    └── episodes/
        └── chunk-000/
            └── file-000.parquet          # Episode 元数据（起止索引、长度等）
```

> v3.0 使用 chunk/file 分层组织，支持大规模数据集。`--dataset.video=false` 时，图像以 PNG 文件存储在 `images/` 目录下，不生成 `videos/`。

### 5.2 字段映射

| LeRobot key | 来源 | shape | 说明 |
|---|---|---|---|
| `observation.state` | [x, y, z, roll, pitch, yaw, gripper] | **(7,) float32** | 末端位姿 (mm/rad) + 夹爪状态 (0=闭/1=开) |
| `action` | [dx, dy, dz, droll, dpitch, dyaw, gripper_action] | **(7,) float32** | SpaceMouse 6D 增量 + 夹爪动作 (0=闭/1=开) |
| `observation.images.image` | rgb_fix（固定相机 L515） | **(H, W, 3) uint8** | 主视角 RGB |
| `observation.images.wrist_image` | rgb_arm（臂上相机 D435i） | **(H, W, 3) uint8** | 手腕视角 RGB |

### 5.3 Parquet 数据列

每个 chunk 的 Parquet 文件包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `index` | int | 全局帧索引 |
| `episode_index` | int | Episode 编号 |
| `frame_index` | int | Episode 内帧索引 |
| `timestamp` | float | 时间戳（秒） |
| `task_index` | int | 任务编号（对应 tasks.parquet） |
| `observation.state` | float32[7] | [x, y, z, roll, pitch, yaw, gripper] |
| `observation.images.image` | — | 固定相机帧（引用视频/图像文件） |
| `observation.images.wrist_image` | — | 臂上相机帧（引用视频/图像文件） |
| `action` | float32[7] | [dx, dy, dz, droll, dpitch, dyaw, gripper_action] |

### 5.4 读取数据（Python）

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载本地数据集
dataset = LeRobotDataset("iffyuan/xarm6_pick_place", root="data/xarm6_pick_place")

# 查看基本信息
print(f"总 episodes: {dataset.num_episodes}")
print(f"总帧数: {len(dataset)}")
print(f"FPS: {dataset.fps}")

# 读取第 0 帧
frame = dataset[0]
print(frame.keys())
# → dict_keys(['observation.state', 'observation.images.image',
#              'observation.images.wrist_image', 'action', ...])

# observation.state 是 (7,) 向量
state = frame["observation.state"]   # [x, y, z, roll, pitch, yaw, gripper]
action = frame["action"]             # [dx, dy, dz, droll, dpitch, dyaw, gripper_action]
```

---

## 六、观测与动作特征说明

插件注册到 LeRobot 的特征映射如下（与 Pi0.5 等 VLA 模型输入格式对齐）：

### 6.1 Observation Features

| 特征名 | shape | dtype | 说明 |
|--------|-------|-------|------|
| `observation.state` | (7,) | float32 | [x, y, z, roll, pitch, yaw, gripper]，xyz 单位 mm，角度 rad，gripper: 0=闭/1=开 |
| `observation.images.image` | (240, 320, 3) | uint8 | 固定相机 L515 RGB（主视角） |
| `observation.images.wrist_image` | (240, 320, 3) | uint8 | 臂上相机 D435i RGB（手腕视角） |

### 6.2 Action Features

| 特征名 | shape | dtype | 说明 |
|--------|-------|-------|------|
| `action` | (7,) | float32 | [dx, dy, dz, droll, dpitch, dyaw, gripper_action]，gripper_action: 0=闭/1=开 |

### 6.3 夹爪映射

| 原始夹爪位置 | observation.state[6] | action[6] |
|:---:|:---:|:---:|
| ≤ 420 | 0.0 (闭合) | 0.0 (关闭) |
| > 420 | 1.0 (张开) | 1.0 (打开) |

---

## 七、与原有 Zarr 采集方案的对比

| | 原有方案 (`collect_data.py`) | LeRobot 方案 (`lerobot-record`) |
|---|---|---|
| **数据格式** | Zarr + Blosc 压缩 | LeRobot Dataset v3.0 (Parquet + MP4) |
| **图像存储** | 原始 uint8 数组 (channel-first) | H.264 视频压缩 |
| **深度图** | ✅ 支持 (uint16) | ❌ 仅 RGB |
| **力传感器** | ✅ 支持 (`--force`) | ❌ 不包含 |
| **数据体积** | 较大（原始像素） | 较小（视频压缩） |
| **生态** | 本地使用，需自己写 DataLoader | HuggingFace Hub 生态，直接喂训练 |
| **遥操作** | SpaceMouse 6DOF + 键盘控制 | SpaceMouse 6DOF（通过内置适配器） |
| **Episode 控制** | 空格开始/Enter 结束 | LeRobot 内置流程 |
| **增量追加** | ✅ | ✅ |
| **夹爪录制逻辑** | 左右键独立控制 + gripper_always_closed | 左键切换开/关 |

### 选择建议

- **要训练 LeRobot 支持的策略**（ACT、Diffusion Policy 等）→ 用 `lerobot-record`
- **需要深度图或力传感器数据** → 用原有 `collect_data.py`
- **两者可以共存**，分别采集互不影响

---

## 八、适配器文件说明

XArm6 和 SpaceMouse 适配器直接内置在本项目的 `lerobot/` 源码中：

```
lerobot/src/lerobot/
├── robots/xarm6/                              # XArm6 Robot 适配器
│   ├── __init__.py
│   ├── config_xarm6.py                        # 配置类（IP、相机、分辨率等）
│   └── xarm6.py                               # Robot 实现（包装 XArmEnv + RealsenseEnv）
└── teleoperators/spacemouse_xarm6/            # SpaceMouse Teleoperator 适配器
    ├── __init__.py
    ├── config_spacemouse_xarm6.py             # 配置类（灵敏度、死区等）
    └── spacemouse_xarm6.py                    # Teleoperator 实现（包装 SpacemouseAgent）
```

**注册原理**：`config_xarm6.py` 中使用 `@RobotConfig.register_subclass("xarm6")` 注册到 draccus 的 ChoiceRegistry，同时在 `robots/utils.py` 的工厂函数中加入 `xarm6` 分支。`lerobot-teleoperate --robot.type=xarm6` 即可使用。

---

## 九、常见问题

| 问题 | 排查 |
|------|------|
| `Unknown robot type: xarm6` | 确认 `cd lerobot && pip install -e .` 已执行 |
| `ModuleNotFoundError: xarm` | 确认主项目 `pip install -e .` 已执行（pyproject.toml 已将 xarm SDK 包含） |
| `ModuleNotFoundError: xarm_toolkit` | 确认主项目 `pip install -e .` 已执行 |
| `ModuleNotFoundError: lerobot` | 确认 `cd lerobot && pip install -e .` 已执行 |
| `pyspacemouse has no attribute 'read'` | pyspacemouse 版本太新，降级：`pip install pyspacemouse==1.0.4` |
| `Missing required field 'dataset'` | `lerobot-record` 必须传 `--dataset.repo_id` 和 `--dataset.single_task` |
| SpaceMouse 无反应 | `lsusb` 检查 USB 连接；确认 hidapi 已安装 |
| 相机连接失败 | 先跑 `python scripts/test_cameras.py` 单独验证 |
| 机械臂连接超时 | 检查 IP 地址和网络，`ping 192.168.31.232` |
| 图像全黑 | 检查相机序列号是否正确（`--robot.cam_arm_serial=...`） |
| `cv2.resize` 空图崩溃 | 相机启动初期帧为空，已内置重试逻辑，一般自动恢复 |
| action_features 不匹配 | 确保 `--robot.action_mode` 和 `--teleop.type` 一致（默认 delta_eef + spacemouse_xarm6） |

---

## 十、测试状态

| 功能 | 状态 | 说明 |
|------|------|------|
| **安装 & import** | ✅ 已测试 | Python import 和 CLI type 注册均通过 |
| **2.1 遥操作 (teleoperate)** | ✅ 已测试 | SpaceMouse 控制机械臂运动正常 |
| **2.2 数据采集 (record)** | ⚠️ 未完整测试 | 功能已实现，启动流程跑通，但尚未完成完整 episode 采集验证 |
| **2.3 回放 (replay)** | ⚠️ 未测试 | 功能已实现，依赖 record 产出数据后验证 |
| **四、常用采集场景** | ⚠️ 未测试 | 参数组合未逐一验证 |

> **已知问题**：Open3D RealSense 采集帧率较低（约 1~5 Hz），可能无法达到 10 Hz 目标帧率。后续可考虑换用 pyrealsense2 原生采集或异步读取优化。
