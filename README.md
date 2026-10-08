# XArm Toolkit

ER Group XArm 6 机械臂工具包，包含真机控制、相机采集、遥操作、数据采集、VLA 部署。

## 硬件一览

| 组件 | 型号 | 关键参数 |
|------|------|---------|
| 机械臂 | XArm 6 (6-DOF) | IP: `192.168.31.232` |
| 夹爪 | 内置 | 行程 0（闭合）~ 840（全开） |
| 力传感器 | 内置 6 轴 FT | 支持阻抗控制 |
| 臂上相机 | Intel RealSense D435i | serial: `327122075644` |
| 固定相机 | Intel RealSense L515 | serial: `f1271506` |
| 遥操作 | 3DConnexion SpaceMouse | 6D 输入 + 左右按钮控制夹爪 |

## 详细文档

`docs/` 目录下有完整的使用文档，按主题分章：

| 文档 | 内容 |
|------|------|
| [01-硬件与环境配置](docs/01-硬件与环境配置.md) | XArm6 连接、RealSense 相机配置、SpaceMouse 设置、网络环境 |
| [02-机械臂控制](docs/02-机械臂控制.md) | XArmEnv 接口、三种控制模式、夹爪控制、力传感器 |
| [03-相机系统](docs/03-相机系统.md) | RealsenseEnv 接口、RGB/RGBD/点云模式、双相机配置 |
| [04-遥操作与数据采集](docs/04-遥操作与数据采集.md) | SpaceMouse 映射、Collector 采集流程、Zarr 数据格式、任务配置 |
| [05-模型部署](docs/05-模型部署.md) | VLAServer/Client 架构、Pi0.5 openpi 部署、推理流程 |
| [06-故障排除](docs/06-故障排除.md) | 常见错误、硬件排查、网络问题 |
| [07-LeRobot 集成采集](docs/07-LeRobot集成采集.md) | LeRobot 插件安装、`lerobot record` 采集、数据格式对比 |
| [4.5-ACT 训练全流程](docs/4.5-ACT训练全流程.md) | Zarr→LeRobot 转换、ACT 训练、推理部署、参数速查 |

**示例数据集:** [IffYuan/xarm6_act_pick2](https://huggingface.co/datasets/IffYuan/xarm6_act_pick2) — 30 episodes, 14643 frames, 15Hz, LeRobot v3 格式

**关于LeRobot详细说明：0.一般情况下，直接装本仓库内的LeRobot就可以了 1.本代码库维护了和LeRobot完全无关的简洁的遥操作和数据采集，采集数据格式为zarr格式，如果你想用LeRobot格式进行训练，直接在后面的数据采集部分有转换zarr到LeRobot 3.0 2.本repo内的lerobot强行加入了xarm机器人和3D mouse遥操作，可以遵循相应文档使用纯LeRobot原生的API跑起来，但是并不好用，调帧严重，仅供体验即可**

**我想跑一个模仿学习策略，我的流程是什么？**  
- 1、学习环境，相机的一些API，了解大体的硬件设备
- 2、基于遥操作与数据采集这个章节的内容，使用3D mouse进行数据采集，然后转换成LeRobotDataset格式
- 3、只用这个数据集，选择ACT或者pi0.5等模仿学习算法进行训练（训练docs待更新）
- 4、使用本repo提供的环境进行同步（Env里面塞policy）或者异步（server client）的部署

## 安装

```bash
git clone https://github.com/pickxiguapi/iffyuan-XArm-Toolkit && cd iffyuan-XArm-Toolkit
pip install -e .

# 开发依赖（pytest / ruff）
pip install -e ".[dev]"

# LeRobot（本地源码，已内置 XArm6 + SpaceMouse 插件）
cd lerobot && pip install -e . && cd ..
```

## ⚠️ 运动速度 (`speed`) 参数说明

`speed` 是机械臂笛卡尔空间运动速度（单位 mm/s），**直接决定机械臂移动快慢，务必注意安全**。

| 速度 (mm/s) | 级别 | 适用场景 |
|:-----------:|:----:|---------|
| **50~100** | 🐢 慢速 | 初次调试、精细操作、接近物体 |
| **200~400** | 🚶 中速 | **日常遥操作、数据采集（推荐默认 400）** |
| **500~800** | 🏃 快速 | 空间大范围移动、复位 |
| **1000+** | ⚡ 极速 | 仅限空旷环境，**新手慎用** |

各脚本 `--speed` 默认值均为 **400 mm/s**，可通过命令行覆盖。

## 真机验证流程

按以下顺序逐步验证，每一步通过了再往下走。

---

### Step 1: 验证相机

确认双 RealSense 相机能正常出图。

```bash
# 双相机预览（按 q 退出）
python scripts/test_cameras.py

# 只测臂上相机
python scripts/test_cameras.py --arm-only

# 只测固定相机
python scripts/test_cameras.py --fix-only
```

**预期:**
- 看到两个相机窗口，画面流畅
- 日志显示 shape、dtype、min/max 等信息
- FPS ≥ 25

---

### Step 2: 验证机械臂环境

逐步测试: 连接 → 读状态 → 复位 → 小幅移动 → 夹爪。每步暂停等你确认，确保安全。

```bash
# 完整测试（交互式，每步需确认）
python scripts/test_env.py

# 带力传感器
python scripts/test_env.py --force

```

**预期:**
- 笛卡尔位姿、关节角、夹爪位置正常打印
- 复位后到达 Home 位姿
- 小幅移动 X±10mm 后回到原位
- 夹爪开合正常

---

### Step 3: Web 键盘遥操作

网页界面 + 双相机画面 + 键盘控制机械臂。

```bash
python scripts/web_control_demo.py          # 真机模式 (默认 speed=400)
python scripts/web_control_demo.py --mock   # mock 模式调试 UI

# 调整速度
python scripts/web_control_demo.py --speed 100                  # 慢速精细操作
python scripts/web_control_demo.py --step-pos 2 --speed 600     # 大步快速移动
```

打开 `http://<机器IP>:8080`

![Web 键盘遥操作界面](assets/web_control_demo.jpg)

| 按键 | 功能 | 按键 | 功能 |
|------|------|------|------|
| W/S | Y 前/后 | I/K | Roll +/- |
| A/D | X 左/右 | J/L | Pitch +/- |
| Q/E | Z 上/下 | U/O | Yaw +/- |
| Space | 切换夹爪 | R | 复位 Home |
| Esc | 急停 | | |

---

### Step 4: SpaceMouse 遥操作

3D SpaceMouse 控制机械臂末端。

```bash
# 基础遥操作
python scripts/teleop_demo.py

# 启用力传感器
python scripts/teleop_demo.py --force

# 调整灵敏度
python scripts/teleop_demo.py --trans-scale 3 --rot-scale 0.003

# 自定义频率和速度
python scripts/teleop_demo.py --hz 50 --speed 400
```

**终端键盘:**
- `q` / `Ctrl+C` — 退出
- `r` — 复位
- `o` — 开夹爪
- `c` — 关夹爪
- SpaceMouse 左键 — 切换夹爪开/关

---

### Step 5: 数据采集

SpaceMouse 遥操作 + 双相机 + 力传感器 → Zarr 数据集。

```bash
# 默认采集 rgbd，3 个 episode
python scripts/collect_data.py --dataset datasets/demo.zarr

# 只采集 RGB
python scripts/collect_data.py --cam-mode rgb --dataset datasets/rgb_only.zarr

# 指定任务初始偏移 + episode 数量
python scripts/collect_data.py --start-bias 0 0 -200 --episodes 10 --dataset datasets/plug.zarr

# 带随机偏移（每个 episode 起始位置不同）
python scripts/collect_data.py --start-bias 0 0 -200 --random-bias-x -50 70 --random-bias-y -20 10 --dataset datasets/plug.zarr

# 夹爪始终闭合（stamp 类任务）
python scripts/collect_data.py --start-bias 0 0 -270 --gripper-closed --dataset datasets/stamp.zarr

# 启用力传感器
python scripts/collect_data.py --force --dataset datasets/force_demo.zarr

# 保存视频回看
python scripts/collect_data.py --save-video --dataset datasets/demo.zarr
```

| 操作 | 说明 |
|------|------|
| Space | 开始录制当前 episode |
| Enter | 结束当前 episode，并进入保存确认 |
| `y` / `Y` | 将当前 episode 写入 Zarr |
| `n` / `N` | 丢弃当前 Zarr buffer；已录制视频仍保留 |
| Ctrl+C | 中止采集 |

---

### Step 6: Zarr → LeRobot 数据转换

将采集的 Zarr 数据集转换为 [LeRobot v3.0](https://github.com/huggingface/lerobot) 格式，用于 VLA 模型训练（Pi0.5 等）。

**安装 LeRobot:**
```bash
cd lerobot && pip install -e . && cd ..
```

**转换:**
```bash
python scripts/convert_to_lerobot.py \
    --input datasets/demo.zarr \
    --output lerobot_datasets/demo \
    --repo-id demo --fps 15 \
    --task "pick up the red block"
```

**验证:**
```bash
python scripts/verify_lerobot.py --path lerobot_datasets/demo --repo-id demo
```

**LeRobot 字段映射:**

| LeRobot key | 来源 | shape |
|---|---|---|
| `observation.image` | rgb_fix（固定相机） | (H, W, 3) uint8 |
| `observation.wrist_image` | rgb_arm（臂上相机） | (H, W, 3) uint8 |
| `observation.state` | pos(6) + gripper_state(1) | (7,) float32 |
| `action` | action(6) + gripper_action(1) | (7,) float32 |

> 输出目录已存在时自动覆盖。`--task` 不指定则默认使用 zarr 文件名。

---

### Step 7: VLA 部署
## 部署还没有经过验证，待加入pi0.5进行验证

Server/Client 架构，GPU 机器跑推理，机器人端执行动作。

**GPU 机器（启动 Server）:**
```bash
pip install websockets msgpack
# 还需要 openpi 环境（torch + transformers 等）

# 自动下载 checkpoint
python scripts/start_server.py --config pi05_droid --port 10093

# 指定本地 checkpoint
python scripts/start_server.py --config pi05_droid --ckpt-dir /data/ckpts/pi05_droid
```

**机器人端（启动 Client）:**
```bash
python scripts/deploy_vla.py \
    --server-host <GPU_IP> \
    --server-port 10093 \
    --instruction "pick up the cup" \
    --max-steps 5
```

> ⚠️ **安全提醒:** 先用 `--max-steps 5` 小步测试，手放在急停按钮上。

---

## 核心概念速查

### 三种控制模式

| 模式 | `action_mode` | action 含义 | 典型场景 |
|------|---------------|------------|---------|
| 增量末端 | `"delta_eef"` | `[dx,dy,dz,dr,dp,dy]` 相对位移 | SpaceMouse 遥操作/VLA使用 |
| 绝对末端 | `"absolute_eef"` | `[x,y,z,r,p,y]` 目标位姿 | VLA 使用 |
| 绝对关节 | `"absolute_joint"` | `[j1..j6]` 目标关节角 (rad) | VLA 关节部署 |

### Observation 字典

`env.reset()` 和 `env.step()` 返回的 obs：

| 字段 | 类型 | 说明 |
|------|------|------|
| `cart_pos` | `np.ndarray (6,)` | 末端位姿 `[x,y,z,roll,pitch,yaw]`，xyz 单位 mm，角度 rad |
| `servo_angle` | `np.ndarray (6,)` | 6 个关节角 (rad) |
| `ext_force` | `np.ndarray (6,)` 或 `None` | 外部力/力矩（需 `use_force=True`） |
| `raw_force` | `np.ndarray (6,)` 或 `None` | 原始力传感器读数 |
| `goal_pos` | `np.ndarray (6,)` | 当前目标位姿 |
| `gripper_position` | `np.ndarray` | 夹爪位置（0=闭合，840=全开） |

### SpaceMouse 灵敏度参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `translation_scale` | 5.0 | 平移灵敏度（mm/tick）|
| `rotation_scale` | 0.004 | 旋转灵敏度（rad/tick）|
| `deadzone` | 0.0 | 死区，过滤手抖（建议 0.05~0.15）|

### Zarr 数据集格式

```
dataset.zarr/
├── data/
│   ├── rgb_arm        # (N, 3, 240, 320) uint8   — 臂上相机 RGB
│   ├── rgb_fix        # (N, 3, 240, 320) uint8   — 固定相机 RGB
│   ├── depth_arm      # (N, 1, 240, 320) uint16  — 臂上相机 depth (rgbd/pcd 模式)
│   ├── depth_fix      # (N, 1, 240, 320) uint16  — 固定相机 depth (rgbd/pcd 模式)
│   ├── pos            # (N, 6) float32 — 末端位姿
│   ├── force          # (N, 6) float32 — 力传感器
│   ├── action         # (N, 6) float32 — SpaceMouse 动作
│   ├── gripper_state  # (N, 1) float32 — 夹爪状态 (0/1)
│   ├── gripper_action # (N, 1) float32 — 夹爪目标动作 (0/1)
│   └── episode        # (N,) uint16 — episode 编号
└── meta/
    └── episode_ends   # (M,) uint32 — 各 episode 的结束索引
```

> `--cam-mode rgb` 时不创建 `depth_arm` / `depth_fix`。

## 项目结构

```
iffyuan-XArm-Toolkit/
├── xarm_toolkit/           # 主 Python 包
│   ├── env/
│   │   ├── xarm_env.py         # 统一环境接口（力控/位控可选）
│   │   └── realsense_env.py    # RealSense D435i + L515 封装
│   ├── teleop/
│   │   ├── spacemouse_expert.py # SERL SpaceMouseExpert（原文件，勿改）
│   │   └── spacemouse.py       # SpacemouseAgent — 缩放/重映射/夹爪切换
│   ├── collect/
│   │   └── collector.py        # Collector — env + cameras + teleop → Zarr
│   ├── deploy/
│   │   ├── msgpack_numpy.py    # ndarray ↔ msgpack 序列化
│   │   ├── server.py           # VLAServer — WebSocket 推理服务
│   │   └── client.py           # VLAClient — 同步客户端
│   └── utils/
│       └── logger.py           # 统一日志
├── scripts/
│   ├── test_cameras.py         # 相机验证
│   ├── test_env.py             # 机械臂验证
│   ├── web_control_demo.py     # Web 键盘遥操作
│   ├── teleop_demo.py          # SpaceMouse 遥操作
│   ├── collect_data.py         # 数据采集入口
│   ├── convert_to_lerobot.py   # Zarr → LeRobot 数据转换
│   ├── verify_lerobot.py       # 数据集验证（Zarr / LeRobot）
│   ├── deploy_vla.py           # VLA 机器人端
│   └── start_server.py         # VLA GPU 端
├── lerobot/                # LeRobot 源码（已内置 XArm6 + SpaceMouse 插件）
│   └── src/lerobot/
│       ├── robots/xarm6/           # XArm6 Robot 适配器
│       └── teleoperators/spacemouse_xarm6/  # SpaceMouse 遥操作适配器
├── xarm/                   # xarm-python-sdk（勿改）
├── tests/                  # 测试（mock 硬件）
├── docs/                   # 详细文档
├── pyproject.toml
└── requirements.txt
```
