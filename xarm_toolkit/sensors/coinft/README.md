# CoinFT 接收代码

此目录移植 CoinFT 项目 `teensy_mlp_workspace/host` 的接收实现，读取 Teensy 输出的 CFTF v2 双传感器六轴数据。

## 移植范围与来源

- 来源仓库：https://github.com/TiMoNavi/coinft
- 来源目录：`teensy_mlp_workspace/host/`
- 移植日期：2026-10-28。
- `protocol.py`、`coinft_interface.py` 和 `__init__.py` 从本地 CoinFT 工作副本逐字节复制，未修改源代码。
- xarm-toolkit 新增父包 `sensors/__init__.py` 和本说明文件；附带来源项目的完整 LICENSE。

运行时仅使用本包的相对导入，不需要原 CoinFT 项目目录。实际打开硬件串口需要安装 `pyserial`；当前移植未修改 xarm-toolkit 的依赖配置。

## 导入方式

```python
from xarm_toolkit.sensors.coinft.coinft_interface import (
    CalibrationFailedError,
    CoinFTInterface,
    CoinFTSample,
    ZeroCheckFailedError,
)
```

接口和默认行为沿用源实现：

- `CoinFTInterface(port, baud=115200, ...)` 默认发送 `i`、`s`，启动传感器流并触发 Teensy 板端归零。
- `wait_until_ready(timeout=60.0)` 等待有效六轴样本；归零期间两侧保持空载、静止。
- `read_sample(timeout=1.0)` 按接收顺序返回样本，不是后台维护的最新值接口。
- `restart_calibration()` 重新发送 `i`、`s` 并清理接收状态。
- `close()` 或 `with` 退出负责释放串口。
- `start_stream=False` 沿用源实现，仅用于接入已运行的流；该模式下正常关闭不会发送停止命令。

左右输出均为 `[Fx, Fy, Fz, Mx, My, Mz]`，前三项单位 N，后三项单位 Nm。以 `wrench_valid` 判断载荷有效性；`zero_check_warning` 为独立警告。

底层接收文件保持原样。现已增加独立 `CoinFTEnv`、后台接收和可选 Collector 接入，使用方式见下方对应章节；GUI 集成尚未实现。

## protocol.py：帧格式与解析机制

[protocol.py](protocol.py) 负责把 Teensy 发来的 CFTF v2 二进制字节还原成经过校验的左右六维力数据，并整理状态、时间和缺帧信息。本文件只依赖 Python 标准库；串口打开和读取由 [coinft_interface.py](coinft_interface.py) 完成，MLP、归零、偏差修正和滤波由当前 Teensy 固件完成。

### 整体处理流程

```text
串口得到任意长度的字节块
    ↓ StreamParser.feed()
拼接上次残留、寻找 CFTF 帧头、提取 65 字节候选帧
    ↓ decode_frame()
检查长度、帧头、版本、CRC、状态、flags 和数值有限性
    ↓ WrenchFrame
保存一帧的序号、设备时间、状态、左右六轴和原始字节
    ↓ sample_to_json()
生成上层可进一步序列化的普通字典
```

| 对象或函数 | 行为 |
| --- | --- |
| `WrenchFrame` | 冻结的数据类，保存完整解码结果；字段不能直接重新赋值，构造本身不自动校验类型或协议语义。 |
| `crc16_ccitt_false(data)` | 计算正文的 16 位 CRC 整数。 |
| `encode_frame(...)` | 从字段构造 65 字节帧，适合生成测试或回放数据。 |
| `decode_frame(raw)` | 校验一帧并返回 `WrenchFrame`；协议检查失败时抛出 `ProtocolError`。 |
| `StreamParser.feed(data)` | 增量消费字节流，返回本次解析出的帧列表，可能为零帧、一帧或多帧。 |
| `sequence_gap(previous, current)` | 按相邻 uint32 序号估计中间缺失的输出帧数。 |
| `sample_to_json(...)` | 整理 Python 字典；不写文件，也不直接生成 JSON 字符串。 |

### 线路帧格式

帧固定为 **65 字节**。整数与浮点数均使用小端序，浮点载荷为 IEEE-754 float32。偏移从 0 开始。

| 字节偏移 | 字节数 | 字段 | 类型及含义 |
| --- | ---: | --- | --- |
| `0–3` | 4 | `magic` | 固定 ASCII `CFTF`，用于查找帧边界。 |
| `4` | 1 | `version` | uint8，当前必须为 `2`。 |
| `5` | 1 | `calibration_status` | uint8，校准状态编号。 |
| `6` | 1 | `flags` | uint8，有效性与零点警告标志。 |
| `7–10` | 4 | `sequence` | uint32，每次 Teensy 配对输出的序号，校准期间的状态帧也参与编号。 |
| `11–14` | 4 | `teensy_time_us` | uint32，Teensy 配对处理时的微秒计数。 |
| `15–38` | 24 | `left` | 左侧 6 × float32。 |
| `39–62` | 24 | `right` | 右侧 6 × float32。 |
| `63–64` | 2 | `crc16` | uint16，随帧传输的 CRC，小端序。 |

大小为 `4 + 1 + 1 + 1 + 4 + 4 + 24 + 24 + 2 = 65`，整体 struct 布局可写为 `<4sBBBII12fH`。

两侧输出的顺序与单位一致：

| 索引 | 轴名 | 含义 | 单位 |
| --- | --- | --- | --- |
| 0 | `Fx` / `fx` | 模型坐标系 X 方向的力 | N |
| 1 | `Fy` / `fy` | 模型坐标系 Y 方向的力 | N |
| 2 | `Fz` / `fz` | 模型坐标系 Z 方向的力 | N |
| 3 | `Mx` / `mx` | 绕模型坐标系 X 轴的力矩 | Nm |
| 4 | `My` / `my` | 绕模型坐标系 Y 轴的力矩 | Nm |
| 5 | `Mz` / `mz` | 绕模型坐标系 Z 轴的力矩 | Nm |

Python 字典使用小写轴名。模块保留模型给出的数值、左右槽位、符号和参考点，不做机器人 TCP 或基坐标系变换。CFTF v2 载荷不包含 12 路原始电容通道；`WrenchFrame.raw` 保存的是这 65 字节完整协议帧。

### CRC 与单帧校验

CRC 使用 CRC-16/CCITT-FALSE：多项式 `0x1021`，初值 `0xFFFF`，输入输出均不反射，最终异或 `0x0000`。

计算范围为字节 **4–62**，对应 Python 切片 `raw[4:63]`，共 59 字节；不包含帧头或 CRC 字段本身。已知检查向量 `crc16_ccitt_false(b"123456789") == 0x29B1`。

`decode_frame()` 按下列顺序检查，任何一步失败都会抛出 `ProtocolError`：

1. 长度必须恰好为 65 字节。
2. 帧头必须为 `CFTF`，版本必须为 2。
3. 随帧 CRC 必须等于按正文重新计算的 CRC。
4. 校准状态必须是已知编号，flags 不允许包含未知位。
5. 状态、有效位和警告位的组合必须符合下述约束。
6. 左右 12 个浮点值必须全部有限，不允许 NaN 或正负无穷。

合法的校准中帧、固件错误状态帧仍可以解码；是否用于业务或抛出校准异常由上层接口处理。模块不检查测量精度、量程、模型与硬件匹配或机械安装方向。CRC 也不能追溯检测已经进入 Teensy 的原始传感器数据错误。

编码器与解码器的检查范围并不完全相同：`encode_frame()` 检查每侧长度、有限数值和已知状态，但不检查全部状态/flags 组合。因此它能生成随后被 `decode_frame()` 拒绝的帧，例如 `calibration_status=6, wrench_valid=True`。整数参数编码时只保留低 32 位。

### 校准状态与有效性

| 编号 | 名称 | 含义 |
| --- | --- | --- |
| 0 | `warming_up` | 启动预热或最短等待阶段。 |
| 1 | `delayed` | 还未满足后续校准条件，继续等待。 |
| 2 | `provisional_tare` | 临时原始通道归零。 |
| 3 | `runtime_settling` | 运行稳定阶段，当前快速归零也使用此状态。 |
| 4 | `final_tare` | 最终原始通道归零。 |
| 5 | `bias_calibration` | 估计六轴模型输出的空载偏差。 |
| 6 | `zero_check` | 检查校正后的空载零点。 |
| 7 | `ready` | 校准就绪，可提供有效六轴载荷。 |
| 8 | `zero_check_failed` | 零点检查失败，保留兼容解析。 |
| 9 | `error` | 推理或校准错误。 |

状态映射只解释固件编号，不在 Python 中执行归零。固件可能在同一次处理调用中推进部分中间阶段，因此不能要求每次启动都观察到表中每个状态。

`flags` 的定义与约束：

- bit 0（`0x01`）为 `wrench_valid`。解码器要求它为 True 当且仅当状态为 `ready`。
- bit 1（`0x02`）为 `zero_check_warning`。只有有效六轴帧允许携带警告；warning 不使载荷失效。
- 其他位必须为 0，当前解码器拒绝未知标志位。

典型合法组合为：非 ready 状态配 `flags=0x00`，ready 配 `0x01`，ready 且有警告配 `0x03`。`ready + 0x00`、`非 ready + 0x01` 以及无效载荷附带 warning 都会被拒绝。

当前 Teensy 固件的 warning 对应启动零点检查窗口中，左右三轴力均值绝对值超出 `0.15 N` 的情况。这是固件一次零点检查的标志，不是 Python 对每个当前读数重新判断，也不代表运行期间持续进行漂移补偿。是否接受该次归零由上层决定。

`WrenchFrame.left/right` 在无效状态仍保存线路中的浮点值，可能是六个零占位值。`sample_to_json()` 则将无效载荷的四个字段 `left_wrench`、`right_wrench`、`left`、`right` 设为 `None`，避免将“尚无有效测量”解释成“测得零力”。

### 流式缓冲与坏帧恢复

串口的单次读取边界不等于协议帧边界。一个 65 字节帧可以分几次到达，一次读取也可能包含多帧。

`StreamParser` 跨 `feed()` 调用保留 `bytearray` 缓冲区，行为如下：

1. 追加新字节，再搜索第一次出现的 `CFTF`。
2. 找不到完整帧头时，最多保留末尾 3 字节，以便下一块补齐跨边界的帧头，其余字节计为丢弃。
3. 找到帧头时，删除并统计之前的杂字节；不足 65 字节则等待下一块数据。
4. 凑够后复制 65 字节为候选，并调用 `decode_frame()`。
5. 校验成功就返回该帧并消费 65 字节，继续处理后面的输入。
6. 校验失败只向前滑动 1 字节，再重新寻找帧头，避免整段删除时丢掉内部真正的下一帧。

解析器维护的是待解释字节；`coinft_interface.py` 的 `_pending` 则保存已经解码、尚未交给调用方的完整帧。`feed()` 不排除重复序号，也不替调用方累计或保存全部历史样本。

### 时间、序号与诊断指标

| 指标 | 含义与使用边界 |
| --- | --- |
| `sequence` / 输出字典的 `sequence_id` | Teensy USB 输出序号，uint32 回绕；只能用于这条输出流的缺口分析，无法直接统计没有原始序号的传感器 UART 丢帧。 |
| `teensy_time_us` | Teensy 配对处理前记录的微秒计数，约 71.6 分钟回绕；不是传感器电容采样时刻，也不是主机绝对时间。本模块不展开回绕。 |
| `host_receive_monotonic_ns` | 上层传入的主机单调时钟时间，单位 ns；本函数不读取时钟。现有接口在串口分块读取后记录时间，同一块中的多帧可能共享该值。 |
| `dropped_before` | 本帧相对上一帧的序号缺口，不是累计丢帧总数。 |
| `StreamParser.bad_frames` | 候选帧解码失败次数；载荷偶然出现帧头也可能形成失败候选，因此不能简单等同于线路真实损坏帧数量。 |
| `StreamParser.discarded_bytes` | 解析过程中明确丢弃的字节总数，包括前导杂字节和恢复时滑过的字节。 |
| `CoinFTInterface.parser_stats` | 按 `(bad_frames, discarded_bytes)` 返回上述两项解析诊断。 |
| `calibration_ready` | 根据状态名称是否为 ready 计算。 |
| `wrench_valid` | 固件 flags 的有效位，合法解码帧中与 ready 状态一致。 |
| `zero_check_warning` | 固件零点检查警告位，和六轴有效性分开表示。 |

缺口计算规则为：

```python
gap = (current - previous - 1) & 0xFFFFFFFF
return gap if gap < 1_000_000 else 0
```

第一帧 `previous=None` 返回 0。连续序号返回 0，`10 → 13` 返回 2，`0xFFFFFFFF → 0` 正常回绕返回 0。重复、回退或很大的缺口常会被一百万阈值过滤成 0，因此返回 0 不能单独证明数据没有重复或设备没有重启；本函数不单独报告这些情况。

设备时间和主机时间属于不同的计时域，不能直接相减作为端到端延迟。当前传感器原始帧没有硬件采样时间戳，模块也不提供左右硬件同步保证。

### 字典输出

`sample_to_json()` 返回 Python dict，包含以下内容：

- 标识：`schema="coinft-teensy-mlp-v2-sample"`、`wire_format="coinft-mlp-v2"`。
- 时序：`sequence_id`、`teensy_time_us`、`host_receive_monotonic_ns`、`dropped_before`。
- 状态：`calibration_status`、`calibration_status_id`、`calibration_ready`、`calibration_error`、`wrench_valid`、`zero_check_warning`、`warning`。
- 布局：`side_order=["left", "right"]` 和 `wrench_axes=["fx", "fy", "fz", "mx", "my", "mz"]`。
- 载荷：有序列表 `left_wrench/right_wrench`，以及按轴名索引的字典 `left/right`；无效时为 `None`。

`calibration_error` 只在 `zero_check_failed` 或 `error` 时生成说明文字，其他状态为 `None`；本函数不会因这些合法状态主动抛校准失败异常。`warning` 根据警告位生成说明文字。真正生成 JSON 字符串需由调用方进一步调用 `json.dumps()`。

### 最小协议示例

此例只演示编码、分块解析和字典转换，不需要硬件串口：

```python
from xarm_toolkit.sensors.coinft.protocol import (
    StreamParser,
    encode_frame,
    sample_to_json,
)

wire = encode_frame(
    sequence=100,
    teensy_time_us=500_000,
    left=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
    right=(4.0, 5.0, 6.0, 0.4, 0.5, 0.6),
)
parser = StreamParser()
assert parser.feed(wire[:10]) == []
frames = parser.feed(wire[10:])
row = sample_to_json(frames[0], host_receive_monotonic_ns=123_000_000, dropped_before=0)
assert row["calibration_status"] == "ready"
assert row["left"]["fz"] == 3.0
```

## coinft_interface.py：串口接收与样本接口

[coinft_interface.py](coinft_interface.py) 在 `protocol.py` 之上提供实际设备接收接口：打开 Teensy USB 串口、发送控制命令、持续调用时按顺序取回完整帧，并把它们转换成上层使用的 `CoinFTSample`。归零、推理、偏差修正和滤波仍由 Teensy 固件完成，Python 端只发起操作并接收结果。

### 整体行为与公开接口

```text
CoinFTInterface(port)
    ↓ 打开串口，默认发送 i、s
Teensy 开始输出校准状态帧与后续有效六轴帧
    ↓ read_sample()
读取串口字节块 → StreamParser.feed() → 完整帧进入 _pending
    ↓ 按顺序取出一帧，计算序号缺口
CoinFTSample.from_frame()
    ↓
上层得到左右六轴、状态、时间与本帧前缺口
```

| 对象或接口 | 作用 |
| --- | --- |
| `CoinFTSample` | 冻结的样本数据类，保存左右六轴、校准状态、时间和序号缺口。 |
| `CoinFTSample.from_frame(...)` | 将协议层 `WrenchFrame` 转为上层样本，无效六轴设为 `None`。 |
| `sample.as_dict()` | 将样本转为普通 Python 字典，包含有序六轴列表和按轴名索引的字典。 |
| `CoinFTInterface(...)` | 打开串口，建立解析状态，并根据参数选择是否发送启动命令。 |
| `read_sample(timeout=None)` | 返回下一份按接收顺序排队的样本，可能尚未校准就绪。 |
| `wait_until_ready(timeout=None)` | 消费校准帧，等待有效六轴；遇到明确失败状态则抛出校准异常。 |
| `restart_calibration(restart_delay=0.2)` | 停止并重新启动传感器流，清理旧接收状态，触发板端重新归零。 |
| `parser_stats` | 返回 `(bad_frames, discarded_bytes)`。 |
| `close()` | 按启动归属决定是否发送停止命令，并关闭串口。 |
| `with CoinFTInterface(...) as device` | 离开上下文时调用 `close()`。 |

### CoinFTSample 的字段与数据语义

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `sequence` | int | Teensy uint32 输出序号，沿用协议中的数值。 |
| `teensy_time_us` | int | Teensy 配对处理时的 uint32 微秒计数，本接口不展开回绕。 |
| `calibration_status` | str | 状态名称，例如 `warming_up`、`ready`、`error`。 |
| `calibration_status_id` | int | 状态编号，例如 ready 对应 7。 |
| `calibration_ready` | bool | 从状态是否为 ready 得到。 |
| `wrench_valid` | bool | 固件声明的六轴载荷有效位。 |
| `zero_check_warning` | bool | 固件零点检查警告，不自动使六轴无效。 |
| `left_wrench` | tuple 或 None | 左侧六轴 `[Fx,Fy,Fz,Mx,My,Mz]`，前三项 N，后三项 Nm。 |
| `right_wrench` | tuple 或 None | 右侧六轴，同样顺序和单位。 |
| `dropped_before` | int | 相邻返回样本之间估计缺失的 USB 输出帧数，不是累计值。 |
| `host_receive_monotonic_ns` | int | 主机读取到包含完整帧的数据块后记录的单调时间，单位 ns。 |

`CoinFTSample` 使用 `@dataclass(frozen=True)`，正常创建后字段不能直接重新赋值。左右六轴保持 Python 元组，尚未转换成 NumPy 数组，也没有追加坐标变换或平滑。

`from_frame()` 调用协议模块的 `sample_to_json()` 获取状态名称和就绪信息，再保留帧中的序号、设备时间和标志。若 `wrench_valid=False`，左右六轴明确设为 `None`，避免把固件占位零值当成有效零力。

`as_dict()` 额外提供 schema、wire_format、错误与警告说明、左右顺序和轴名，并把元组转成列表。例如 `row["left_wrench"][2]` 和 `row["left"]["fz"]` 表达同一个左侧 Fz。它返回字典，不直接序列化 JSON 或写入文件。

### 构造参数与依赖

```python
CoinFTInterface(
    port,
    baud=115200,
    *,
    start_stream=True,
    startup_delay=1.0,
    read_timeout=0.05,
    serial_factory=None,
)
```

| 参数 | 含义 |
| --- | --- |
| `port` | 必须明确指定的 Teensy USB 串口名，例如 Windows 的 `COM3`。 |
| `baud` | 传给串口库的波特率配置，默认 115200。 |
| `start_stream` | 默认 True，构造时发送 `i/s` 并发起新的板端归零。 |
| `startup_delay` | 打开串口后、发送命令前的等待秒数，默认 1.0；负值在等待时按 0 处理。 |
| `read_timeout` | 底层串口单次读取超时秒数，默认 0.05，不是采样周期或设备帧率。 |
| `serial_factory` | 可选串口构造函数，默认 `serial.Serial`；用于注入模拟串口等替代实现。 |

真实串口按 `serial_factory(port, baud, timeout=read_timeout, write_timeout=1.0)` 创建。

模块尝试导入 pyserial；缺少时将 `serial` 设为 `None`，因此仍可导入本模块。创建接口时若既无 pyserial 又未提供 `serial_factory`，才会抛出带安装提示的 `RuntimeError`。本次移植保持该行为，未修改项目依赖配置。

### 默认启动过程与 start_stream

默认构造过程：

1. 打开串口，创建解析器、待取队列和生命周期状态。
2. 等待 `startup_delay`，默认 1 秒。
3. 发送 `i`，停止传感器流。
4. 等待 0.2 秒，让停止操作有时间生效。
5. 调用 `reset_input_buffer()`，丢弃主机串口中已有的输入字节。
6. 发送 `s`，启动流并触发 Teensy 校准，设置 `_started=True`。
7. 构造函数返回，校准是否完成由之后的状态帧判断。

`_send()` 通过 `write()` 写入命令，再调用 `flush()` 等待串口发送排出；它不读取命令确认，也不保证操作已在固件中完成。实际进展由后续状态帧反映。

构造函数没有等待首次归零完成。正常使用时下一步调用 `wait_until_ready(timeout=60.0)`；归零期间保持左右空载、静止。当前固件首次上电归零通常约 28–30 秒，后续快速归零通常约 3 秒，实际等待可能因数据不稳定延长。

若构造阶段的等待或命令发送出错，接口会尝试关闭已打开的串口并重新抛出异常。

`start_stream=False` 时不执行上述等待、停止、清理输入和启动步骤，可用于附着到已运行的流；它本身也不会触发归零。若之后调用 `restart_calibration()`，则仍会启动一轮新校准并设置 `_started=True`。

### 内部状态与两层缓冲

| 内部成员 | 保存什么 |
| --- | --- |
| `_port` | 当前串口对象。 |
| `_parser` | 增量协议解析器及其半帧字节缓冲、解析统计。 |
| `_pending` | FIFO 队列，保存已经解析好的 `(WrenchFrame, 接收时间)`。 |
| `_previous_sequence` | 上一份已经交给调用方的样本序号，初始为 None。 |
| `_started` | 本接口是否执行过启动流操作，决定关闭时是否发送 `i`。 |
| `_closed` | 本接口是否已经正常完成关闭。 |

`_parser.buffer` 处理分包；`_pending` 处理一次解析得到多帧。比如串口一次读取到 3 帧，解析后全部排队，后续 3 次 `read_sample()` 分别返回它们。

### read_sample() 的接收过程

每次 `read_sample(timeout=...)` 按以下流程工作：

1. 已关闭则抛出 `RuntimeError`。
2. 若给出 timeout，使用 `time.monotonic()` 建立本次截止时间。
3. 队列已有帧时直接取队首；队列为空时进入串口读取循环。
4. 调用 `read(max(1, in_waiting))`：当前已有字节时尝试一次取走，没有字节时至少请求 1 字节。
5. 本次没读到字节就继续循环并检查截止时间；读到字节块后记录 `time.monotonic_ns()`。
6. 将数据交给 `StreamParser.feed()`，得到的完整帧附上该接收时间进入 `_pending`；不足一帧就继续等待。
7. 用 `popleft()` 取出队首帧，调用 `sequence_gap()` 计算它与上一返回帧的缺口，再更新上一序号。
8. 调用 `CoinFTSample.from_frame()`，返回样本。

此方法返回的样本可以处于预热、校准或错误状态；有效性需要查看 `wrench_valid`，固件错误状态需要检查 `calibration_status`。CRC 等协议错误由解析器统计并尝试恢复，不会因一个偶发坏候选直接抛给调用方。持续收不到合法完整帧时可能最终表现为读取超时。

没有指定 timeout 时可一直等待。若队列已有帧，即使 timeout 为 0 也会直接返回队首，不先检查截止时间；没有待取帧时 timeout 为 0 会立即超时。底层 `read()` 另有 `read_timeout`，截止条件在读取循环边界检查，因此不能把外层 timeout 当作严格实时返回保证。

同一次读取解析出的多帧共享该次 `host_receive_monotonic_ns`。一帧分块到达时，记录的是完成该帧所在读取块的时间。帧在 `_pending` 中等待后再返回时，这个时间不会改写成返回时刻。

### wait_until_ready() 与校准异常

该方法在同一个总截止时间内反复调用 `read_sample()`，处理规则为：

| 收到的状态或标志 | 行为 |
| --- | --- |
| 普通预热、稳定或校准状态 | 继续读下一帧。 |
| `zero_check_failed` | 抛出 `ZeroCheckFailedError`。 |
| `error` | 抛出 `CalibrationFailedError`。 |
| `wrench_valid=True` | 返回当前有效样本，包括带 warning 的有效样本。 |
| 总等待到期或读取超时 | 抛出 `TimeoutError`，具体提示取决于在哪一层超时。 |

`ZeroCheckFailedError` 继承 `CalibrationFailedError`，后者继承 `RuntimeError`。上层可以分别捕获，也可以捕获父类统一处理校准失败。串口异常等其他错误按源实现继续向上传播。

此方法不会主动再发送 `s` 或重新开始归零；它只是等待当前流的结果。连续收到预热帧不会重置总等待时限。默认 timeout=None 可以一直等待。

成功返回的第一份有效样本已经从接收队列消费，调用方可以直接使用；下一次读取继续取后续帧。warning 保留在返回样本中，由调用方决定是否接受该次归零。

### restart_calibration() 的清理与重新启动

接口保持打开时，可重新执行归零：

```python
device.restart_calibration()
ready_sample = device.wait_until_ready(timeout=60.0)
```

重启过程为：检查未关闭 → 发送 `i` → 等待 restart_delay → 清理串口输入 → 创建新 StreamParser → 清空 `_pending` → 将上一序号设为 None → 发送 `s` → 设置 `_started=True`。

三处缓存分别清理：主机串口中的旧字节、解析器中的半帧、完整帧队列中的旧样本。新解析器同时使 `bad_frames/discarded_bytes` 统计归零。上一序号清空后，本轮第一帧的 dropped_before 按 0 处理；这不会重置 Teensy 自身的输出序号。

`restart_calibration()` 只发起校准，不等待 ready。首次还是快速校准由固件决定。本方法没有接收线程锁或命令确认机制，不应与另一个线程中的读取或关闭并发调用。

### 诊断、关闭与上下文管理

`parser_stats` 直接返回当前解析器的 `(bad_frames, discarded_bytes)`，分别表示候选帧失败次数和丢弃字节总数。样本的 `dropped_before` 则来自序号，三者含义不同。

`close()` 的行为：

- 已经正常关闭过时直接返回。
- `_started=True` 时先尝试发送 `i`。
- 在 finally 中尝试关闭串口；正常关闭返回后设置 `_closed=True`。
- 使用 `start_stream=False` 且之后未启动流时，关闭只释放串口，不发送停止命令。
- 停止命令失败时仍会尝试关闭串口；异常不会被这个方法主动吞掉。

`__enter__()` 返回当前实例，`__exit__()` 调用 `close()`，支持 `with` 自动清理。上下文退出不主动抑制业务异常；正常关闭后再调用 `read_sample()` 或 `restart_calibration()` 会报接口已关闭。

### 当前接收方式与使用边界

这是同步、按接收顺序返回的接口。它没有后台接收线程、自动重连、数据年龄检查或线程安全锁。只有调用读取方法时才消费输入；上层读取显著慢于设备输出时，主机串口或完整帧队列可能积压，随后返回较旧的数据。

新增的 `CoinFTEnv` 在该同步接口之上实现持续后台读取、最新样本缓存、数据过期检测和串口操作协调。直接使用 `CoinFTInterface` 仍保持上述源代码行为。

### 最小硬件读取示例

```python
from xarm_toolkit.sensors.coinft.coinft_interface import CoinFTInterface

# COM3 替换为实际 Teensy USB 串口；先关闭其他占用该串口的程序。
with CoinFTInterface("COM3") as device:
    # 构造已经发起本轮归零；等待期间两侧保持空载、静止。
    first = device.wait_until_ready(timeout=60.0)
    print(first.left_wrench, first.right_wrench, first.zero_check_warning)

    for _ in range(10):
        sample = device.read_sample(timeout=1.0)
        if sample.wrench_valid:
            print(sample.sequence, sample.left_wrench, sample.right_wrench)
            # 按轴名访问：sample.as_dict()["left"]["fz"]
```

示例退出 with 时释放串口；默认模式下还会发送 `i` 停止流。程序需要运行环境已安装 pyserial，并且设备输出为当前 CFTF v2 协议。

## CoinFTEnv：后台接收与最新观测

环境实现位于 `xarm_toolkit/env/coinft_env.py`，通过 `from xarm_toolkit.env import CoinFTEnv` 或 `from xarm_toolkit import CoinFTEnv` 导入。

### 运行方式

```python
from xarm_toolkit.env import CoinFTEnv

env = CoinFTEnv(port="COM3", max_sample_age=0.1)
try:
    # 打开环境不会自动启动或归零；在空载静止时显式调用 reset。
    obs = env.reset(timeout=60.0)
    while True:
        obs = env.step()
        left = obs["left_wrench"]
        right = obs["right_wrench"]
        # 按自己的控制周期使用数据，不必按设备约 200 Hz 的频率调用。
finally:
    env.cleanup()
```

构造参数为 `port`、`baud=115200`、`max_sample_age=0.1` 和 `read_timeout=0.05`。后两项单位为秒，须为有限正数。实际连接需运行环境已安装 pyserial；本次未修改依赖配置。

### 生命周期和接口

| 接口 | 行为 |
| --- | --- |
| 构造函数 | 以 `start_stream=False` 打开底层接口并创建接收线程，等待首次 reset；不自动触发归零。 |
| `reset(timeout=60.0)` | 立即使旧观测失效，提交归零请求；后台发送 i/s、清理旧接收缓存，再等待本轮新鲜有效数据。 |
| `step(action=None)` | 立即返回当前最新有效样本的观测副本；不等待新帧，不比较两次调用的序号，允许重复返回同一帧。 |
| `get_status()` | 查询接收、校准、统计及错误；未 ready、发生故障或关闭后也可查询。 |
| `cleanup()` | 通知后台停止，由同一线程关闭串口并等待线程退出，可重复调用。 |

一条后台线程执行启动、归零、读取和关闭；主线程通过受锁保护的请求与缓存交互，避免归零清理与串口读取同时执行。每次归零后，需先观察校准中状态再接受 ready，防止把残留 ready 帧误当作新一轮归零结果。默认超时覆盖首次启动等待。

reset 的 timeout 只结束调用方的等待，不会自动中止板端校准；之后可查询状态，或再次 reset 重新发起。两侧须保持空载静止。固件零点警告随有效数据返回，不自动拒绝或重新归零。

### 输出观测

`reset()` 和 `step()` 返回同样结构：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `left_wrench` | NumPy float32，shape (6,) | 左侧六轴。 |
| `right_wrench` | NumPy float32，shape (6,) | 右侧六轴。 |
| `sequence` | int | 原输出序号，不做重复过滤。 |
| `teensy_time_us` | int | 原设备微秒计数。 |
| `host_receive_monotonic_ns` | int | 原主机接收时间。 |
| `calibration_status` | str | 校准状态，成功观测为 ready。 |
| `wrench_valid` | bool | 成功观测为 True。 |
| `zero_check_warning` | bool | 保留固件警告。 |

左右数组都是独立副本，外部修改不影响接收缓存。轴顺序与单位沿用底层接口，环境不重复推理、平滑或坐标变换。

### 读取速率和过期处理

后台持续调用原接口的 `read_sample()`，即使上层没有调用 step，也继续消费全部输入并覆盖最新缓存。环境不为上层排队保存历史帧。比如设备约 200 Hz、控制循环 20 Hz 时，后台仍按设备到达情况接收，控制循环每次只取当时最新样本。

允许重复取到同一帧，包括上层调用快于传感器输出时。只检查该样本的主机接收年龄是否超过 `max_sample_age`，默认 0.1 秒；超出即抛 `TimeoutError`。重复调用不会刷新原时间戳。这个阈值是接收健康检查，不能代表真实传感器到主机的总延迟，也不能单独识别字节此前在系统缓冲中的等待。

接收线程只解析和更新缓存，状态变化时记录日志，不逐帧打印或绘图。若后台本身跟不上设备，系统缓冲仍可能积压，因此可用接收统计辅助诊断；本实现不承诺硬实时或自动补偿积压。

### 错误与状态统计

- 尚未 reset、校准未完成或当前帧无效：`CoinFTNotReadyError`。
- 当前样本接收年龄超限：`TimeoutError`。
- reset 等待超时：`TimeoutError`。
- 固件 zero_check_failed / error：沿用 `ZeroCheckFailedError` / `CalibrationFailedError`，可在检查负载后再次 reset。
- 接口已关闭或后台串口读取失败：`CoinFTError`；后台失败会关闭串口，原始错误保留为异常原因。

`CoinFTError`、`CoinFTNotReadyError` 以及沿用的校准异常可从 `xarm_toolkit.env.coinft_env` 导入。环境没有自动重连；串口接收失败后应清理并重新创建设备对象。

`get_status()` 包含 `connected`、`streaming`、`calibrating`、`calibration_status`、`wrench_valid`、`zero_check_warning`、`sequence`、`sample_age`、`received_frames`、`receive_rate_hz`、`dropped_frames`、`bad_frames`、`discarded_bytes`、`last_error`。

计数从本轮 reset 开始，包含校准帧。receive_rate_hz 是基于主机接收时间估算的本轮平均帧率；样本不足或接收时间相同则为 None，集中排空缓冲时也可能暂时偏高。`get_status().wrench_valid` 还综合当前连接、校准和年龄条件，表示此刻是否可读取有效观测。

上层 step 跳过的中间样本属于主动降采样，不计作通信丢帧；`dropped_frames` 累计的是后台按原序号规则观察到的缺口。

### 独立检查与无硬件验证

在仓库根目录运行：

```powershell
python -m scripts.test_coinft --port COM3 --hz 10 --max-sample-age 0.1
```

COM3 替换为实际 Teensy 串口。hz 只控制终端显示频率，后台接收独立持续运行。开始后保持两侧空载，等待校准完成；Ctrl+C 关闭。

无硬件测试使用模拟串口，但经过真实 protocol.py 和 coinft_interface.py，覆盖后台连续接收、慢速上层、重复样本、过期、归零缓存失效、固件错误、CRC 恢复、断连与清理：

```powershell
python -B -m unittest discover -s tests -p test_coinft_env.py -v
```

## Collector：可选保存左右六轴

采集入口增加 `--coinft` 开关，默认关闭；启用时同时指定 Teensy USB 串口：

```powershell
python -m scripts.data.collect_data --dataset datasets/coinft.zarr --coinft --coinft-port COM3
```

`--force` 仍表示 xArm 自身的 FT 传感器，与 CoinFT 开关分开。

Collector 构造参数新增 `use_coinft=False` 和 `coinft_env=None`，设备由入口脚本按需创建并注入，结束或其他设备出错退出时由入口清理。接收和读取速率仍交给 CoinFTEnv 后台处理。

```python
collector = Collector(
    env=env,
    cam_arm=cam_arm,
    cam_fix=cam_fix,
    agent=agent,
    use_coinft=True,
    coinft_env=coinft_env,
    # 其他采集配置按任务设置。
)
collector.run()
```

生命周期和故障行为：

- `Collector.run()` 开始时只调用一次 CoinFT reset，成功读数用于初始化缓存；episode 切换和机械臂复位不再初始化 CoinFT。
- 串口连接失败，入口立即记日志并结束本次记录；启动校准失败，Collector 记日志并返回，不进入记录流程。
- 每个记录循环在追加 buffer 的集中代码块读取最新 CoinFT 值，与其他数组保存相同行数。
- 运行期读取失败，立即记错误日志，沿用最近一次成功的左右值并继续采集；不因 CoinFT 故障自动丢弃 episode、复位或结束。
- 有效值带 warning 时，立即记警告日志，仍保存该次新读数。持续故障/警告会在后续取样时继续输出日志。
- 用户按现有 Enter/Ctrl+C 和保存确认流程决定何时结束与是否保存。

Zarr 只增加两路数组，不新增 CoinFT 元数据或状态/时间数组：

| 字段 | 形状 | 类型 | 顺序 |
| --- | --- | --- | --- |
| `data/coinft_left` | (N, 6) | float32 | Fx、Fy、Fz、Mx、My、Mz |
| `data/coinft_right` | (N, 6) | float32 | Fx、Fy、Fz、Mx、My、Mz |

前三项 N，后三项 Nm。故障时保存的沿用值与普通值没有额外状态字段区分，相关故障情况通过日志查看。

续采要求开关与已有数据集字段一致，启用时两路 shape 和 dtype 也要匹配；不自动为已有历史记录补写数据。没有 CoinFT 的数据集继续关闭采集，已有两路字段的数据集继续启用采集。

采集接入测试使用模拟硬件和内存 Zarr API，验证控制流、两路数组内容和行数；不替代真实 Zarr 存储或硬件验证：

```powershell
python -B -m unittest discover -s tests -p test_collector_coinft.py -v
```

## 来源项目署名与许可

项目名：CoinFT。

Copyright © 2026 The Board of Trustees of the Leland Stanford Junior University.

作者：Hojung Choi, Tae Myung Huh, Julia Di, Seongheon Hong, Mark Cutkosky。

来源项目声明采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，完整文本见本目录 [LICENSE](LICENSE)。
