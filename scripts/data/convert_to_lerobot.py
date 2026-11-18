#!/usr/bin/env python3
"""Interactively convert an xArm Zarr dataset to a LeRobot dataset.

Usage: python scripts/data/convert_to_lerobot.py --input DATA.zarr --output OUTPUT
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

# 保留 LeRobot 新旧版本的数据集导入路径。
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# Zarr 源字段与 LeRobot 字段的映射，组合字段作为一个选项。
FIELD_MAPPING = {
    ("rgb_fix",): "observation.image",
    ("rgb_arm",): "observation.wrist_image",
    ("pos", "gripper_state"): "observation.state",
    ("action", "gripper_action"): "action",
    ("force",): "observation.force",
    ("coinft_left",): "observation.coinft_left",
    ("coinft_right",): "observation.coinft_right",
}

# 图像 shape 在创建数据集时从各自的 Zarr 字段读取。
FEATURES = {
    "observation.image": {"dtype": "image", "shape": (), "names": ["height", "width", "channel"]},
    "observation.wrist_image": {"dtype": "image", "shape": (), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    "observation.force": {"dtype": "float32", "shape": (6,), "names": ["fx", "fy", "fz", "mx", "my", "mz"]},
    "observation.coinft_left": {"dtype": "float32", "shape": (6,), "names": ["fx", "fy", "fz", "mx", "my", "mz"]},
    "observation.coinft_right": {"dtype": "float32", "shape": (6,), "names": ["fx", "fy", "fz", "mx", "my", "mz"]},
}

@dataclass
class ConversionConfig:
    """交互阶段收集、最终确认后用于创建和写入数据集的配置。"""

    repo_id: str  # LeRobot 数据集标识，传给 LeRobotDataset.create。
    fps: int  # 输出数据集的帧率，用于生成时间信息；交互默认 30。
    robot_type: str  # 写入数据集元数据的机器人类型；交互默认 xarm6。
    task: str  # 每一帧关联的任务描述，由 LeRobot 保存任务信息。
    ranges: list[tuple[int, int]]  # 要转换的各轮帧区间，每项为 [start, end)；交互默认全部轮次。
    fields: list[tuple[str, ...]]  # 选中的 Zarr 源字段组合；默认空选择，需重新选择有效字段。
    overwrite: bool  # 是否覆盖已有输出；默认不覆盖，确认转换后才删除目录。


def print_summary(title, lines):
    """用上下横线打印概览或转换结果。"""
    print("\n" + "=" * 60)
    print(title)
    for line in lines:
        print(f"  {line}")
    print("=" * 60)


def choose(prompt, options, default=None):
    """显示数字选项并返回编号；default 为回车时使用的编号。"""
    for number, label in options.items():
        print(f"  {number}. {label}")
    suffix = f" [默认 {default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    if not answer and default is not None:
        answer = str(default)
    if answer not in options:
        raise ValueError("无效选项: " + answer)
    return answer


def input_text(prompt, default=None):
    """读取文本，允许使用明确给出的默认值。"""
    suffix = f" [默认 {default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    if not value:
        raise ValueError(f"{prompt}不能为空")
    return value


def input_positive_int(prompt, default=None):
    """读取正整数，用于 fps 和 episode 数量。"""
    value = int(input_text(prompt, str(default) if default is not None else None))
    if value <= 0:
        raise ValueError(f"{prompt}必须大于 0")
    return value


def episode_ranges(meta):
    """从 episode_ends 生成每轮的 [start, end) 帧区间。"""
    # episode_ends 保存每轮的结束位置，转换为左闭右开区间。
    ends = meta["episode_ends"][:]
    ranges = []
    start = 0
    for end in ends:
        end = int(end)
        ranges.append((start, end))
        start = end
    return ranges


def print_group(group, path, skip_episode_config=False):
    """递归展示组结构和数组形状；可跳过逐轮配置详情。"""
    for name in sorted(group.keys()):
        item = group[name]
        item_path = f"{path}/{name}"
        if hasattr(item, "shape"):
            shape = tuple(item.shape)
            frame_shape = shape[1:] if path == "data" and shape else "-"
            print(f"  {item_path}: shape={shape}, 单帧 shape={frame_shape}, dtype={item.dtype}")
        else:
            print(f"  {item_path}/")
            if skip_episode_config and item_path == "meta/episode_config":
                continue
            if item.attrs:
                print(f"    属性: {dict(item.attrs)}")
            print_group(item, item_path, skip_episode_config)


def show_dataset(root, data, meta, ranges, input_path):
    """展示 Zarr 全局属性、轮次和字段结构。"""
    lines = [f"输入: {input_path}", f"总帧数: {data['episode'].shape[0]}", f"Episodes: {len(ranges)}"]
    for label, group in (("根", root), ("data", data), ("meta", meta)):
        lines.append(f"{label} 属性: {dict(group.attrs) if group.attrs else '未记录'}")
    lines.extend(f"Episode {i}: {end - start} 帧 [{start}, {end})" for i, (start, end) in enumerate(ranges))
    print_summary("数据集概览", lines)
    print("字段和元数据结构:")
    print_group(data, "data")
    print_group(meta, "meta", skip_episode_config=True)


def inspect_episode_meta(meta, ranges):
    """按用户选择展示指定轮次的采集配置属性。"""
    while choose("查看 episode meta？", {"1": "查看", "2": "继续"}, "2") == "1":
        index = int(input_text("episode 编号（从 0 开始）"))
        if index < 0 or index >= len(ranges):
            raise ValueError(f"episode 编号不存在: {index}")
        name = f"episode_{index:06d}"
        if "episode_config" not in meta or name not in meta["episode_config"]:
            print(f"Episode {index} 没有记录 meta")
            continue
        attrs = dict(meta["episode_config"][name].attrs)
        print_summary(f"Episode {index} meta", [f"{key}: {value}" for key, value in attrs.items()] or ["无属性"])


def select_fields(data):
    """列出字段映射，接收多个编号并返回选中的源字段组合。"""
    # episode 是分轮索引，其余字段都列入选择，未知映射单独标记。
    names = set(data.keys()) - {"episode"}
    # entries 保存菜单中的源字段组合及其目标名；covered 标记已纳入组合的字段。
    entries = []
    covered = set()
    for sources, target in FIELD_MAPPING.items():
        present = names.intersection(sources)
        if present:
            entries.append((sources, target))
            covered.update(present)
    for name in sorted(names - covered):
        entries.append(((name,), None))
    # 选择错误只在此处重选；其他交互错误由入口报告并退出。
    while True:
        print("选择转换字段（一次输入多个编号，空格分隔）:")
        for i, (sources, target) in enumerate(entries, 1):
            target = target if target else "无映射"
            print(f"  {i}. {' + '.join(sources)} -> {target}")
        answer = input("字段编号 [默认 空]: ").strip()
        if not answer:
            print("字段选择为空，请重新选择。")
            continue
        try:
            numbers = [int(value) for value in answer.split()]
        except ValueError:
            print("字段选择不合理，请重新选择。")
            continue
        if len(numbers) != len(set(numbers)) or any(n < 1 or n > len(entries) for n in numbers):
            print("字段选择不合理，请重新选择。")
            continue
        selected = [entries[n - 1] for n in numbers]
        unmapped = [sources[0] for sources, rule in selected if rule is None]
        if unmapped:
            print(f"字段没有映射关系: {', '.join(unmapped)}，请自行补充规则，然后重选。")
            continue
        if any(not set(sources).issubset(names) for sources, _ in selected):
            print("字段选择不合理，请重新选择。")
            continue
        return [sources for sources, _ in selected]


def configure(data, ranges, output_path):
    """收集转换参数；选择退出时返回 None。"""
    repo_id = input_text("repo ID")
    fps = input_positive_int("fps", 30)
    robot_type = input_text("robot type", "xarm6")
    task = input_text("task")
    mode = choose("选择转换轮次", {"1": "全部", "2": "前 N 轮"}, "1")
    # ranges 为全部轮次的帧区间；按用户选择截取前 N 轮。
    selected_ranges = ranges
    if mode == "2":
        count = input_positive_int("episode 数量")
        if count > len(ranges):
            raise ValueError(f"episode 数量超过数据集总数 {len(ranges)}")
        selected_ranges = ranges[:count]
    fields = select_fields(data)
    overwrite = False
    if output_path.exists():
        overwrite = choose("输出路径已存在", {"1": "覆盖", "2": "退出"}, "2") == "1"
        if not overwrite:
            return None
    return ConversionConfig(repo_id, fps, robot_type, task, selected_ranges, fields, overwrite)


def confirm_config(config, data, input_path, output_path):
    """展示完整配置，并由用户确认是否开始。"""
    sources = {name for fields in config.fields for name in fields}
    # episode 是分轮索引，不作为未转换的普通字段。
    unused = sorted(set(data.keys()) - sources - {"episode"})
    lines = [f"输入: {input_path}", f"输出: {output_path}", f"repo ID: {config.repo_id}",
             f"fps: {config.fps}", f"robot type: {config.robot_type}", f"task: {config.task}",
             f"Episodes: {len(config.ranges)}, 帧数: {sum(end - start for start, end in config.ranges)}",
             f"覆盖已有输出: {'是' if config.overwrite else '否'}", "字段映射:"]
    lines.extend(f"  {' + '.join(fields)} -> {FIELD_MAPPING[fields]}" for fields in config.fields)
    lines.append(f"未转换的数据字段: {', '.join(unused) if unused else '无'}")
    print_summary("转换配置", lines)
    return choose("开始转换？", {"1": "开始", "2": "取消"}, "2") == "1"


def create_features(data, fields):
    """根据选中字段构建 LeRobot features，图像形状取自对应源数组。"""
    features = {}
    for sources in fields:
        target = FIELD_MAPPING[sources]
        # 复制模板，避免修改全局 FEATURES 中的图像形状占位值。
        feature = FEATURES[target].copy()
        if feature["dtype"] == "image":
            _, channels, height, width = data[sources[0]].shape
            feature["shape"] = (height, width, channels)
        features[target] = feature
    return features


def read_episode(data, fields, start, end):
    """按选中字段批量读取一轮数据，返回源字段到数组的字典。"""
    return {name: data[name][start:end] for sources in fields for name in sources}


def make_frame(batch, fields, index, task):
    """将本轮指定帧组装为 LeRobot 所需的字段和任务。"""
    # 图像由 CHW 转为 HWC，组合字段按映射顺序拼接。
    frame = {"task": task}
    for sources in fields:
        target = FIELD_MAPPING[sources]
        values = [batch[name][index] for name in sources]
        if FEATURES[target]["dtype"] == "image":
            frame[target] = Image.fromarray(values[0].transpose(1, 2, 0))
        elif len(values) > 1:
            frame[target] = np.concatenate([value.astype(np.float32) for value in values])
        else:
            frame[target] = values[0].astype(np.float32)
    return frame


def convert(data, output_path, config):
    """创建目标数据集，按轮写入并完成元数据保存。"""
    # 此函数仅在用户确认后调用；覆盖也在此时执行。
    features = create_features(data, config.fields)
    if config.overwrite:
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(repo_id=config.repo_id, fps=config.fps, robot_type=config.robot_type,
                                    features=features, root=output_path, use_videos=False,
                                    image_writer_threads=4)
    started = time.time()
    # 每次批量读取一轮，再逐帧写入并保存。
    for ep_index, (start, end) in enumerate(config.ranges):
        print(f"Episode {ep_index}: frames [{start}, {end}) = {end - start} steps")
        batch = read_episode(data, config.fields, start, end)
        for index in range(end - start):
            dataset.add_frame(make_frame(batch, config.fields, index, config.task))
        dataset.save_episode()
    # 完成写入并写出数据集元数据。
    dataset.finalize()
    frames = sum(end - start for start, end in config.ranges)
    print_summary("转换完成", [f"输出: {output_path}", f"Episodes: {len(config.ranges)}, Frames: {frames}",
                                f"耗时: {time.time() - started:.1f}s"])


def main():
    """解析路径，依次执行概览、交互配置和转换。"""
    parser = argparse.ArgumentParser(description="交互式 Zarr → LeRobot 数据集转换")
    parser.add_argument("--input", "-i", required=True, help="输入 Zarr 数据集路径")
    parser.add_argument("--output", "-o", required=True, help="LeRobot 输出目录")
    args = parser.parse_args()
    try:
        input_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve()
        if output_path == input_path or output_path in input_path.parents or input_path in output_path.parents:
            raise ValueError("输入和输出路径不能相同或相互包含")
        # 只读打开源数据；输出路径不得覆盖或包含源数据。
        root = zarr.open(str(input_path), mode="r")
        data = root["data"]
        meta = root["meta"]
        ranges = episode_ranges(meta)
        show_dataset(root, data, meta, ranges, input_path)
        inspect_episode_meta(meta, ranges)
        config = configure(data, ranges, output_path)
        if config is not None and confirm_config(config, data, input_path, output_path):
            convert(data, output_path, config)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
