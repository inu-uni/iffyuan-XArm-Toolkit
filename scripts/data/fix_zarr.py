#!/usr/bin/env python3
"""修复破损的 Zarr 数据集.

自动重建缺失的 meta/episode_ends，并检测丢弃最后一个不完整的 episode。

用法::

    python scripts/data/fix_zarr.py --input datasets/duck.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr


def fix(zarr_path: str):
    """修复 Zarr 数据集: 重建 episode_ends，自动丢弃不完整的末尾 episode."""
    ds = zarr.open(str(zarr_path), "a")
    data = ds["data"]

    if "meta" not in ds:
        meta = ds.create_group("meta")
    else:
        meta = ds["meta"]

    all_ep = data["episode"][:]
    unique_eps = np.unique(all_ep)
    total_frames = len(all_ep)

    # 各 episode 的帧数和结束位置
    ep_ends = []
    ep_lengths = []
    running = 0
    for ep in unique_eps:
        count = int(np.sum(all_ep == ep))
        running += count
        ep_ends.append(running)
        ep_lengths.append(count)

    print(f"[INFO] 发现 {len(unique_eps)} 个 episode, 总帧数 {total_frames}")
    for i, ep in enumerate(unique_eps):
        start = ep_ends[i - 1] if i > 0 else 0
        print(f"  Episode {ep}: frames [{start}, {ep_ends[i]}) = {ep_lengths[i]} steps")

    # 自动检测不完整 episode: 最后一个帧数 < 平均值 30%
    if len(ep_lengths) >= 2:
        avg_len = np.mean(ep_lengths[:-1])
        last_len = ep_lengths[-1]
        if last_len < avg_len * 0.3:
            keep = len(unique_eps) - 1
            cutoff = ep_ends[keep - 1]
            print(f"\n[WARN] 最后一个 episode ({unique_eps[-1]}) 疑似不完整: "
                  f"{last_len} 帧 (平均 {avg_len:.0f})")
            print(f"[INFO] 截断: 保留前 {keep} 个 episode ({cutoff} 帧), "
                  f"丢弃 {total_frames - cutoff} 帧")

            for key in data.keys():
                arr = data[key]
                truncated = arr[:cutoff]
                del data[key]
                data.create_dataset(key, data=truncated, chunks=arr.chunks,
                                    compressor=arr.compressor, dtype=arr.dtype)

            ep_ends = ep_ends[:keep]
            total_frames = cutoff

    # 写入 episode_ends
    if "episode_ends" in meta:
        del meta["episode_ends"]
    meta.create_dataset("episode_ends", data=np.array(ep_ends, dtype=np.uint32))

    print(f"\n[DONE] 修复完成!")
    print(f"  Episodes: {len(ep_ends)}")
    print(f"  Frames: {total_frames}")
    print(f"  episode_ends: {ep_ends}")


def main():
    parser = argparse.ArgumentParser(description="修复破损的 Zarr 数据集")
    parser.add_argument("--input", "-i", required=True, help="Zarr 数据集路径")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] 路径不存在: {args.input}")
        sys.exit(1)

    fix(args.input)


if __name__ == "__main__":
    main()
