#!/usr/bin/env python3
"""上传 LeRobot 数据集到 HuggingFace Hub.

将本地 LeRobot v3 数据集上传到 HuggingFace Hub，供团队共享或在线可视化。

用法::

    python scripts/data/upload_dataset.py \
        --path lerobot_datasets/act_pick2 \
        --repo IffYuan/xarm6_act_pick2 \
        --token hf_xxxxx

    # 也可以用环境变量传 token
    HF_TOKEN=hf_xxxxx python scripts/data/upload_dataset.py \
        --path lerobot_datasets/act_pick2 \
        --repo IffYuan/xarm6_act_pick2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="上传 LeRobot 数据集到 HuggingFace Hub")
    parser.add_argument("--path", "-p", required=True, help="本地数据集路径")
    parser.add_argument("--repo", "-r", required=True,
                        help="HuggingFace repo ID (如 iffyuan/xarm6_act_pick2)")
    parser.add_argument("--token", "-t", default=None,
                        help="HuggingFace token (也可通过 HF_TOKEN 环境变量传入)")
    parser.add_argument("--private", action="store_true", help="设为私有仓库 (默认公开)")

    args = parser.parse_args()

    dataset_path = Path(args.path).resolve()
    if not dataset_path.exists():
        print(f"[ERROR] 路径不存在: {dataset_path}")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] 请通过 --token 或 HF_TOKEN 环境变量提供 HuggingFace token")
        sys.exit(1)

    repo_id = args.repo
    private = args.private

    # ── 创建/获取仓库 ────────────────────────────────────
    api = HfApi(token=token)

    print(f"[INFO] 创建仓库: {repo_id} ({'私有' if private else '公开'})")
    create_repo(repo_id, repo_type="dataset", private=private,
                token=token, exist_ok=True)

    # ── 上传整个目录 ─────────────────────────────────────
    print(f"[INFO] 上传数据集: {dataset_path} → {repo_id}")
    print(f"[INFO] 可能需要较长时间，请耐心等待...\n")

    api.upload_folder(
        folder_path=str(dataset_path),
        repo_id=repo_id,
        repo_type="dataset",
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\n{'=' * 60}")
    print(f"  上传完成!")
    print(f"  仓库: {url}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
