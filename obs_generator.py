"""内蒙古自治区多源融合预报 - 站点观测数据生成器
支持两种模式:
- 模式 A (默认): 基于合理的气象统计规律合成观测数据（适合无真实观测数据时）
- 模式 B: 从本地 CSV/文本文件导入真实观测数据
输出格式为 pipe 分隔的文本文件，符合 config.py 中 OBS 源的 obs_format 定义
"""

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

from config import INNER_MONGOLIA_CITIES, FORECAST_CONFIG


def wind_speed_to_grade(speed_mps: float) -> int:
    """根据蒲福风级阈值将风速(m/s)转为蒲福风级整数"""
    thresholds = FORECAST_CONFIG["wind_force_thresholds"]
    grade = 0
    for i, thr in enumerate(thresholds):
        if speed_mps < thr:
            grade = i
            break
    else:
        grade = len(thresholds)
    return grade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="内蒙古自治区12个盟市站点观测数据生成器",
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        required=True,
        help="日期 YYYYMMDD",
    )
    parser.add_argument(
        "--cycle", "-c",
        type=int,
        choices=[8, 20],
        default=8,
        help="起报时刻(北京时间)，8 或 20",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["synthetic", "real"],
        default="synthetic",
        help="数据来源模式: synthetic(合成) 或 real(从CSV导入)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子(仅合成模式使用)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/obs/",
        help="输出目录",
    )
    parser.add_argument(
        "--precip-mean", "-p",
        type=float,
        default=2.0,
        help="区域平均降水强度(mm)",
    )
    parser.add_argument(
        "--wind-mean", "-w",
        type=float,
        default=4.0,
        help="区域平均风速(m/s)",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default=None,
        help="模式 B 时指定的输入 CSV 路径",
    )
    return parser.parse_args()


def generate_synthetic(
    cities: List[Dict],
    seed: int,
    precip_mean: float,
    wind_mean: float,
) -> List[Dict]:
    """基于统计规律合成观测数据"""
    rng = np.random.default_rng(seed)
    records = []
    for city in cities:
        lon_perturb = (city["lon"] - 112.0) * 0.02
        lat_perturb = (city["lat"] - 42.0) * 0.02
        loc_perturb = lon_perturb + lat_perturb

        shape = max(0.1, precip_mean + loc_perturb)
        precip_12h = max(0.0, float(rng.gamma(shape=shape, scale=1.0)))

        ratio = float(rng.uniform(0.25, 0.5))
        precip_1h = max(0.0, precip_12h * ratio)

        wind_max_mps = max(0.5, float(rng.normal(loc=wind_mean, scale=1.5)))

        wind_max_grade = wind_speed_to_grade(wind_max_mps)

        records.append({
            "station_id": city["station_id"],
            "name": city["name_short"],
            "lon": city["lon"],
            "lat": city["lat"],
            "precip_12h": precip_12h,
            "precip_1h": precip_1h,
            "wind_max_mps": wind_max_mps,
            "wind_max_grade": wind_max_grade,
        })
    return records


def load_real_data(
    source_file: str,
    cities: List[Dict],
) -> List[Dict]:
    """从本地 CSV 文件导入真实观测数据"""
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"源文件不存在: {source_file}")

    city_map = {c["station_id"]: c for c in cities}
    records = []
    with open(source_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("station_id", "").strip()
            city = city_map.get(sid)
            if city is None:
                for c in cities:
                    if c["name_short"] == row.get("name", "").strip() or \
                       c["name"] == row.get("name", "").strip():
                        city = c
                        break
            if city is None:
                continue

            def _to_float(v, default=0.0):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return default

            wind_mps = _to_float(row.get("wind_max_mps"), 0.0)
            grade = _to_float(row.get("wind_max_grade"), None)
            if grade is None or grade <= 0:
                grade_int = wind_speed_to_grade(wind_mps)
            else:
                grade_int = int(grade)

            records.append({
                "station_id": city["station_id"],
                "name": city["name_short"],
                "lon": city["lon"],
                "lat": city["lat"],
                "precip_12h": _to_float(row.get("precip_12h"), 0.0),
                "precip_1h": _to_float(row.get("precip_1h"), 0.0),
                "wind_max_mps": wind_mps,
                "wind_max_grade": grade_int,
            })

    found_ids = {r["station_id"] for r in records}
    for city in cities:
        if city["station_id"] not in found_ids:
            records.append({
                "station_id": city["station_id"],
                "name": city["name_short"],
                "lon": city["lon"],
                "lat": city["lat"],
                "precip_12h": 0.0,
                "precip_1h": 0.0,
                "wind_max_mps": 0.5,
                "wind_max_grade": 0,
            })
    records.sort(key=lambda r: r["station_id"])
    return records


def format_value(v: float, is_int: bool = False) -> str:
    if is_int:
        return f"{int(v)}"
    if abs(v) >= 10.0:
        return f"{v:.1f}"
    return f"{v:.2f}"


def write_obs_file(
    output_path: str,
    records: List[Dict],
    date: str,
    cycle: int,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append(
        f"# 内蒙古12盟市观测数据 | 日期: {date} | 起报: {cycle:02d}时(BJT) | 时效: 12h"
    )
    lines.append(
        "# station_id|name|lon|lat|precip_12h|precip_1h|wind_max_mps|wind_max_grade"
    )

    for r in records:
        parts = [
            str(r["station_id"]),
            str(r["name"]),
            format_value(float(r["lon"])),
            format_value(float(r["lat"])),
            format_value(float(r["precip_12h"])),
            format_value(float(r["precip_1h"])),
            format_value(float(r["wind_max_mps"])),
            str(int(r["wind_max_grade"])),
        ]
        lines.append("|".join(parts))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(records: List[Dict], output_path: str) -> None:
    print(f"生成观测文件: {output_path}")

    precips = [r["precip_12h"] for r in records]
    winds = [r["wind_max_mps"] for r in records]

    def _stats(values):
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
        }

    p = _stats(precips)
    w = _stats(winds)
    print(f"降水统计 - 均值: {p['mean']:.2f} mm | 最大: {p['max']:.2f} mm | 最小: {p['min']:.2f} mm")
    print(f"风速统计 - 均值: {w['mean']:.2f} m/s | 最大: {w['max']:.2f} m/s | 最小: {w['min']:.2f} m/s")


def main() -> int:
    args = parse_args()

    if args.mode == "real":
        if not args.source_file:
            print("错误: 模式 real 需要指定 --source-file", file=sys.stderr)
            return 2
        records = load_real_data(args.source_file, INNER_MONGOLIA_CITIES)
    else:
        records = generate_synthetic(
            cities=INNER_MONGOLIA_CITIES,
            seed=args.seed,
            precip_mean=args.precip_mean,
            wind_mean=args.wind_mean,
        )

    output_path = os.path.join(args.output, args.date, f"OBS_{args.date}.txt")
    write_obs_file(output_path, records, args.date, args.cycle)
    print_summary(records, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
