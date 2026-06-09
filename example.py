# -*- coding: utf-8 -*-
"""
多源融合预报程序 - 使用示例 / 测试脚本
演示如何直接调用融合模块进行预报，以及如何自定义数据源。
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# ============================================================================
# 示例1：直接运行主程序（默认方式）
# ============================================================================
def example_1_basic_usage():
    """
    示例1：最基本的使用方式 —— 直接运行主程序
    此方式会使用 config.py 中的所有默认配置。
    """
    print("\n" + "=" * 60)
    print("【示例1】基本用法：直接运行主程序")
    print("=" * 60)

    print("""
    # 方式一：默认参数（自动使用当天日期 + 08和20时起报）
    python main.py

    # 方式二：指定起报日期
    python main.py --date 20250101

    # 方式三：指定起报日期和起报时刻
    python main.py --date 20250101 --cycles 8 20

    # 方式四：选择融合方法
    python main.py --method ensemble_median

    # 方式五：查看帮助
    python main.py --help
    """)


# ============================================================================
# 示例2：通过Python API调用融合模块（编程方式）
# ============================================================================
def example_2_python_api():
    """
    示例2：通过Python API直接调用融合算法
    适合需要将融合功能嵌入到其他程序中的场景。
    """
    print("\n" + "=" * 60)
    print("【示例2】Python API调用：加权平均融合")
    print("=" * 60)

    from fusion import (
        fuse_values,
        FUSION_METHODS,
        wind_speed_to_force_level,
        compute_fusion_statistics,
    )
    from config import FORECAST_CONFIG

    # 假设以下是5个模式对呼和浩特市12小时累计降水的预报
    source_values = [12.5, 18.3, 15.0, 9.8, 20.1]  # mm
    source_names = ["EC", "GRAPES", "GRAPES_MESO", "NCEP", "OBS"]
    source_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    total_w = sum(source_weights)
    source_weights = [w / total_w for w in source_weights]

    print("各数据源12小时累计降水预报 (mm):")
    for n, v in zip(source_names, source_values):
        print(f"  {n:>15}: {v:.2f}")

    print("\n不同融合方法的结果:")
    for method in FUSION_METHODS.keys():
        result = fuse_values(
            method=method,
            values=source_values,
            weights=source_weights,
            source_names=source_names,
        )
        print(f"  {method:>20}: {result:.2f} mm")

    # 统计信息
    stats = compute_fusion_statistics(source_values, source_names,
                                      source_weights)
    print(f"\n统计量:")
    print(f"  最小值: {stats['min']:.2f}, 最大值: {stats['max']:.2f}")
    print(f"  均值: {stats['mean']:.2f}, 中位数: {stats['median']:.2f}")
    print(f"  标准差: {stats['std']:.2f}, 极差: {stats['range']:.2f}")

    # 风力等级转换示例
    print("\n风力等级转换示例:")
    for ws in [0.5, 3.0, 8.0, 15.0, 25.0, 40.0]:
        level = wind_speed_to_force_level(
            ws, FORECAST_CONFIG["wind_force_thresholds"]
        )
        print(f"  风速 {ws:6.2f} m/s -> {level} 级")


# ============================================================================
# 示例3：自定义数据源（使用字典读取器）
# ============================================================================
def example_3_custom_sources():
    """
    示例3：直接传入内存中的数据进行融合
    当已有各模式的预报结果（例如从其他系统解析得到），
    可以直接传入字典进行融合，而不需要从文件读取。
    """
    print("\n" + "=" * 60)
    print("【示例3】自定义数据源：直接传入字典数据")
    print("=" * 60)

    from config import INNER_MONGOLIA_CITIES, FORECAST_CONFIG, DATA_SOURCES
    from data_reader import create_reader
    from fusion import fuse_values, compute_weights_by_name

    # 模拟4个数据源的预报结果（实际使用时替换为真实数据）
    custom_data = {
        "EC": {
            "precip_12h": {"呼和浩特": 15.2, "包头": 12.1, "乌海": 5.8,
                           "赤峰": 20.5, "通辽": 18.9, "鄂尔多斯": 8.3,
                           "呼伦贝尔": 3.2, "巴彦淖尔": 9.5, "乌兰察布": 11.2,
                           "兴安盟": 14.8, "锡林郭勒": 10.5, "阿拉善": 2.1},
            "precip_1h": {"呼和浩特": 6.8, "包头": 5.2, "乌海": 2.1,
                          "赤峰": 8.3, "通辽": 7.5, "鄂尔多斯": 3.8,
                          "呼伦贝尔": 1.5, "巴彦淖尔": 4.2, "乌兰察布": 5.1,
                          "兴安盟": 6.5, "锡林郭勒": 4.8, "阿拉善": 0.8},
            "wind_max": {"呼和浩特": 12.5, "包头": 14.2, "乌海": 10.8,
                         "赤峰": 16.5, "通辽": 18.2, "鄂尔多斯": 11.5,
                         "呼伦贝尔": 13.8, "巴彦淖尔": 12.0, "乌兰察布": 15.5,
                         "兴安盟": 13.2, "锡林郭勒": 14.8, "阿拉善": 9.5},
        },
        "GRAPES": {
            "precip_12h": {"呼和浩特": 18.5, "包头": 14.2, "乌海": 4.5,
                           "赤峰": 18.8, "通辽": 22.1, "鄂尔多斯": 10.2,
                           "呼伦贝尔": 5.5, "巴彦淖尔": 11.8, "乌兰察布": 13.5,
                           "兴安盟": 16.2, "锡林郭勒": 12.8, "阿拉善": 3.2},
            "precip_1h": {"呼和浩特": 7.5, "包头": 6.1, "乌海": 1.8,
                          "赤峰": 7.8, "通辽": 9.2, "鄂尔多斯": 4.5,
                          "呼伦贝尔": 2.2, "巴彦淖尔": 5.1, "乌兰察布": 5.8,
                          "兴安盟": 6.8, "锡林郭勒": 5.5, "阿拉善": 1.2},
            "wind_max": {"呼和浩特": 14.2, "包头": 13.8, "乌海": 11.5,
                         "赤峰": 15.8, "通辽": 16.5, "鄂尔多斯": 12.8,
                         "呼伦贝尔": 15.2, "巴彦淖尔": 13.5, "乌兰察布": 14.2,
                         "兴安盟": 14.8, "锡林郭勒": 15.5, "阿拉善": 10.8},
        },
        "GRAPES_MESO": {
            "precip_12h": {"呼和浩特": 14.8, "包头": 13.5, "乌海": 6.2,
                           "赤峰": 22.1, "通辽": 19.5, "鄂尔多斯": 9.1,
                           "呼伦贝尔": 4.8, "巴彦淖尔": 10.5, "乌兰察布": 12.8,
                           "兴安盟": 15.5, "锡林郭勒": 11.5, "阿拉善": 2.5},
            "precip_1h": {"呼和浩特": 6.5, "包头": 5.8, "乌海": 2.5,
                          "赤峰": 8.8, "通辽": 8.2, "鄂尔多斯": 4.2,
                          "呼伦贝尔": 1.8, "巴彦淖尔": 4.8, "乌兰察布": 5.5,
                          "兴安盟": 7.2, "锡林郭勒": 5.2, "阿拉善": 1.0},
            "wind_max": {"呼和浩特": 13.8, "包头": 14.5, "乌海": 11.2,
                         "赤峰": 17.2, "通辽": 17.5, "鄂尔多斯": 12.5,
                         "呼伦贝尔": 14.5, "巴彦淖尔": 12.8, "乌兰察布": 14.8,
                         "兴安盟": 13.5, "锡林郭勒": 15.2, "阿拉善": 10.2},
        },
        "NCEP": {
            "precip_12h": {"呼和浩特": 19.8, "包头": 15.8, "乌海": 5.2,
                           "赤峰": 17.5, "通辽": 23.5, "鄂尔多斯": 11.5,
                           "呼伦贝尔": 6.2, "巴彦淖尔": 12.8, "乌兰察布": 14.2,
                           "兴安盟": 17.8, "锡林郭勒": 13.5, "阿拉善": 3.8},
            "precip_1h": {"呼和浩特": 8.2, "包头": 6.8, "乌海": 2.0,
                          "赤峰": 7.5, "通辽": 9.8, "鄂尔多斯": 5.0,
                          "呼伦贝尔": 2.5, "巴彦淖尔": 5.5, "乌兰察布": 6.2,
                          "兴安盟": 7.5, "锡林郭勒": 5.8, "阿拉善": 1.5},
            "wind_max": {"呼和浩特": 11.8, "包头": 12.5, "乌海": 9.8,
                         "赤峰": 14.5, "通辽": 15.8, "鄂尔多斯": 10.5,
                         "呼伦贝尔": 12.8, "巴彦淖尔": 11.5, "乌兰察布": 13.2,
                         "兴安盟": 12.5, "锡林郭勒": 13.5, "阿拉善": 8.8},
        },
    }

    # 构建config对象供字典读取器使用
    source_names = list(custom_data.keys())
    method = FORECAST_CONFIG["fusion_method"]

    print(f"\n使用自定义数据源进行融合预报（{method}）")
    print(f"{'城市':<12} {'12h降水(mm)':>12} {'1h最大(mm)':>12} {'最大风级':>10}")
    print("-" * 50)

    for city in INNER_MONGOLIA_CITIES:
        city_name = city["name_short"]
        result_row = {}

        for var in ["precip_12h", "precip_1h", "wind_max"]:
            values = []
            names_for_var = []
            for src in source_names:
                if city_name in custom_data[src].get(var, {}):
                    values.append(custom_data[src][var][city_name])
                    names_for_var.append(src)

            # 构建权重
            weight_map = {cfg["name"]: cfg["weight"] for cfg in DATA_SOURCES}
            weights = [weight_map.get(n, 1.0 / len(names_for_var))
                       for n in names_for_var]
            total = sum(weights)
            weights = [w / total for w in weights]

            result = fuse_values(method, values, weights,
                                 source_names=names_for_var)
            result_row[var] = result

        # 风力等级转换
        wind_level = wind_speed_to_force_level(
            result_row["wind_max"], FORECAST_CONFIG["wind_force_thresholds"]
        )

        print(f"{city_name:<12} {result_row['precip_12h']:>12.2f} "
              f"{result_row['precip_1h']:>12.2f} {wind_level:>8} 级")


# ============================================================================
# 示例4：对比不同融合方法的结果
# ============================================================================
def example_4_compare_methods():
    """
    示例4：使用同一组数据对比不同融合方法的结果，帮助选择最合适的方法
    """
    print("\n" + "=" * 60)
    print("【示例4】对比不同融合方法对同一组数据的结果")
    print("=" * 60)

    from fusion import fuse_values, FUSION_METHODS, compute_fusion_statistics

    # 模拟一个"难"情况：有一个模式极端偏高，一个极端偏低
    source_values = [15.0, 14.5, 16.2, 35.0, 8.5]  # 存在极值
    source_names = ["模式A", "模式B", "模式C", "模式D(偏高)", "模式E(偏低)"]
    weights = [0.25, 0.25, 0.25, 0.125, 0.125]

    print("各数据源预报值:")
    for n, v in zip(source_names, source_values):
        print(f"  {n:>15}: {v:.2f} mm")

    stats = compute_fusion_statistics(source_values, source_names, weights)
    print(f"\n样本统计: 均值={stats['mean']:.2f}, "
          f"中位数={stats['median']:.2f}, "
          f"标准差={stats['std']:.2f}")

    print("\n各融合方法结果对比:")
    results = {}
    for method, chinese_name in FUSION_METHODS.items():
        val = fuse_values(method, source_values, weights,
                          source_names=source_names)
        results[method] = val
        deviation = val - stats["median"]
        marker = " <<< 对异常值更稳健" if abs(deviation) < 1.0 else ""
        print(f"  {chinese_name:>12} ({method:>20}): "
              f"{val:.2f} mm (距中位数: {deviation:+.2f}){marker}")

    print("\n【提示】当数据源分歧较大时，建议:")
    print("  - 使用 'ensemble_median' 对异常值更稳健")
    print("  - 使用 'object_based' 自动降低离群模式的权重")
    print("  - 有观测数据时使用 'bias_corrected' 可进一步提升质量")


# ============================================================================
# 示例5：生成JSON示例数据文件
# ============================================================================
def example_5_create_sample_data():
    """
    示例5：演示如何准备一个JSON格式的数据源文件
    """
    print("\n" + "=" * 60)
    print("【示例5】生成JSON格式示例数据文件（测试用）")
    print("=" * 60)

    from config import INNER_MONGOLIA_CITIES

    os.makedirs("./data/sample", exist_ok=True)
    sample_data = []

    np.random.seed(42)
    for city in INNER_MONGOLIA_CITIES:
        station = {
            "station_id": city["station_id"],
            "name": city["name"],
            "lon": city["lon"],
            "lat": city["lat"],
            "precip_12h": float(np.random.gamma(2.0, 5.0)),
            "precip_1h": float(np.random.gamma(1.5, 2.0)),
            "wind_max": float(np.random.uniform(5.0, 20.0)),
        }
        sample_data.append(station)

    filepath = "./data/sample/sample_forecast.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"示例数据已生成: {os.path.abspath(filepath)}")
    print(f"包含 {len(sample_data)} 个站点的示例预报")
    print("\n此文件可与 data_reader.JSONDataReader 配合使用，")
    print("或修改 config.py 中的数据源路径指向它。")


# ============================================================================
# 主程序：依次执行所有示例
# ============================================================================
def main():
    print("#" * 60)
    print("#  多源融合预报程序 - 使用示例 / 测试脚本")
    print("#  运行时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 60)

    examples = [
        ("示例1", example_1_basic_usage),
        ("示例2", example_2_python_api),
        ("示例3", example_3_custom_sources),
        ("示例4", example_4_compare_methods),
        ("示例5", example_5_create_sample_data),
    ]

    if len(sys.argv) > 1:
        # 只运行指定的示例
        targets = sys.argv[1:]
        for name, func in examples:
            if any(t in name for t in targets):
                func()
    else:
        # 运行所有示例
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"{name} 执行失败: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "#" * 60)
    print("#  所有示例运行完成！")
    print("#  下一步: 执行 'python main.py' 运行完整预报流程")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
