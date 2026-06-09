# -*- coding: utf-8 -*-
"""
多源融合预报程序 - 主程序
预报内蒙古自治区12个盟市的:
  - 12小时累计降水量
  - 最大小时降水量
  - 最大风力等级

起报时刻: 08时和20时（北京时间）
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np

from config import (
    INNER_MONGOLIA_CITIES,
    DATA_SOURCES,
    FORECAST_CONFIG,
    OUTPUT_CONFIG,
)
from data_reader import create_reader
from fusion import (
    fuse_values,
    collect_source_values,
    compute_weights_by_name,
    wind_speed_to_force_level,
    compute_fusion_statistics,
    FUSION_METHODS,
)
from output_writer import write_all_outputs, print_to_console

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("multi_source_fusion")


# ============================================================================
# 参数解析
# ============================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='内蒙古自治区多源融合预报系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用当前日期，默认起报时刻08和20时
  python main.py

  # 指定起报日期
  python main.py --date 20250101

  # 指定起报日期和起报时刻
  python main.py --date 20250101 --cycles 8 20

  # 指定融合方法
  python main.py --method ensemble_median

  # 使用其他输出目录
  python main.py --output-dir ./results
        """
    )

    parser.add_argument(
        '--date', '-d', type=str, default=None,
        help='起报日期 (YYYYMMDD格式)，默认为当前日期'
    )
    parser.add_argument(
        '--cycles', '-c', type=int, nargs='+', default=None,
        help='起报时刻列表 (如 8 20)，默认为配置中的所有时刻'
    )
    parser.add_argument(
        '--method', '-m', type=str, default=None,
        choices=list(FUSION_METHODS.keys()),
        help=f'融合方法，默认使用配置文件中的设置。可选: {", ".join(FUSION_METHODS.keys())}'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help='输出目录，默认为配置中的目录'
    )
    parser.add_argument(
        '--formats', '-f', type=str, nargs='+',
        default=None,
        choices=['txt', 'csv', 'json', 'html'],
        help='输出格式列表，默认使用配置文件中的设置'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='不生成可视化图'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='显示详细日志'
    )

    return parser.parse_args()


# ============================================================================
# 文件路径构建
# ============================================================================
def build_source_path(source_config: Dict, forecast_date: str,
                      cycle: int, forecast_hours: int) -> str:
    """
    根据配置构建数据源文件路径
    """
    path_pattern = source_config.get(
        "path_pattern",
        "./data/{name}/{date}/{name}_{date}{cycle:02d}_{fhour:03d}.grb"
    )

    path = path_pattern.format(
        name=source_config.get("name", "UNKNOWN"),
        date=forecast_date,
        cycle=cycle,
        fhour=forecast_hours,
    )
    return path


# ============================================================================
# 读取单变量的所有源数据
# ============================================================================
def read_variable_from_sources(source_configs: List[Dict],
                               forecast_date: str,
                               cycle: int,
                               forecast_hours: int,
                               variable_name: str,
                               cities: List[Dict],
                               interpolation_method: str) -> Dict[str, Dict[str, Optional[float]]]:
    """
    从所有数据源读取指定变量在各城市的值

    返回:
        {source_name: {city_name_short: value, ...}, ...}
    """
    results = {}

    for source_cfg in source_configs:
        if not source_cfg.get("enabled", True):
            continue

        source_name = source_cfg["name"]
        var_key = source_cfg.get("variables", {}).get(variable_name, variable_name)

        filepath = build_source_path(source_cfg, forecast_date,
                                     cycle, forecast_hours)

        logger.debug(f"  [{source_name}] 读取 {variable_name} "
                     f"(文件变量: {var_key})")
        logger.debug(f"    文件: {filepath}")

        reader = create_reader(source_cfg, interpolation_method)
        values = reader.read_variable(filepath, var_key, cities)

        results[source_name] = values

    return results


# ============================================================================
# 主预报流程
# ============================================================================
def run_forecast(forecast_date: str,
                 cycle: int,
                 fusion_method: str,
                 output_dir: str,
                 output_formats: List[str],
                 generate_plot: bool = True) -> List[Dict]:
    """
    执行单一起报时刻的多源融合预报流程

    步骤:
    1. 从各数据源读取每个变量在每个站点的值
    2. 对每个站点、每个变量执行多源融合
    3. 对风速变量转换为蒲福风级
    4. 输出结果
    """
    logger.info(f"=" * 60)
    logger.info(f"开始预报 | 日期: {forecast_date} | 起报: {cycle:02d}时")
    logger.info(f"融合方法: {fusion_method} | 预报时效: "
                f"{FORECAST_CONFIG['forecast_hours']} 小时")
    logger.info(f"=" * 60)

    cities = INNER_MONGOLIA_CITIES
    variables = FORECAST_CONFIG["variables"]
    forecast_hours = FORECAST_CONFIG["forecast_hours"]
    wind_thresholds = FORECAST_CONFIG["wind_force_thresholds"]
    interp_method = FORECAST_CONFIG["interpolation_method"]
    auto_renormalize = FORECAST_CONFIG["auto_renormalize_weights"]

    # 步骤1: 从各数据源读取各变量
    logger.info("步骤1: 读取各数据源...")
    source_data_by_var = {}
    for var in variables:
        logger.info(f"  读取变量: {var}")
        var_name_for_read = var
        # 最大小时降水量和12小时累计降水，都是降水，
        # 实际文件中需要处理不同时效。这里简化处理为同一变量名，
        # 实际业务中可根据配置映射到不同文件/变量。
        if var == "precip_1h_max":
            var_name_for_read = "precip_1h"
        source_data_by_var[var] = read_variable_from_sources(
            DATA_SOURCES, forecast_date, cycle,
            forecast_hours, var_name_for_read, cities, interp_method
        )

    # 步骤2: 对每个站点、每个变量进行融合
    logger.info("步骤2: 多源融合计算...")
    results = []

    for city in cities:
        city_name = city["name_short"]
        city_result: Dict[str, Any] = {
            "city": city_name,
            "city_full": city["name"],
            "lon": city["lon"],
            "lat": city["lat"],
            "station_id": city["station_id"],
            "sources": {},
            "fused": {},
            "statistics": {},
        }

        for var in variables:
            # 收集各数据源对该站点该变量的值
            values, src_names, _ = collect_source_values(
                source_data_by_var[var], city_name
            )

            # 记录各源原始值
            raw_sources = {}
            for src_name, city_vals in source_data_by_var[var].items():
                if city_name in city_vals:
                    raw_sources[src_name] = city_vals[city_name]
            city_result["sources"][var] = raw_sources

            # 计算权重
            weights = compute_weights_by_name(src_names, DATA_SOURCES,
                                              auto_renormalize)

            # 融合
            if values:
                fused_val = fuse_values(
                    method=fusion_method,
                    values=values,
                    weights=weights,
                    source_names=src_names,
                )

                # 对风力等级做特殊处理：如果fusion得到的是风速(m/s)，
                # 转换为蒲福风级；如果已经是风级则直接使用。
                if var == "wind_max":
                    # 判断是否需要转换：典型风速值在0-50 m/s，
                    # 蒲福风级在0-12。若值大于15则视为风速，需转换。
                    if fused_val is not None and fused_val > 15.0:
                        fused_val = wind_speed_to_force_level(fused_val,
                                                               wind_thresholds)
                    elif fused_val is not None and fused_val < 13.0:
                        # 可能已经是风级，或需要判断
                        # 这里保守：检查各数据源的量级
                        if any(v > 15.0 for v in values if v is not None):
                            fused_val = wind_speed_to_force_level(fused_val,
                                                                   wind_thresholds)

                city_result["fused"][var] = fused_val

                # 统计信息
                city_result["statistics"][var] = compute_fusion_statistics(
                    values, src_names, weights
                )
            else:
                city_result["fused"][var] = None
                city_result["statistics"][var] = {"count": 0}

            logger.debug(f"  [{city_name}] {var}: "
                         f"{' '.join(f'{k}={v:.1f}' for k, v in raw_sources.items() if v is not None)}"
                         f" -> 融合={city_result['fused'][var]}")

        results.append(city_result)

    # 步骤3: 输出
    logger.info("步骤3: 输出结果...")
    if generate_plot:
        OUTPUT_CONFIG["generate_plot"] = True
    else:
        OUTPUT_CONFIG["generate_plot"] = False

    output_files = write_all_outputs(results, forecast_date, cycle,
                                     output_dir, output_formats)

    # 控制台打印
    print_to_console(results, forecast_date, cycle)

    logger.info(f"生成文件:")
    for fmt, path in output_files.items():
        logger.info(f"  - [{fmt.upper()}] {path}")

    logger.info(f"预报完成 | 日期: {forecast_date} | 起报: {cycle:02d}时")

    return results


# ============================================================================
# 主函数
# ============================================================================
def main():
    """程序入口"""
    args = parse_args()

    # 日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 处理日期
    if args.date:
        try:
            datetime.strptime(args.date, "%Y%m%d")
            forecast_date = args.date
        except ValueError:
            logger.error(f"日期格式错误: {args.date}，请使用YYYYMMDD格式")
            sys.exit(1)
    else:
        forecast_date = datetime.now().strftime("%Y%m%d")

    # 处理起报时刻
    if args.cycles:
        cycles = [c for c in args.cycles if c in FORECAST_CONFIG["cycles"]]
        if not cycles:
            logger.error(f"起报时刻必须从 {FORECAST_CONFIG['cycles']} 中选择")
            sys.exit(1)
    else:
        cycles = FORECAST_CONFIG["cycles"]

    # 融合方法
    fusion_method = args.method or FORECAST_CONFIG["fusion_method"]

    # 输出目录
    output_dir = args.output_dir or OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 输出格式
    output_formats = args.formats or OUTPUT_CONFIG.get(
        "formats", ["txt", "csv", "json"]
    )

    # 是否生成图
    generate_plot = not args.no_plot

    # 运行各起报时刻的预报
    logger.info(f"多源融合预报系统启动")
    logger.info(f"预报日期: {forecast_date}")
    logger.info(f"起报时刻: {cycles}")
    logger.info(f"融合方法: {fusion_method} ({FUSION_METHODS.get(fusion_method, fusion_method)})")
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")
    logger.info(f"输出格式: {output_formats}")
    logger.info(f"预报区域: 内蒙古自治区 {len(INNER_MONGOLIA_CITIES)} 个盟市")
    logger.info(f"预报变量: {FORECAST_CONFIG['variables']}")

    all_results = {}
    for cycle in cycles:
        results = run_forecast(
            forecast_date=forecast_date,
            cycle=cycle,
            fusion_method=fusion_method,
            output_dir=output_dir,
            output_formats=output_formats,
            generate_plot=generate_plot,
        )
        all_results[cycle] = results

    # 总结信息
    logger.info("")
    logger.info("=" * 60)
    logger.info("全部预报任务完成！")
    logger.info(f"共处理 {len(cycles)} 个起报时刻，"
                f"{len(INNER_MONGOLIA_CITIES)} 个站点，"
                f"{len(FORECAST_CONFIG['variables'])} 个变量")
    logger.info(f"结果目录: {os.path.abspath(output_dir)}")
    logger.info("=" * 60)

    return all_results


if __name__ == "__main__":
    main()
