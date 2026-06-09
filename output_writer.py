# -*- coding: utf-8 -*-
"""
多源融合预报程序 - 输出模块
支持文本、CSV、JSON、HTML表格及可视化输出
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

from config import OUTPUT_CONFIG, FORECAST_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# 格式化函数
# ============================================================================
def format_value(value: Optional[float], variable: str) -> str:
    """将数值格式化为指定精度的字符串"""
    if value is None or np.isnan(value):
        return "-"
    precision = OUTPUT_CONFIG.get("precision", {}).get(variable, 1)
    return f"{value:.{precision}f}"


def get_unit(variable: str) -> str:
    """获取变量单位"""
    return OUTPUT_CONFIG.get("units", {}).get(variable, "")


def get_variable_display_name(variable: str) -> str:
    """获取变量中文显示名"""
    names = {
        "precip_12h": "12小时累计降水量",
        "precip_1h_max": "最大小时降水量",
        "wind_max": "最大风力等级",
    }
    return names.get(variable, variable)


# ============================================================================
# 1. 文本格式输出
# ============================================================================
def write_text_output(results: List[Dict], forecast_date: str,
                      cycle: int, output_dir: str = None) -> str:
    """
    输出易读的文本格式预报结果
    """
    output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    filename = f"FORECAST_{forecast_date}_{cycle:02d}.txt"
    filepath = os.path.join(output_dir, filename)

    var_names = FORECAST_CONFIG["variables"]
    display_names = [get_variable_display_name(v) for v in var_names]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("内蒙古自治区多源融合预报\n")
        f.write(f"起报时间: {forecast_date} {cycle:02d}:00 (BT)\n")
        f.write(f"预报时效: {FORECAST_CONFIG['forecast_hours']} 小时\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # 表头
        header = f"{'序号':<6}{'城市':<12}"
        for name, var in zip(display_names, var_names):
            unit = get_unit(var)
            header += f"{name}({unit})".center(20)
        f.write(header + "\n")
        f.write("-" * 80 + "\n")

        # 数据行
        for i, r in enumerate(results, 1):
            line = f"{i:<6}{r['city']:<12}"
            for var in var_names:
                val = r.get("fused", {}).get(var)
                line += f"{format_value(val, var):^20}"
            f.write(line + "\n")

        # 详细信息（各数据源原始值）
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("详细信息 - 各数据源原始值\n")
        f.write("-" * 80 + "\n")

        for i, r in enumerate(results, 1):
            f.write(f"\n[{i:02d}] {r['city']} "
                    f"(经度: {r.get('lon', 'N/A')}, "
                    f"纬度: {r.get('lat', 'N/A')})\n")
            for var in var_names:
                dn = get_variable_display_name(var)
                unit = get_unit(var)
                f.write(f"  {dn}({unit}):\n")
                src_data = r.get("sources", {}).get(var, {})
                if src_data:
                    for src, val in src_data.items():
                        f.write(f"    - {src:<15}: "
                                f"{format_value(val, var)}\n")
                fused = r.get("fused", {}).get(var)
                f.write(f"    {'融合结果':<15}: "
                        f"{format_value(fused, var)}\n")

                stats = r.get("statistics", {}).get(var, {})
                if stats:
                    f.write(f"    {'统计:最小/最大/标准差':<15}: "
                            f"{format_value(stats.get('min'), var)}/"
                            f"{format_value(stats.get('max'), var)}/"
                            f"{format_value(stats.get('std'), var)}\n")

    logger.info(f"文本输出: {filepath}")
    return filepath


# ============================================================================
# 2. CSV格式输出
# ============================================================================
def write_csv_output(results: List[Dict], forecast_date: str,
                     cycle: int, output_dir: str = None) -> str:
    """
    输出CSV格式预报结果（便于导入Excel等工具处理）
    """
    output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    filename = f"FORECAST_{forecast_date}_{cycle:02d}.csv"
    filepath = os.path.join(output_dir, filename)

    var_names = FORECAST_CONFIG["variables"]

    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        # 头信息
        writer.writerow(["# 内蒙古自治区多源融合预报"])
        writer.writerow([f"# 起报时间: {forecast_date} {cycle:02d}:00 (BT)"])
        writer.writerow([f"# 预报时效: {FORECAST_CONFIG['forecast_hours']} 小时"])
        writer.writerow([f"# 融合方法: {FORECAST_CONFIG['fusion_method']}"])
        writer.writerow([])

        # 主表 - 融合结果
        header = ["序号", "城市", "经度", "纬度"]
        for var in var_names:
            header.append(f"{get_variable_display_name(var)}({get_unit(var)})")
        for var in var_names:
            header.append(f"{get_variable_display_name(var)}_标准差")
        for var in var_names:
            header.append(f"{get_variable_display_name(var)}_数据源数")
        writer.writerow(header)

        for i, r in enumerate(results, 1):
            row = [i, r["city"], r.get("lon", ""), r.get("lat", "")]
            for var in var_names:
                val = r.get("fused", {}).get(var)
                row.append(format_value(val, var))
            for var in var_names:
                stats = r.get("statistics", {}).get(var, {})
                row.append(format_value(stats.get("std"), var))
            for var in var_names:
                stats = r.get("statistics", {}).get(var, {})
                row.append(stats.get("count", 0))
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["=== 各数据源详细数据 ==="])

        # 副表 - 各数据源值
        source_names = set()
        for r in results:
            for var in var_names:
                for src in r.get("sources", {}).get(var, {}).keys():
                    source_names.add(src)
        source_list = sorted(source_names)

        header2 = ["城市", "变量"] + source_list + ["融合结果"]
        writer.writerow(header2)

        for r in results:
            for var in var_names:
                row = [r["city"], get_variable_display_name(var)]
                src_data = r.get("sources", {}).get(var, {})
                for src in source_list:
                    row.append(format_value(src_data.get(src), var))
                row.append(format_value(r.get("fused", {}).get(var), var))
                writer.writerow(row)

    logger.info(f"CSV输出: {filepath}")
    return filepath


# ============================================================================
# 3. JSON格式输出
# ============================================================================
def write_json_output(results: List[Dict], forecast_date: str,
                      cycle: int, output_dir: str = None) -> str:
    """
    输出JSON格式预报结果（便于程序化调用）
    """
    output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    filename = f"FORECAST_{forecast_date}_{cycle:02d}.json"
    filepath = os.path.join(output_dir, filename)

    output_data = {
        "metadata": {
            "title": "内蒙古自治区多源融合预报",
            "forecast_date": forecast_date,
            "cycle": cycle,
            "forecast_hours": FORECAST_CONFIG["forecast_hours"],
            "fusion_method": FORECAST_CONFIG["fusion_method"],
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "variables": [
                {
                    "name": var,
                    "display_name": get_variable_display_name(var),
                    "unit": get_unit(var),
                }
                for var in FORECAST_CONFIG["variables"]
            ],
        },
        "results": results,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"JSON输出: {filepath}")
    return filepath


# ============================================================================
# 4. HTML格式输出
# ============================================================================
def write_html_output(results: List[Dict], forecast_date: str,
                      cycle: int, output_dir: str = None) -> str:
    """
    输出HTML格式预报结果（支持浏览器查看）
    """
    output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    filename = f"FORECAST_{forecast_date}_{cycle:02d}.html"
    filepath = os.path.join(output_dir, filename)

    var_names = FORECAST_CONFIG["variables"]
    display_names = [get_variable_display_name(v) for v in var_names]
    units = [get_unit(v) for v in var_names]

    # 定义颜色分级
    def precip_color(v: Optional[float]) -> str:
        if v is None or np.isnan(v):
            return "#f5f5f5"
        if v < 0.1:
            return "#ffffff"
        elif v < 5:
            return "#c8e6c9"
        elif v < 10:
            return "#81c784"
        elif v < 25:
            return "#4fc3f7"
        elif v < 50:
            return "#2196f3"
        elif v < 100:
            return "#ff9800"
        else:
            return "#f44336"

    def wind_color(v: Optional[float]) -> str:
        if v is None or np.isnan(v):
            return "#f5f5f5"
        if v < 3:
            return "#e8f5e9"
        elif v < 5:
            return "#fff9c4"
        elif v < 7:
            return "#ffe0b2"
        elif v < 9:
            return "#ffccbc"
        else:
            return "#ffab91"

    rows_html = ""
    for i, r in enumerate(results, 1):
        cells = f"<td>{i}</td><td><b>{r['city']}</b></td>"
        for var in var_names:
            val = r.get("fused", {}).get(var)
            if "wind" in var:
                color = wind_color(val)
            else:
                color = precip_color(val)
            cells += (f"<td style='background-color:{color}; "
                      f"text-align:center;'>{format_value(val, var)}</td>")
        rows_html += f"<tr>{cells}</tr>\n"

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>内蒙古自治区多源融合预报 - {forecast_date} {cycle:02d}时</title>
    <style>
        body {{ font-family: "Microsoft YaHei", sans-serif; margin: 20px;
               background: #fafafa; }}
        h1 {{ color: #1565c0; text-align: center; }}
        .meta {{ text-align: center; color: #666; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; margin: 20px auto;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th, td {{ padding: 10px 16px; border: 1px solid #ddd;
                  text-align: center; }}
        th {{ background: #1976d2; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background: #f5f5f5; }}
        tr:hover {{ background: #e3f2fd; }}
        .legend {{ margin: 20px auto; max-width: 800px; padding: 10px;
                   background: white; border-radius: 4px; }}
        .legend-item {{ display: inline-block; margin: 5px; padding: 5px 10px;
                        border-radius: 3px; }}
        .details {{ margin-top: 20px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>内蒙古自治区多源融合预报</h1>
    <div class="meta">
        <p><b>起报时间:</b> {forecast_date} {cycle:02d}:00 (北京时间)
           &nbsp;&nbsp;<b>预报时效:</b> {FORECAST_CONFIG['forecast_hours']} 小时
           &nbsp;&nbsp;<b>融合方法:</b> {FORECAST_CONFIG['fusion_method']}</p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <table>
        <tr>
            <th>序号</th><th>城市</th>
            {"".join(f"<th>{dn}<br/>({u})</th>" for dn, u in zip(display_names, units))}
        </tr>
        {rows_html}
    </table>

    <div class="legend">
        <b>降水量色标说明 (mm):</b>
        <span class="legend-item" style="background:#c8e6c9">0.1-5</span>
        <span class="legend-item" style="background:#81c784">5-10</span>
        <span class="legend-item" style="background:#4fc3f7">10-25</span>
        <span class="legend-item" style="background:#2196f3">25-50</span>
        <span class="legend-item" style="background:#ff9800">50-100</span>
        <span class="legend-item" style="background:#f44336">&ge;100</span>
        <br/>
        <b>风力等级色标说明 (级):</b>
        <span class="legend-item" style="background:#e8f5e9">&lt;3</span>
        <span class="legend-item" style="background:#fff9c4">3-5</span>
        <span class="legend-item" style="background:#ffe0b2">5-7</span>
        <span class="legend-item" style="background:#ffccbc">7-9</span>
        <span class="legend-item" style="background:#ffab91">&ge;9</span>
    </div>

    <div class="details">
        <h3 style="color:#1565c0;">各数据源详细信息</h3>
    </div>
</body>
</html>
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"HTML输出: {filepath}")
    return filepath


# ============================================================================
# 5. 可视化输出
# ============================================================================
def write_plot_output(results: List[Dict], forecast_date: str,
                      cycle: int, output_dir: str = None) -> Optional[str]:
    """
    生成可视化柱状图（需要matplotlib）
    """
    output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    filename = f"FORECAST_{forecast_date}_{cycle:02d}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                           'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        var_names = FORECAST_CONFIG["variables"]
        n_vars = len(var_names)
        cities = [r["city"] for r in results]

        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4 * n_vars))
        if n_vars == 1:
            axes = [axes]

        for ax, var in zip(axes, var_names):
            values = [r.get("fused", {}).get(var, 0) or 0 for r in results]
            colors = ['#2196f3' if 'precip' in var else '#ff9800'
                      for _ in values]
            bars = ax.bar(range(len(cities)), values, color=colors,
                          alpha=0.8, edgecolor='white')
            ax.set_xticks(range(len(cities)))
            ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(get_unit(var))
            ax.set_title(get_variable_display_name(var))
            ax.grid(True, alpha=0.3, axis='y')
            for bar, v in zip(bars, values):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2.,
                            bar.get_height(),
                            f'{v:.1f}', ha='center', va='bottom',
                            fontsize=8)

        plt.suptitle(f'内蒙古多源融合预报\n'
                     f'{forecast_date} {cycle:02d}时起报',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"可视化输出: {filepath}")
        return filepath

    except ImportError:
        logger.warning("matplotlib未安装，无法生成可视化图")
        return None
    except Exception as e:
        logger.warning(f"可视化生成失败: {e}")
        return None


# ============================================================================
# 统一输出接口
# ============================================================================
def write_all_outputs(results: List[Dict], forecast_date: str,
                      cycle: int, output_dir: str = None,
                      formats: List[str] = None) -> Dict[str, str]:
    """
    按照配置格式输出所有结果

    参数:
        results: 融合结果列表
        forecast_date: 起报日期(YYYYMMDD)
        cycle: 起报时刻(8或20)
        output_dir: 输出目录
        formats: 输出格式列表
    返回:
        {格式名: 文件路径}
    """
    formats = formats or OUTPUT_CONFIG.get("formats", ["txt", "csv", "json"])
    output_files = {}

    format_writers = {
        "txt": write_text_output,
        "csv": write_csv_output,
        "json": write_json_output,
        "html": write_html_output,
    }

    for fmt in formats:
        writer = format_writers.get(fmt)
        if writer:
            try:
                fp = writer(results, forecast_date, cycle, output_dir)
                output_files[fmt] = fp
            except Exception as e:
                logger.error(f"{fmt}格式输出失败: {e}")

    if OUTPUT_CONFIG.get("generate_plot", True):
        plot_path = write_plot_output(results, forecast_date,
                                      cycle, output_dir)
        if plot_path:
            output_files["plot"] = plot_path

    return output_files


# ============================================================================
# 控制台打印输出
# ============================================================================
def print_to_console(results: List[Dict], forecast_date: str, cycle: int):
    """
    在控制台打印融合结果（用于快速预览）
    """
    var_names = FORECAST_CONFIG["variables"]
    display_names = [get_variable_display_name(v) for v in var_names]

    print("\n" + "=" * 70)
    print("内蒙古自治区多源融合预报")
    print(f"起报时间: {forecast_date} {cycle:02d}:00 (BT) | "
          f"预报时效: {FORECAST_CONFIG['forecast_hours']} 小时")
    print(f"融合方法: {FORECAST_CONFIG['fusion_method']}")
    print("=" * 70)

    header = f"{'序号':<6}{'城市':<12}"
    for dn, var in zip(display_names, var_names):
        header += f"{dn}({get_unit(var)})".center(18)
    print(header)
    print("-" * 70)

    for i, r in enumerate(results, 1):
        line = f"{i:<6}{r['city']:<12}"
        for var in var_names:
            val = r.get("fused", {}).get(var)
            line += f"{format_value(val, var):^18}"
        print(line)

    print("=" * 70 + "\n")
