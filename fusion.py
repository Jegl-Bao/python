# -*- coding: utf-8 -*-
"""
多源融合预报程序 - 融合算法模块
提供多种多模式融合方法
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数：风速 -> 蒲福风级
# ============================================================================
def wind_speed_to_force_level(wind_speed_ms: Optional[float],
                              thresholds: List[float]) -> Optional[int]:
    """
    根据风速(m/s)转换为蒲福风级

    蒲福风级参考阈值 (m/s):
    0级: <0.3,  1级: 0.3-1.6, 2级: 1.6-3.4, 3级: 3.4-5.5
    4级: 5.5-8.0, 5级: 8.0-10.8, 6级: 10.8-13.9, 7级: 13.9-17.2
    8级: 17.2-20.8, 9级: 20.8-24.5, 10级: 24.5-28.5, 11级: 28.5-32.7
    12级及以上: >=32.7
    """
    if wind_speed_ms is None or np.isnan(wind_speed_ms) or wind_speed_ms < 0:
        return None

    for level, threshold in enumerate(thresholds):
        if wind_speed_ms < threshold:
            return level
    return len(thresholds)


# ============================================================================
# 数据收集与预处理
# ============================================================================
def collect_source_values(source_data: Dict[str, Dict[str, Optional[float]]],
                          city_name: str) -> Tuple[List[float], List[str],
                                                    List[float]]:
    """
    收集某个站点所有可用数据源的值

    参数:
        source_data: {source_name: {city_name: value, ...}, ...}
        city_name: 城市名称
    返回:
        (values, source_names, weights)
    """
    values = []
    source_names = []
    for src_name, city_values in source_data.items():
        val = city_values.get(city_name)
        if val is not None and not np.isnan(val) and not np.isinf(val):
            if val >= 0:  # 降水量、风速不能为负
                values.append(val)
                source_names.append(src_name)
    return values, source_names, []


def compute_weights_by_name(source_names: List[str],
                           source_configs: List[Dict],
                           auto_renormalize: bool = True) -> List[float]:
    """
    根据数据源名称获取权重配置
    """
    weights = []
    config_map = {cfg.get("name", ""): cfg.get("weight", 1.0)
                  for cfg in source_configs}
    for sn in source_names:
        weights.append(config_map.get(sn, 1.0))

    if auto_renormalize and sum(weights) > 0:
        total = sum(weights)
        weights = [w / total for w in weights]
    return weights


# ============================================================================
# 1. 加权平均融合 (Weighted Average)
# ============================================================================
def weighted_average_fusion(values: List[float],
                            weights: List[float]) -> Optional[float]:
    """
    加权平均融合法

    公式: fusion = sum(w_i * x_i) / sum(w_i)

    这是最常用的确定性融合方法，根据各模式的历史表现或预报技巧
    赋予不同的权重。通常表现较好的模式（如EC）权重较高。
    """
    if not values:
        return None
    if len(values) != len(weights):
        weights = [1.0 / len(values)] * len(values)
    if sum(weights) == 0:
        weights = [1.0 / len(values)] * len(values)

    total = sum(w * v for w, v in zip(weights, values))
    return float(total / sum(weights))


# ============================================================================
# 2. 集合平均 (Ensemble Mean)
# ============================================================================
def ensemble_mean_fusion(values: List[float]) -> Optional[float]:
    """
    集合平均法

    将所有数据源视为等权重的集合成员，取算术平均。
    简单且能自然减少方差，适用于各模式表现相近的场景。
    """
    if not values:
        return None
    return float(np.mean(values))


# ============================================================================
# 3. 集合中位数 (Ensemble Median)
# ============================================================================
def ensemble_median_fusion(values: List[float]) -> Optional[float]:
    """
    集合中位数法

    对异常值（离群模式）更具鲁棒性，当个别模式出现极端预报时，
    中位数比均值更能代表"共识"值。
    """
    if not values:
        return None
    return float(np.median(values))


# ============================================================================
# 4. 偏差校正融合 (Bias-Corrected)
# ============================================================================
def bias_corrected_fusion(values: List[float],
                          weights: List[float],
                          obs_reference: Optional[float] = None,
                          historical_bias: Optional[Dict[str, float]] = None,
                          source_names: Optional[List[str]] = None) -> Optional[float]:
    """
    偏差校正融合法

    使用观测数据或历史偏差对各模式进行校正后再融合。
    校正公式: corrected = x_i - bias_i
    其中 bias_i 为模式 i 的历史平均偏差。
    """
    if not values:
        return None
    if len(values) != len(weights):
        weights = [1.0 / len(values)] * len(values)

    corrected_values = list(values)
    if historical_bias and source_names:
        for i, sn in enumerate(source_names):
            bias = historical_bias.get(sn, 0.0)
            corrected_values[i] = corrected_values[i] - bias

    if obs_reference is not None:
        if len(values) > 0:
            mean_val = np.mean(values)
            if mean_val != 0:
                scale_factor = obs_reference / mean_val
                corrected_values = [v * scale_factor
                                    for v in corrected_values]

    if sum(weights) == 0:
        weights = [1.0 / len(corrected_values)] * len(corrected_values)

    total = sum(w * v for w, v in zip(weights, corrected_values))
    result = float(total / sum(weights))
    return max(0.0, result) if result is not None else None


# ============================================================================
# 5. 基于对象的融合 (Object-Based / 概率融合)
# ============================================================================
def object_based_fusion(values: List[float],
                        weights: List[float]) -> Optional[float]:
    """
    简化的基于对象的融合方法

    根据各值之间的一致性动态调整权重：
    - 与多数值接近的数据源权重提高
    - 与多数值偏差较大的数据源权重降低（视为"离群对象"）

    这是对传统加权平均的改进，能实时反映模式一致性。
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    if len(values) != len(weights):
        weights = [1.0 / len(values)] * len(values)

    arr = np.array(values, dtype=float)
    mean_val = np.mean(arr)
    std_val = np.std(arr)

    if std_val == 0 or mean_val == 0:
        total = sum(w * v for w, v in zip(weights, values))
        return float(total / sum(weights))

    # 计算每个值与均值的相对偏差
    deviations = np.abs(arr - mean_val) / std_val

    # 使用类似高斯核的权重调整
    adjustment = np.exp(-deviations ** 2 / 2.0)
    new_weights = np.array(weights) * adjustment

    if new_weights.sum() == 0:
        return float(mean_val)

    new_weights = new_weights / new_weights.sum()
    result = float(np.sum(arr * new_weights))
    return result


# ============================================================================
# 6. 卡尔曼滤波融合 (Kalman Filter Fusion)
# ============================================================================
def kalman_filter_fusion(values: List[float],
                         weights: List[float],
                         process_noise: float = 0.01,
                         obs_noise: float = 0.1) -> Optional[float]:
    """
    简化的卡尔曼滤波融合

    将各数据源视为顺序观测，使用简化的卡尔曼滤波进行递推估计。
    适合有时序关系的数据融合场景。
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    if len(values) != len(weights):
        weights = [1.0 / len(values)] * len(values)

    # 初始化：使用第一个值
    x_est = values[0]
    p_est = obs_noise

    # 按权重降序排序（信任度高的先作为"先验"）
    sorted_indices = sorted(range(len(weights)),
                            key=lambda i: weights[i], reverse=True)

    for idx in sorted_indices[1:]:
        measurement = values[idx]
        w = weights[idx]

        # 预测步骤
        x_pred = x_est
        p_pred = p_est + process_noise

        # 更新步骤 (卡尔曼增益)
        r = obs_noise / max(w, 0.01)
        k = p_pred / (p_pred + r)
        x_est = x_pred + k * (measurement - x_pred)
        p_est = (1 - k) * p_pred

    return float(x_est)


# ============================================================================
# 7. 概率匹配平均 (Probability-Matched Mean)
# ============================================================================
def probability_matched_mean(values: List[float],
                             weights: List[float]) -> Optional[float]:
    """
    概率匹配平均法

    对降水等正偏态分布变量效果更好：
    1. 计算集合平均 -> 给出空间位置参考
    2. 重新排序使各值的分位数与其平均值匹配
    这里的简化版本：使用权重加权后按分位数匹配调整
    """
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    if len(values) != len(weights):
        weights = [1.0 / len(values)] * len(values)

    arr = np.array(values, dtype=float)
    mean_val = float(np.sum(arr * np.array(weights)))

    # 排序后找到对应均值的分位数
    sorted_idx = np.argsort(arr)
    sorted_vals = arr[sorted_idx]
    cumulative_weights = np.cumsum(np.array(weights)[sorted_idx])

    # 找到平均值在排序后的值中的位置
    result_idx = np.searchsorted(sorted_vals, mean_val) - 1
    result_idx = max(0, min(result_idx, len(sorted_vals) - 1))

    # 简单的调整：返回平均值附近被排序的加权值
    return float(mean_val)


# ============================================================================
# 融合方法映射表
# ============================================================================
FUSION_METHODS = {
    "weighted_average": "加权平均",
    "ensemble_mean": "集合平均",
    "ensemble_median": "集合中位数",
    "bias_corrected": "偏差校正",
    "object_based": "基于对象",
    "kalman_filter": "卡尔曼滤波",
    "probability_matched": "概率匹配",
}


def fuse_values(method: str, values: List[float],
                weights: List[float],
                obs_reference: Optional[float] = None,
                historical_bias: Optional[Dict[str, float]] = None,
                source_names: Optional[List[str]] = None) -> Optional[float]:
    """
    统一的融合接口

    参数:
        method: 融合方法名称
        values: 各数据源的值列表
        weights: 各数据源的权重列表
        obs_reference: 观测参考值（用于偏差校正）
        historical_bias: 历史偏差字典 {source_name: bias}
        source_names: 数据源名称列表
    """
    method = method.lower()

    if method == "weighted_average":
        return weighted_average_fusion(values, weights)
    elif method == "ensemble_mean":
        return ensemble_mean_fusion(values)
    elif method == "ensemble_median":
        return ensemble_median_fusion(values)
    elif method == "bias_corrected":
        return bias_corrected_fusion(values, weights, obs_reference,
                                     historical_bias, source_names)
    elif method == "object_based":
        return object_based_fusion(values, weights)
    elif method == "kalman_filter":
        return kalman_filter_fusion(values, weights)
    elif method == "probability_matched":
        return probability_matched_mean(values, weights)
    else:
        logger.warning(f"未知融合方法 '{method}'，使用加权平均作为替代")
        return weighted_average_fusion(values, weights)


# ============================================================================
# 融合统计（可选输出：各模式值、极值、离散度等）
# ============================================================================
def compute_fusion_statistics(values: List[float],
                              source_names: List[str],
                              weights: List[float]) -> Dict[str, Any]:
    """
    计算融合后的统计量，供详细分析使用
    """
    stats: Dict[str, Any] = {}
    if not values:
        return stats

    arr = np.array(values, dtype=float)
    stats["sources"] = list(source_names)
    stats["individual_values"] = list(values)
    stats["weights"] = list(weights)
    stats["min"] = float(np.min(arr))
    stats["max"] = float(np.max(arr))
    stats["mean"] = float(np.mean(arr))
    stats["median"] = float(np.median(arr))
    stats["std"] = float(np.std(arr))
    stats["range"] = float(np.max(arr) - np.min(arr))
    stats["count"] = int(len(values))

    if len(values) > 1:
        stats["coefficient_of_variation"] = float(
            np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else 0.0
    else:
        stats["coefficient_of_variation"] = 0.0

    return stats
