# -*- coding: utf-8 -*-
"""
多源融合预报程序 - 配置文件
包含内蒙古自治区12个盟市的地理信息和数据源配置
支持多种真实气象数据源下载与本地缓存
"""

import os
from typing import Dict

# ============================================================================
# 内蒙古自治区12个盟市信息
# ============================================================================
INNER_MONGOLIA_CITIES = [
    {
        "name": "呼和浩特市",
        "name_short": "呼和浩特",
        "province": "内蒙古自治区",
        "lon": 111.65,
        "lat": 40.82,
        "station_id": "53463",
    },
    {
        "name": "包头市",
        "name_short": "包头",
        "province": "内蒙古自治区",
        "lon": 109.85,
        "lat": 40.66,
        "station_id": "53466",
    },
    {
        "name": "乌海市",
        "name_short": "乌海",
        "province": "内蒙古自治区",
        "lon": 106.78,
        "lat": 39.65,
        "station_id": "53512",
    },
    {
        "name": "赤峰市",
        "name_short": "赤峰",
        "province": "内蒙古自治区",
        "lon": 118.95,
        "lat": 42.28,
        "station_id": "54218",
    },
    {
        "name": "通辽市",
        "name_short": "通辽",
        "province": "内蒙古自治区",
        "lon": 122.28,
        "lat": 43.61,
        "station_id": "54135",
    },
    {
        "name": "鄂尔多斯市",
        "name_short": "鄂尔多斯",
        "province": "内蒙古自治区",
        "lon": 109.78,
        "lat": 39.61,
        "station_id": "53543",
    },
    {
        "name": "呼伦贝尔市",
        "name_short": "呼伦贝尔",
        "province": "内蒙古自治区",
        "lon": 119.75,
        "lat": 49.22,
        "station_id": "50527",
    },
    {
        "name": "巴彦淖尔市",
        "name_short": "巴彦淖尔",
        "province": "内蒙古自治区",
        "lon": 107.38,
        "lat": 40.74,
        "station_id": "53439",
    },
    {
        "name": "乌兰察布市",
        "name_short": "乌兰察布",
        "province": "内蒙古自治区",
        "lon": 113.12,
        "lat": 41.03,
        "station_id": "53479",
    },
    {
        "name": "兴安盟",
        "name_short": "兴安盟",
        "province": "内蒙古自治区",
        "lon": 122.07,
        "lat": 46.07,
        "station_id": "50842",
    },
    {
        "name": "锡林郭勒盟",
        "name_short": "锡林郭勒",
        "province": "内蒙古自治区",
        "lon": 116.03,
        "lat": 43.95,
        "station_id": "54102",
    },
    {
        "name": "阿拉善盟",
        "name_short": "阿拉善",
        "province": "内蒙古自治区",
        "lon": 105.73,
        "lat": 38.85,
        "station_id": "52576",
    },
]

# ============================================================================
# 数据源配置
# 支持的数据源类型：
#   - 'grib': GRIB1/GRIB2格式（需要pygrib或ecCodes库）
#   - 'netcdf': NetCDF格式（需要netCDF4或xarray库）
#   - 'text': 文本格式（如MICAPS第3类、第4类格式或自定义文本）
#   - 'csv': CSV表格格式
#   - 'json': JSON格式
#   - 'dict': Python字典（直接传入数据）
# ============================================================================
DATA_SOURCES = [
    {
        "name": "EC",
        "description": "欧洲中期天气预报中心(ECMWF)模式",
        "type": "grib",
        "weight": 0.35,
        "variables": {
            "precip_12h": "tp",
            "precip_1h": "tp",
            "wind_max": "10si",
        },
        "path_pattern": "./data/ec/{date}/EC_{date}{cycle:02d}_{fhour:03d}.grb",
        "enabled": True,
        "url_template": "https://data.ecmwf.int/forecasts/{date}/{cycle}z/ifs/{resolution}/oper/{date}{cycle}0000-{fhour}h-oper-fc.{file_format}",
        "download": {
            "enabled": True,
            "cache_dir": "./data/EC/",
            "timeout": 60,
            "retries": 3,
            "backend": "ecmwf",
        },
        "resolution": "0p4-beta",
        "stream": "oper",
        "product": "fc",
        "file_format": "grib2",
    },
    {
        "name": "GRAPES",
        "description": "中国气象局GRAPES全球模式",
        "type": "grib",
        "weight": 0.25,
        "variables": {
            "precip_12h": "tp",
            "precip_1h": "tp",
            "wind_max": "10si",
        },
        "path_pattern": "./data/grapes/{date}/GRAPES_{date}{cycle:02d}_{fhour:03d}.grb",
        "enabled": True,
        "url_template": "https://data.cma.cn/api/grapes-gfs/{resolution}/{date}/{cycle}/grapes_gfs_{date}{cycle}_f{fhour}.{file_format}",
        "download": {
            "enabled": True,
            "cache_dir": "./data/GRAPES/",
            "timeout": 60,
            "retries": 3,
            "backend": "cma_api",
        },
        "resolution": "0p25",
        "stream": "gfs",
        "product": "gra grap",
        "file_format": "grib2",
    },
    {
        "name": "GRAPES_MESO",
        "description": "中国气象局GRAPES中尺度模式",
        "type": "grib",
        "weight": 0.20,
        "variables": {
            "precip_12h": "tp",
            "precip_1h": "tp",
            "wind_max": "10si",
        },
        "path_pattern": "./data/grapes_meso/{date}/GRAPES_MESO_{date}{cycle:02d}_{fhour:03d}.grb",
        "enabled": True,
        "url_template": "https://data.cma.cn/api/grapes-meso/{resolution}/{date}/{cycle}/grapes_meso_{date}{cycle}_f{fhour}.{file_format}",
        "download": {
            "enabled": True,
            "cache_dir": "./data/GRAPES_MESO/",
            "timeout": 60,
            "retries": 3,
            "backend": "cma_api",
        },
        "resolution": "10km",
        "stream": "meso",
        "product": "grapes_meso",
        "file_format": "grib2",
    },
    {
        "name": "NCEP",
        "description": "美国国家环境预报中心GFS模式",
        "type": "grib",
        "weight": 0.10,
        "variables": {
            "precip_12h": "tp",
            "precip_1h": "tp",
            "wind_max": "10si",
        },
        "path_pattern": "./data/ncep/{date}/NCEP_{date}{cycle:02d}_{fhour:03d}.grb",
        "enabled": True,
        "url_template": "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.{resolution}.f{fhour}",
        "download": {
            "enabled": True,
            "cache_dir": "./data/NCEP/",
            "timeout": 60,
            "retries": 3,
            "backend": "http",
        },
        "resolution": "0p25",
        "stream": "atmos",
        "product": "gfs",
        "file_format": "grib2",
    },
    {
        "name": "OBS",
        "description": "实况观测（用于偏差校正）",
        "type": "text",
        "weight": 0.10,
        "variables": {
            "precip_12h": "precip_12h",
            "precip_1h": "precip_1h",
            "wind_max": "wind_max",
        },
        "path_pattern": "./data/obs/{date}/OBS_{date}.txt",
        "enabled": True,
        "url_template": "https://data.rda.ucar.edu/ds083.2/{year}/{month}/fnl_{date}_{cycle}_00.grib2",
        "download": {
            "enabled": True,
            "cache_dir": "./data/OBS/",
            "timeout": 60,
            "retries": 3,
            "backend": "http",
        },
        "resolution": "1deg",
        "stream": "fnl",
        "product": "analysis",
        "file_format": "grib2",
    },
]

# ============================================================================
# 预报配置
# ============================================================================
FORECAST_CONFIG = {
    # 起报时刻（08时和20时，北京时间）
    "cycles": [8, 20],
    # 预报时效（12小时累计，所以预报12小时）
    "forecast_hours": 12,
    # 需要输出的变量
    "variables": [
        "precip_12h",   # 12小时累计降水量 (mm)
        "precip_1h_max",  # 最大小时降水量 (mm/h)
        "wind_max",      # 最大风力等级 (蒲福风级)
    ],
    # 融合方法
    # 可选: 'weighted_average', 'ensemble_mean', 'ensemble_median',
    #      'bias_corrected', 'object_based', 'kalman_filter'
    "fusion_method": "weighted_average",
    # 是否对缺失数据源进行权重自动重归一化
    "auto_renormalize_weights": True,
    # 风力等级阈值 (蒲福风级，m/s)
    "wind_force_thresholds": [
        0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2,
        20.8, 24.5, 28.5, 32.7, 999.0
    ],
    # 插值方法
    # 可选: 'nearest', 'bilinear', 'bicubic'
    "interpolation_method": "bilinear",
    # 搜索半径（度）
    "search_radius": 1.0,
    # 数据缓存目录
    "data_cache_dir": "./data",
    # 缓存文件最大保留时间（小时）
    "max_cache_age_hours": 24,
    # 下载超时时间（秒）
    "download_timeout": 60,
    # 下载重试次数
    "download_retries": 3,
}

# ============================================================================
# 输出配置
# ============================================================================
OUTPUT_CONFIG = {
    # 数据缓存根目录
    "data_cache_dir": "./data",
    # 输出目录
    "output_dir": "./output",
    # 输出格式列表
    # 可选: 'txt', 'csv', 'json', 'html'
    "formats": ["txt", "csv", "json"],
    # 是否生成可视化图
    "generate_plot": True,
    # 精度控制
    "precision": {
        "precip_12h": 1,
        "precip_1h_max": 1,
        "wind_max": 0,
    },
    # 变量单位
    "units": {
        "precip_12h": "mm",
        "precip_1h_max": "mm/h",
        "wind_max": "级",
    },
}


# ============================================================================
# 辅助函数
# ============================================================================
def build_url(source_config: Dict, forecast_date: str, cycle: int, forecast_hours: int, variable: str = "") -> str:
    defaults = {
        "date": forecast_date,
        "cycle": f"{cycle:02d}",
        "fhour": f"{forecast_hours:03d}",
        "variable": variable if variable else source_config.get("variable", ""),
        "resolution": source_config.get("resolution", ""),
        "stream": source_config.get("stream", ""),
        "product": source_config.get("product", ""),
        "file_format": source_config.get("file_format", ""),
    }
    url = source_config.get("url_template", "")
    for key, value in defaults.items():
        url = url.replace("{" + key + "}", str(value))
    return url


def get_cache_path(source_config: Dict, forecast_date: str, cycle: int, forecast_hours: int) -> str:
    cache_dir = source_config.get("download", {}).get("cache_dir", "./data")
    os.makedirs(cache_dir, exist_ok=True)
    name = source_config.get("name", "data")
    file_format = source_config.get("file_format", "bin")
    filename = f"{name}_{forecast_date}_t{cycle:02d}z_f{forecast_hours:03d}.{file_format}"
    return os.path.join(cache_dir, filename)
