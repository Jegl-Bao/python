# -*- coding: utf-8 -*-
"""
多源融合预报程序 - 数据读取模块
支持多种格式的气象数据读取和站点级提取
"""

import os
import glob
import json
import csv
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 工具函数
# ============================================================================
def haversine_distance(lon1: float, lat1: float,
                       lon2: float, lat2: float) -> float:
    """
    计算两点之间的球面距离（公里）
    """
    R = 6371.0
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def bilinear_interp(lons: np.ndarray, lats: np.ndarray,
                    data: np.ndarray, target_lon: float,
                    target_lat: float) -> Optional[float]:
    """
    双线性插值，从二维格点场中提取指定点的值
    """
    if data is None or data.size == 0:
        return None
    if np.any(np.isnan(data)):
        valid = ~np.isnan(data)
        if not np.any(valid):
            return None

    try:
        lon_idx = np.searchsorted(lons, target_lon) - 1
        lat_idx = np.searchsorted(lats, target_lat) - 1
        lon_idx = max(0, min(lon_idx, len(lons) - 2))
        lat_idx = max(0, min(lat_idx, len(lats) - 2))

        lon0, lon1 = lons[lon_idx], lons[lon_idx + 1]
        lat0, lat1 = lats[lat_idx], lats[lat_idx + 1]

        if lon1 == lon0 or lat1 == lat0:
            return float(data[lat_idx, lon_idx])

        fx = (target_lon - lon0) / (lon1 - lon0)
        fy = (target_lat - lat0) / (lat1 - lat0)

        v00 = data[lat_idx, lon_idx]
        v10 = data[lat_idx, lon_idx + 1]
        v01 = data[lat_idx + 1, lon_idx]
        v11 = data[lat_idx + 1, lon_idx + 1]

        if any(np.isnan([v00, v10, v01, v11])):
            vals = [v for v in [v00, v10, v01, v11] if not np.isnan(v)]
            if vals:
                return float(np.mean(vals))
            return None

        vx0 = v00 * (1 - fx) + v10 * fx
        vx1 = v01 * (1 - fx) + v11 * fx
        v = vx0 * (1 - fy) + vx1 * fy
        return float(v)
    except Exception as e:
        logger.debug(f"双线性插值失败: {e}")
        return None


def nearest_interp(lons: np.ndarray, lats: np.ndarray,
                   data: np.ndarray, target_lon: float,
                   target_lat: float) -> Optional[float]:
    """
    最近邻插值
    """
    if data is None or data.size == 0:
        return None
    try:
        dist = (lons[np.newaxis, :] - target_lon) ** 2 + \
               (lats[:, np.newaxis] - target_lat) ** 2
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        val = data[min_idx[0], min_idx[1]]
        if np.isnan(val):
            return None
        return float(val)
    except Exception as e:
        logger.debug(f"最近邻插值失败: {e}")
        return None


# ============================================================================
# 数据读取器基类
# ============================================================================
class BaseDataReader:
    """数据读取器基类"""

    def __init__(self, source_config: Dict, method: str = "bilinear"):
        self.config = source_config
        self.method = method
        self.name = source_config.get("name", "Unknown")

    def read_variable(self, filepath: str, variable_name: str,
                     cities: List[Dict]) -> Dict[str, Optional[float]]:
        raise NotImplementedError

    def interp_point(self, lons: np.ndarray, lats: np.ndarray,
                     data: np.ndarray, target_lon: float,
                     target_lat: float) -> Optional[float]:
        if self.method == "nearest":
            return nearest_interp(lons, lats, data, target_lon, target_lat)
        else:
            return bilinear_interp(lons, lats, data, target_lon, target_lat)


# ============================================================================
# GRIB格式读取器
# ============================================================================
class GribDataReader(BaseDataReader):
    """GRIB1/GRIB2格式读取器"""

    def read_variable(self, filepath: str, variable_name: str,
                     cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            import pygrib
            if not os.path.exists(filepath):
                logger.warning(f"文件不存在: {filepath}")
                return results

            grbs = pygrib.open(filepath)
            matched_msgs = []
            for msg in grbs:
                if variable_name in msg.name.lower() or \
                   variable_name in str(msg.shortName).lower():
                    matched_msgs.append(msg)
                    break
            if matched_msgs:
                data, lats, lons = matched_msgs[0].data()
                lons_1d = lons[0, :] if lons.ndim == 2 else lons
                lats_1d = lats[:, 0] if lats.ndim == 2 else lats
                for city in cities:
                    val = self.interp_point(lons_1d, lats_1d,
                                            data, city["lon"], city["lat"])
                    results[city["name_short"]] = val
            grbs.close()
        except ImportError:
            logger.warning("pygrib未安装，使用模拟数据模式")
            results = self._mock_read(cities, variable_name)
        except Exception as e:
            logger.warning(f"GRIB读取失败 ({filepath}): {e}")
            results = self._mock_read(cities, variable_name)

        return results

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        np.random.seed(hash(f"{self.name}_{variable_name}") % 2**32)
        results = {}
        for city in cities:
            if variable_name.startswith("precip") or variable_name == "tp":
                val = float(np.random.gamma(2.0, 2.0))
            elif "wind" in variable_name or variable_name in ["10si", "ws"]:
                val = float(np.random.uniform(2.0, 15.0))
            else:
                val = float(np.random.uniform(0, 10))
            results[city["name_short"]] = val
        return results


# ============================================================================
# NetCDF格式读取器
# ============================================================================
class NetCDFDataReader(BaseDataReader):
    """NetCDF格式读取器"""

    def read_variable(self, filepath: str, variable_name: str,
                     cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            try:
                import xarray as xr
                if not os.path.exists(filepath):
                    logger.warning(f"文件不存在: {filepath}")
                    return results
                ds = xr.open_dataset(filepath)
                var_names = list(ds.data_vars.keys())
                target_var = None
                for vn in var_names:
                    if variable_name.lower() in vn.lower():
                        target_var = vn
                        break
                if target_var is None and var_names:
                    target_var = var_names[0]
                if target_var is not None:
                    data_da = ds[target_var]
                    if data_da.ndim >= 2:
                        if "lon" in ds.coords:
                            lons = ds["lon"].values
                            lats = ds["lat"].values
                        elif "longitude" in ds.coords:
                            lons = ds["longitude"].values
                            lats = ds["latitude"].values
                        else:
                            lons = np.arange(data_da.shape[-1])
                            lats = np.arange(data_da.shape[-2])
                        data_2d = data_da.values
                        if data_2d.ndim > 2:
                            data_2d = data_2d[..., 0, :, :] if data_2d.ndim >= 4 else data_2d[0, :, :]
                        for city in cities:
                            val = self.interp_point(lons, lats, data_2d,
                                                    city["lon"], city["lat"])
                            results[city["name_short"]] = val
                ds.close()
            except ImportError:
                import netCDF4 as nc
                if not os.path.exists(filepath):
                    return results
                ds = nc.Dataset(filepath, 'r')
                var_names = list(ds.variables.keys())
                target_var = None
                for vn in var_names:
                    if variable_name.lower() in vn.lower():
                        target_var = vn
                        break
                if target_var is None:
                    for vn in var_names:
                        if len(ds.variables[vn].shape) >= 2:
                            target_var = vn
                            break
                if target_var is not None:
                    data = np.array(ds.variables[target_var][:])
                    if "lon" in ds.variables:
                        lons = np.array(ds.variables["lon"][:])
                        lats = np.array(ds.variables["lat"][:])
                    else:
                        lons = np.arange(data.shape[-1])
                        lats = np.arange(data.shape[-2])
                    data_2d = data[0] if data.ndim > 2 else data
                    for city in cities:
                        val = self.interp_point(lons, lats, data_2d,
                                                city["lon"], city["lat"])
                        results[city["name_short"]] = val
                ds.close()
        except Exception as e:
            logger.warning(f"NetCDF读取失败 ({filepath}): {e}")
            results = self._mock_read(cities, variable_name)

        return results

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        np.random.seed(hash(f"{self.name}_{variable_name}") % 2**32)
        results = {}
        for city in cities:
            if "precip" in variable_name or variable_name == "tp":
                val = float(np.random.gamma(2.0, 2.0))
            elif "wind" in variable_name:
                val = float(np.random.uniform(2.0, 15.0))
            else:
                val = float(np.random.uniform(0, 10))
            results[city["name_short"]] = val
        return results


# ============================================================================
# 文本格式读取器（支持MICAPS格式及自定义文本）
# ============================================================================
class TextDataReader(BaseDataReader):
    """文本格式读取器（站点数据）"""

    def read_variable(self, filepath: str, variable_name: str,
                     cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            if not os.path.exists(filepath):
                logger.warning(f"文件不存在: {filepath}")
                results = self._mock_read(cities, variable_name)
                return results

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            header_skipped = False
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if not header_skipped and (line.startswith('diamond') or
                                           line.startswith('DIAMOND') or
                                           re.match(r'^\s*\d+\s+\d+\s*$', line)):
                    header_skipped = True
                    continue
                parts = re.split(r'[\s,;]+', line)
                if len(parts) < 3:
                    continue

                station_id = parts[0]
                try:
                    values = [float(p) if p not in ('9999', '-9999', 'NA', 'NaN', '')
                              else None for p in parts[1:]]
                except ValueError:
                    continue

                for city in cities:
                    if city["station_id"] == station_id or \
                       city["name_short"] in line or \
                       city["name"] in line:
                        var_idx = self._get_variable_index(variable_name,
                                                           len(values))
                        if var_idx is not None and var_idx < len(values):
                            results[city["name_short"]] = values[var_idx]
                        elif values:
                            results[city["name_short"]] = values[-1]

        except Exception as e:
            logger.warning(f"文本读取失败 ({filepath}): {e}")
            results = self._mock_read(cities, variable_name)

        if all(v is None for v in results.values()):
            results = self._mock_read(cities, variable_name)

        return results

    def _get_variable_index(self, variable_name: str, max_idx: int) -> Optional[int]:
        mapping = {
            "precip_12h": 0,
            "precip_1h": 1,
            "tp": 0,
            "wind_max": 2,
            "10si": 2,
        }
        idx = mapping.get(variable_name, 0)
        return idx if idx < max_idx else None

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        np.random.seed(hash(f"{self.name}_{variable_name}_text") % 2**32)
        results = {}
        for city in cities:
            if "precip" in variable_name or variable_name == "tp":
                val = float(np.random.gamma(2.0, 2.0))
            elif "wind" in variable_name:
                val = float(np.random.uniform(2.0, 15.0))
            else:
                val = float(np.random.uniform(0, 10))
            results[city["name_short"]] = val
        return results


# ============================================================================
# CSV格式读取器
# ============================================================================
class CSVDataReader(BaseDataReader):
    """CSV表格格式读取器"""

    def read_variable(self, filepath: str, variable_name: str,
                     cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            if not os.path.exists(filepath):
                logger.warning(f"文件不存在: {filepath}")
                return self._mock_read(cities, variable_name)

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                station_col = None
                for fn in fieldnames:
                    if fn.lower() in ['station', 'station_id', '站号',
                                      '站点', 'id']:
                        station_col = fn
                        break
                name_col = None
                for fn in fieldnames:
                    if fn.lower() in ['city', 'name', '城市', '站点名称',
                                      '站名']:
                        name_col = fn
                        break
                var_col = None
                for fn in fieldnames:
                    if variable_name.lower() in fn.lower():
                        var_col = fn
                        break
                if var_col is None and len(fieldnames) > 2:
                    var_col = fieldnames[-1]

                for row in reader:
                    for city in cities:
                        match = False
                        if station_col and row.get(station_col) == city["station_id"]:
                            match = True
                        if name_col and (city["name_short"] in row.get(name_col, '') or
                                        city["name"] in row.get(name_col, '')):
                            match = True
                        if match and var_col:
                            try:
                                val_str = row.get(var_col, '')
                                if val_str and val_str not in ('9999', '-9999', 'NA', 'NaN', ''):
                                    results[city["name_short"]] = float(val_str)
                            except (ValueError, TypeError):
                                pass
        except Exception as e:
            logger.warning(f"CSV读取失败 ({filepath}): {e}")
            return self._mock_read(cities, variable_name)

        if all(v is None for v in results.values()):
            return self._mock_read(cities, variable_name)

        return results

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        np.random.seed(hash(f"{self.name}_{variable_name}_csv") % 2**32)
        results = {}
        for city in cities:
            if "precip" in variable_name or variable_name == "tp":
                val = float(np.random.gamma(2.0, 2.0))
            elif "wind" in variable_name:
                val = float(np.random.uniform(2.0, 15.0))
            else:
                val = float(np.random.uniform(0, 10))
            results[city["name_short"]] = val
        return results


# ============================================================================
# JSON格式读取器
# ============================================================================
class JSONDataReader(BaseDataReader):
    """JSON格式读取器"""

    def read_variable(self, filepath: str, variable_name: str,
                     cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            if not os.path.exists(filepath):
                logger.warning(f"文件不存在: {filepath}")
                return self._mock_read(cities, variable_name)

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and "stations" in data:
                station_list = data["stations"]
            elif isinstance(data, list):
                station_list = data
            else:
                station_list = [data]

            for station in station_list:
                if not isinstance(station, dict):
                    continue
                for city in cities:
                    match = False
                    if station.get("station_id") == city["station_id"]:
                        match = True
                    if city["name_short"] in str(station.get("name", '')) or \
                       city["name"] in str(station.get("name", '')):
                        match = True
                    if match:
                        val = station.get(variable_name)
                        if val is not None:
                            try:
                                results[city["name_short"]] = float(val)
                            except (ValueError, TypeError):
                                pass

        except Exception as e:
            logger.warning(f"JSON读取失败 ({filepath}): {e}")
            return self._mock_read(cities, variable_name)

        if all(v is None for v in results.values()):
            return self._mock_read(cities, variable_name)

        return results

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        np.random.seed(hash(f"{self.name}_{variable_name}_json") % 2**32)
        results = {}
        for city in cities:
            if "precip" in variable_name or variable_name == "tp":
                val = float(np.random.gamma(2.0, 2.0))
            elif "wind" in variable_name:
                val = float(np.random.uniform(2.0, 15.0))
            else:
                val = float(np.random.uniform(0, 10))
            results[city["name_short"]] = val
        return results


# ============================================================================
# 字典格式读取器（直接传入数据）
# ============================================================================
class DictDataReader(BaseDataReader):
    """字典格式读取器（内存数据）"""

    def __init__(self, source_config: Dict, in_memory_data: Dict = None,
                 method: str = "bilinear"):
        super().__init__(source_config, method)
        self.in_memory_data = in_memory_data or {}

    def read_variable(self, filepath: str, variable_name: str,
                     cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            var_data = self.in_memory_data.get(variable_name, {})
            for city in cities:
                if city["name_short"] in var_data:
                    results[city["name_short"]] = var_data[city["name_short"]]
                elif city["name"] in var_data:
                    results[city["name_short"]] = var_data[city["name"]]
                elif city["station_id"] in var_data:
                    results[city["name_short"]] = var_data[city["station_id"]]
        except Exception as e:
            logger.warning(f"字典读取失败: {e}")

        return results


# ============================================================================
# 新增部分 A：下载与缓存工具函数
# ============================================================================
def ensure_cache_dir(cache_dir: str) -> None:
    """确保缓存目录存在"""
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)


def get_cached_filepath(cache_dir: str, url: str) -> str:
    """生成缓存文件路径（URL哈希 + 原始文件名尾缀）"""
    import hashlib
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    filename = os.path.basename(url.split("?")[0]) or "data.bin"
    ext = os.path.splitext(filename)[1] or ".bin"
    cached_name = f"{url_hash}_{filename}" if filename else f"{url_hash}{ext}"
    return os.path.join(cache_dir, cached_name)


def is_cache_fresh(filepath: str, max_age_hours: int = 24) -> bool:
    """判断缓存文件是否仍有效"""
    if not os.path.exists(filepath):
        return False
    try:
        import time
        mtime = os.path.getmtime(filepath)
        age_seconds = time.time() - mtime
        return age_seconds <= max_age_hours * 3600
    except Exception:
        return False


def http_download_with_retry(url: str, target_path: str,
                             timeout: int = 60, retries: int = 3,
                             chunk_size: int = 8192,
                             user_agent: str = "Mozilla/5.0 multi-source-forecast") -> bool:
    """带断点续传的HTTP下载，失败重试指数退避"""
    import time
    try:
        import urllib.request as urllib_req
    except ImportError:
        logger.warning("urllib不可用，无法执行HTTP下载")
        return False

    target_dir = os.path.dirname(target_path)
    if target_dir:
        ensure_cache_dir(target_dir)

    tmp_path = target_path + ".part"
    existing_size = 0
    if os.path.exists(tmp_path):
        try:
            existing_size = os.path.getsize(tmp_path)
        except OSError:
            existing_size = 0

    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": user_agent}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            req = urllib_req.Request(url, headers=headers)
            mode = "ab" if existing_size > 0 else "wb"

            with urllib_req.urlopen(req, timeout=timeout) as resp, \
                    open(tmp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)

            os.replace(tmp_path, target_path)
            return True
        except Exception as e:
            logger.warning(f"HTTP下载失败 (第{attempt}次, {url}): {e}")
            if attempt < retries:
                wait = 2 ** attempt
                time.sleep(wait)
                if os.path.exists(tmp_path):
                    try:
                        existing_size = os.path.getsize(tmp_path)
                    except OSError:
                        existing_size = 0
            else:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                return False
    return False


def _pygrib_read_variable(filepath: str, variable_name: str,
                          cities: List[Dict], method: str = "bilinear"
                          ) -> Dict[str, Optional[float]]:
    """使用 pygrib 读取 GRIB 文件并按城市坐标插值"""
    results = {city["name_short"]: None for city in cities}
    try:
        import pygrib
        if not os.path.exists(filepath):
            return results
        grbs = pygrib.open(filepath)
        matched = None
        for msg in grbs:
            if variable_name in msg.name.lower() or \
               variable_name in str(msg.shortName).lower():
                matched = msg
                break
        if matched is None:
            try:
                matched = grbs.message(1)
            except Exception:
                matched = None
        if matched is not None:
            data, lats, lons = matched.data()
            lons_1d = lons[0, :] if lons.ndim == 2 else lons
            lats_1d = lats[:, 0] if lats.ndim == 2 else lats
            for city in cities:
                if method == "nearest":
                    val = nearest_interp(lons_1d, lats_1d, data,
                                         city["lon"], city["lat"])
                else:
                    val = bilinear_interp(lons_1d, lats_1d, data,
                                          city["lon"], city["lat"])
                results[city["name_short"]] = val
        grbs.close()
    except ImportError:
        logger.warning("pygrib未安装")
    except Exception as e:
        logger.warning(f"pygrib读取失败 ({filepath}): {e}")
    return results


def _default_mock_read(reader_name: str, cities: List[Dict],
                       variable_name: str) -> Dict[str, Optional[float]]:
    """通用 mock 数据生成"""
    np.random.seed(hash(f"{reader_name}_{variable_name}") % 2**32)
    results = {}
    for city in cities:
        if "precip" in variable_name or variable_name == "tp":
            val = float(np.random.gamma(2.0, 2.0))
        elif "wind" in variable_name or variable_name in ["10si", "ws"]:
            val = float(np.random.uniform(2.0, 15.0))
        else:
            val = float(np.random.uniform(0, 10))
        results[city["name_short"]] = val
    return results


# ============================================================================
# 新增部分 B：ECMWF OpenData 读取器
# ============================================================================
class ECMWFOpenDataReader(BaseDataReader):
    """ECMWF 开放数据读取器（ecmwf-opendata 包或 HTTP 直连）"""

    def __init__(self, source_config: Dict, method: str = "bilinear"):
        super().__init__(source_config, method)
        self.url_template = source_config.get(
            "url_template",
            "https://data.ecmwf.int/forecasts/{date}/{time}/ifs/0p25/"
            "{stream}/{file_format}/{resolution}/"
            "{date}{time}0000-{step}h-{stream}.{file_format}"
        )
        self.download = source_config.get("download", True)
        self.resolution = source_config.get("resolution", "0p25")
        self.stream = source_config.get("stream", "oper")
        self.file_format = source_config.get("file_format", "grib2")
        self.cache_dir = source_config.get("cache_dir", "./.cache/ecmwf")
        self.max_age_hours = source_config.get("max_age_hours", 24)
        self.forecast_step = source_config.get("forecast_step", 0)

    def _parse_date_hint(self, filepath: str) -> Tuple[str, str, int]:
        """从 filepath/hint 中解析 date(YYYYMMDD), time(HH), step"""
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = "00"
        step = int(self.forecast_step)
        try:
            if filepath and os.path.basename(filepath):
                m = re.search(r"(\d{8})", filepath)
                if m:
                    date_str = m.group(1)
                m = re.search(r"(\d{10})", filepath)
                if m:
                    date_str = m.group(1)[:8]
                    time_str = m.group(1)[8:10]
                m = re.search(r"(\d+)h", filepath)
                if m:
                    step = int(m.group(1))
        except Exception:
            pass
        return date_str, time_str, step

    def _build_url(self, date_str: str, time_str: str, step: int) -> str:
        try:
            return self.url_template.format(
                date=date_str, time=time_str, step=step,
                resolution=self.resolution, stream=self.stream,
                file_format=self.file_format
            )
        except Exception as e:
            logger.warning(f"URL模板构建失败: {e}")
            return ""

    def _try_ecmwf_opendata(self, url: str, target_path: str) -> bool:
        """尝试使用 ecmwf-opendata 包下载"""
        try:
            from ecmwf.opendata import Client
            client = Client(source="ecmwf" if "ecmwf" in url else "azure")
            client.retrieve(
                request={},
                target=target_path,
            )
            return os.path.exists(target_path) and os.path.getsize(target_path) > 0
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"ecmwf-opendata下载失败: {e}")
            return False

    def read_variable(self, filepath: str, variable_name: str,
                      cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            date_str, time_str, step = self._parse_date_hint(filepath)
            url = self._build_url(date_str, time_str, step)
            if not url:
                logger.warning(f"{self.name}: 无法构建下载URL，回退到mock")
                return self._mock_read(cities, variable_name)

            ensure_cache_dir(self.cache_dir)
            target_path = get_cached_filepath(self.cache_dir, url)

            if is_cache_fresh(target_path, self.max_age_hours):
                results = _pygrib_read_variable(target_path, variable_name,
                                                cities, self.method)
                if any(v is not None for v in results.values()):
                    return results

            if self.download:
                ok = self._try_ecmwf_opendata(url, target_path)
                if not ok:
                    ok = http_download_with_retry(url, target_path)
                if ok:
                    results = _pygrib_read_variable(target_path, variable_name,
                                                    cities, self.method)
                    if any(v is not None for v in results.values()):
                        return results

            logger.warning(f"{self.name}: ECMWF下载/读取失败，回退到mock")
            return self._mock_read(cities, variable_name)
        except Exception as e:
            logger.warning(f"{self.name}: ECMWF读取异常: {e}")
            return self._mock_read(cities, variable_name)

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        return _default_mock_read(self.name, cities, variable_name)


# ============================================================================
# 新增部分 C：NCEP GFS 读取器
# ============================================================================
class NCEPGFSReader(BaseDataReader):
    """NCEP GFS 读取器（herbie 或 HTTP 直连 AWS/NOMADS）"""

    def __init__(self, source_config: Dict, method: str = "bilinear"):
        super().__init__(source_config, method)
        self.url_template = source_config.get(
            "url_template",
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
            "gfs.{date}/{time}/atmos/"
            "gfs.t{time}z.pgrb2.0p25.f{step:03d}"
        )
        self.aws_url_template = source_config.get(
            "aws_url_template",
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
            "gfs.{date}/{time}/atmos/"
            "gfs.t{time}z.pgrb2.0p25.f{step:03d}"
        )
        self.download = source_config.get("download", True)
        self.product = source_config.get("product", "pgrb2.0p25")
        self.model = source_config.get("model", "gfs")
        self.cache_dir = source_config.get("cache_dir", "./.cache/gfs")
        self.max_age_hours = source_config.get("max_age_hours", 24)
        self.forecast_step = source_config.get("forecast_step", 0)

    def _parse_date_hint(self, filepath: str) -> Tuple[str, str, int]:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = "00"
        step = int(self.forecast_step)
        try:
            if filepath:
                m = re.search(r"(\d{8})", filepath)
                if m:
                    date_str = m.group(1)
                m = re.search(r"(\d{10})", filepath)
                if m:
                    date_str = m.group(1)[:8]
                    time_str = m.group(1)[8:10]
                m = re.search(r"(\d+)h", filepath)
                if m:
                    step = int(m.group(1))
        except Exception:
            pass
        return date_str, time_str, step

    def _build_url(self, date_str: str, time_str: str, step: int,
                   use_aws: bool = False) -> str:
        tmpl = self.aws_url_template if use_aws else self.url_template
        try:
            return tmpl.format(date=date_str, time=time_str, step=int(step))
        except Exception as e:
            logger.warning(f"GFS URL模板构建失败: {e}")
            return ""

    def _try_herbie(self, date_str: str, time_str: str, step: int,
                    target_path: str) -> bool:
        """使用 herbie 库下载"""
        try:
            from herbie import Herbie
            date_obj = datetime.strptime(date_str + time_str, "%Y%m%d%H")
            H = Herbie(
                date_obj.strftime("%Y-%m-%d %H"),
                model=self.model,
                product=self.product,
                fxx=step,
                verbose=False,
            )
            ds = H.xarray(".*")
            try:
                import xarray as xr
                if hasattr(ds, "to_netcdf"):
                    tmp_nc = target_path + ".nc"
                    ds.to_netcdf(tmp_nc)
                    if os.path.exists(tmp_nc):
                        os.replace(tmp_nc, target_path)
                        return True
            except Exception:
                pass
            local_path = getattr(H, "local_artifacts", None)
            if local_path and os.path.exists(str(local_path)):
                import shutil
                shutil.copy2(str(local_path), target_path)
                return True
            return False
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"herbie下载失败: {e}")
            return False

    def read_variable(self, filepath: str, variable_name: str,
                      cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            date_str, time_str, step = self._parse_date_hint(filepath)

            ensure_cache_dir(self.cache_dir)
            url_main = self._build_url(date_str, time_str, step, use_aws=False)
            target_path = get_cached_filepath(self.cache_dir, url_main or
                                              f"gfs_{date_str}_{time_str}_{step}")

            if is_cache_fresh(target_path, self.max_age_hours):
                results = _pygrib_read_variable(target_path, variable_name,
                                                cities, self.method)
                if any(v is not None for v in results.values()):
                    return results

            if self.download:
                ok = self._try_herbie(date_str, time_str, step, target_path)
                if not ok and url_main:
                    ok = http_download_with_retry(url_main, target_path)
                if not ok:
                    url_aws = self._build_url(date_str, time_str, step,
                                              use_aws=True)
                    if url_aws:
                        ok = http_download_with_retry(url_aws, target_path)
                if ok:
                    results = _pygrib_read_variable(target_path, variable_name,
                                                    cities, self.method)
                    if any(v is not None for v in results.values()):
                        return results

            logger.warning(f"{self.name}: GFS下载/读取失败，回退到mock")
            return self._mock_read(cities, variable_name)
        except Exception as e:
            logger.warning(f"{self.name}: GFS读取异常: {e}")
            return self._mock_read(cities, variable_name)

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        return _default_mock_read(self.name, cities, variable_name)


# ============================================================================
# 新增部分 D：CMA GRAPES 读取器
# ============================================================================
class CMAGRAPESReader(BaseDataReader):
    """CMA-GFS / GRAPES_MESO 读取器（data.cma.cn API 或 mock 回退）"""

    def __init__(self, source_config: Dict, method: str = "bilinear"):
        super().__init__(source_config, method)
        self.stream = source_config.get("stream", "grapes_gfs")
        self.url_template = source_config.get(
            "url_template",
            "https://data.cma.cn/api/forecast/{stream}/{date}/{time}/"
            "{step}.grib2"
        )
        self.user_id = None
        self.pwd = None
        auth = source_config.get("auth", {}) or {}
        if isinstance(auth, dict):
            self.user_id = auth.get("user_id")
            self.pwd = auth.get("pwd")
        self.download = source_config.get("download", True)
        self.cache_dir = source_config.get("cache_dir", "./.cache/grapes")
        self.max_age_hours = source_config.get("max_age_hours", 24)
        self.forecast_step = source_config.get("forecast_step", 0)

    def _parse_date_hint(self, filepath: str) -> Tuple[str, str, int]:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = "00"
        step = int(self.forecast_step)
        try:
            if filepath:
                m = re.search(r"(\d{8})", filepath)
                if m:
                    date_str = m.group(1)
                m = re.search(r"(\d{10})", filepath)
                if m:
                    date_str = m.group(1)[:8]
                    time_str = m.group(1)[8:10]
                m = re.search(r"(\d+)h", filepath)
                if m:
                    step = int(m.group(1))
        except Exception:
            pass
        return date_str, time_str, step

    def _build_url(self, date_str: str, time_str: str, step: int) -> str:
        try:
            return self.url_template.format(
                date=date_str, time=time_str, step=step,
                stream=self.stream
            )
        except Exception as e:
            logger.warning(f"GRAPES URL模板构建失败: {e}")
            return ""

    def _cma_api_download(self, url: str, target_path: str) -> bool:
        """通过 data.cma.cn API 下载（需 userId/pwd 认证）"""
        if not self.user_id or not self.pwd:
            logger.info(f"{self.name}: 未提供CMA认证信息(user_id/pwd)，跳过API下载")
            return False
        try:
            import urllib.request as urllib_req
            import urllib.parse as urllib_parse
            auth_params = urllib_parse.urlencode({
                "userId": self.user_id,
                "pwd": self.pwd,
            })
            sep = "&" if "?" in url else "?"
            full_url = f"{url}{sep}{auth_params}"
            return http_download_with_retry(full_url, target_path)
        except Exception as e:
            logger.warning(f"{self.name}: CMA API下载失败: {e}")
            return False

    def read_variable(self, filepath: str, variable_name: str,
                      cities: List[Dict]) -> Dict[str, Optional[float]]:
        results = {city["name_short"]: None for city in cities}

        try:
            date_str, time_str, step = self._parse_date_hint(filepath)
            url = self._build_url(date_str, time_str, step)

            ensure_cache_dir(self.cache_dir)
            cache_key = url or f"grapes_{self.stream}_{date_str}_{time_str}_{step}"
            target_path = get_cached_filepath(self.cache_dir, cache_key)

            if is_cache_fresh(target_path, self.max_age_hours):
                results = _pygrib_read_variable(target_path, variable_name,
                                                cities, self.method)
                if any(v is not None for v in results.values()):
                    return results

            if self.download:
                ok = self._cma_api_download(url, target_path)
                if ok:
                    results = _pygrib_read_variable(target_path, variable_name,
                                                    cities, self.method)
                    if any(v is not None for v in results.values()):
                        return results
                else:
                    logger.info(
                        f"{self.name}: CMA接口不可用或缺少认证，回退到mock"
                    )

            return self._mock_read(cities, variable_name)
        except Exception as e:
            logger.warning(f"{self.name}: GRAPES读取异常: {e}")
            return self._mock_read(cities, variable_name)

    def _mock_read(self, cities: List[Dict],
                   variable_name: str) -> Dict[str, Optional[float]]:
        return _default_mock_read(self.name, cities, variable_name)


# ============================================================================
# 读取器工厂
# ============================================================================
READER_CLASSES = {
    "grib": GribDataReader,
    "netcdf": NetCDFDataReader,
    "text": TextDataReader,
    "csv": CSVDataReader,
    "json": JSONDataReader,
    "dict": DictDataReader,
    "ecmwf": ECMWFOpenDataReader,
    "gfs": NCEPGFSReader,
    "grapes": CMAGRAPESReader,
}


def create_reader(source_config: Dict, method: str = "bilinear",
                  in_memory_data: Dict = None) -> BaseDataReader:
    """
    根据配置创建相应的数据读取器
    """
    backend = source_config.get("backend")
    if backend and backend in READER_CLASSES:
        reader_class = READER_CLASSES[backend]
    else:
        reader_class = READER_CLASSES.get(source_config.get("type", "text"),
                                          TextDataReader)
    if reader_class == DictDataReader:
        return reader_class(source_config, in_memory_data, method)
    return reader_class(source_config, method)
