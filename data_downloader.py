# -*- coding: utf-8 -*-
"""
多源融合预报程序 - GRIB2数据批量下载器
从真实气象数据源（ECMWF/GFS/GRAPES等）批量下载GRIB2文件
支持多种后端：HTTP直连、ecmwf-opendata、herbie-data、CMA（中国气象局）
支持断点续传与指数退避重试
"""

import argparse
import logging
import sys
import os
import time
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    import requests
except ImportError:
    requests = None

try:
    import ecmwf.opendata
except ImportError:
    ecmwf_module = None
else:
    ecmwf_module = ecmwf

try:
    from herbie import Herbie
except ImportError:
    Herbie = None

from config import DATA_SOURCES

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("data_downloader")


# ============================================================================
# 各数据源默认URL模板
# 当配置中未提供 url_pattern 时使用以下内置模板
# ============================================================================
DEFAULT_URL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "EC": {
        "backend": "ecmwf",
        "url_pattern": (
            "https://data.ecmwf.int/forecasts/{date}/{cycle:02d}z/"
            "ifs/0p25/oper/"
            "{stream}/{type}/{date}{cycle:02d}0000-{fhour:03d}h-{type}-{stream}.grb2"
        ),
        "stream": "oper",
        "type": "fc",
    },
    "NCEP": {
        "backend": "http",
        "url_pattern": (
            "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
            "file=gfs.t{cycle:02d}z.pgrb2.0p25.f{fhour:03d}"
            "&lev_10_m_above_ground=on&lev_surface=on"
            "&var_APCP=on&var_GUST=on&var_TMAX=on&var_TMIN=on"
            "&leftlon=0&rightlon=360&toplat=90&bottomlat=-90"
            "&dir=%2Fgfs.{date}%2F{cycle:02d}%2Fatmos"
        ),
    },
    "GRAPES": {
        "backend": "cma",
        "url_pattern": (
            "http://data.cma.cn/data_service/dataService/download/"
            "{date}/{cycle:02d}/GRAPES_GFS_{date}{cycle:02d}_{fhour:03d}.grb2"
        ),
    },
    "GRAPES_MESO": {
        "backend": "cma",
        "url_pattern": (
            "http://data.cma.cn/data_service/dataService/download/"
            "{date}/{cycle:02d}/GRAPES_MESO_{date}{cycle:02d}_{fhour:03d}.grb2"
        ),
    },
}


# ============================================================================
# 参数解析
# ============================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GRIB2数据批量下载器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载2025年1月1日00时起报，默认预报时效
  python data_downloader.py --date 20250101

  # 指定起报时刻和预报时效
  python data_downloader.py --date 20250101 --cycle 0 --fhours 12 24

  # 只下载EC和NCEP数据源
  python data_downloader.py --date 20250101 --sources EC NCEP

  # 使用herbie后端下载GFS
  python data_downloader.py --date 20250101 --sources NCEP --backend herbie

  # 忽略缓存，强制重新下载
  python data_downloader.py --date 20250101 --force
        """
    )

    parser.add_argument(
        '--date', '-d', type=str, required=True,
        help='起报日期 (YYYYMMDD格式，必填)'
    )
    parser.add_argument(
        '--cycle', '-c', type=int, nargs='+', default=[0, 6, 12, 18],
        help='起报时刻列表，默认 [0, 6, 12, 18]'
    )
    parser.add_argument(
        '--fhours', '-f', type=int, nargs='+', default=[0, 12, 24, 36, 48, 72],
        help='预报时效列表（小时），默认 [0, 12, 24, 36, 48, 72]'
    )
    parser.add_argument(
        '--sources', '-s', type=str, nargs='+', default=None,
        help='指定数据源名称列表，默认为所有enabled=True的数据源'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default='./data',
        help='输出目录，默认 ./data'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='忽略缓存，强制重新下载'
    )
    parser.add_argument(
        '--backend', type=str, default=None,
        choices=['http', 'ecmwf', 'herbie', 'cma'],
        help='覆盖配置中的下载后端'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='显示详细日志'
    )
    parser.add_argument(
        '--max-retries', type=int, default=5,
        help='最大重试次数，默认 5'
    )
    parser.add_argument(
        '--timeout', type=int, default=180,
        help='单次下载超时（秒），默认 180'
    )

    return parser.parse_args()


# ============================================================================
# 工具函数
# ============================================================================
def http_download_with_retry(
    url: str,
    output_path: str,
    max_retries: int = 5,
    timeout: int = 180,
    force: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    HTTP下载文件，支持断点续传与指数退避重试

    参数:
        url: 下载URL
        output_path: 输出文件路径
        max_retries: 最大重试次数
        timeout: 超时秒数
        force: 是否强制重新下载

    返回:
        (是否成功, 错误信息)
    """
    if requests is None:
        return False, "requests库未安装，无法进行HTTP下载"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 检查缓存
    if os.path.exists(output_path) and not force:
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            logger.debug(f"  缓存命中，跳过: {output_path} "
                         f"({file_size} bytes)")
            return True, None
        # 文件为空，删除后重新下载
        os.remove(output_path)

    # 断点续传：记录已有文件大小
    existing_size = 0
    tmp_path = output_path + ".part"
    if os.path.exists(tmp_path):
        existing_size = os.path.getsize(tmp_path)

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
                logger.debug(f"  尝试断点续传，已下载 {existing_size} bytes")

            mode = "ab" if existing_size > 0 else "wb"

            with requests.get(url, stream=True, timeout=timeout,
                            headers=headers, allow_redirects=True) as r:
                if r.status_code == 416:
                    # Range请求范围无效，可能文件已完整下载
                    logger.debug(f"  范围请求已完成 (HTTP 416)")
                    if os.path.exists(tmp_path):
                        os.rename(tmp_path, output_path)
                    return True, None

                r.raise_for_status()

                total_size = int(r.headers.get("content-length", 0))
                if existing_size > 0 and r.status_code == 200:
                    # 服务器不支持断点续传，从头开始
                    existing_size = 0
                    mode = "wb"

                downloaded = existing_size
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # 检查下载完成
                if total_size > 0 and downloaded < total_size:
                    raise Exception(
                        f"下载不完整: {downloaded}/{total_size} bytes"
                    )

                # 重命名临时文件
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(tmp_path, output_path)

                final_size = os.path.getsize(output_path)
                logger.debug(f"  下载完成: {output_path} "
                             f"({final_size} bytes)")
                return True, None

        except Exception as e:
            wait_time = min(2 ** (attempt - 1), 60)  # 指数退避，最大60秒
            error_msg = str(e)

            if attempt < max_retries:
                logger.warning(
                    f"  第 {attempt}/{max_retries} 次尝试失败: {error_msg}")
                logger.warning(f"  {wait_time} 秒后重试...")
                time.sleep(wait_time)
                # 更新已有文件大小
                if os.path.exists(tmp_path):
                    existing_size = os.path.getsize(tmp_path)
            else:
                logger.error(f"  下载失败 (已尝试 {max_retries} 次): "
                             f"{error_msg}")
                return False, error_msg

    return False, "达到最大重试次数"


def download_via_ecmwf_opendata(
    date_str: str,
    cycle: int,
    fhour: int,
    output_path: str,
    max_retries: int = 5,
    force: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    使用ecmwf-opendata库下载ECMWF公开数据
    """
    if ecmwf_module is None:
        return False, "ecmwf-opendata 库未安装"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not force:
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            logger.debug(f"  缓存命中，跳过: {output_path}")
            return True, None

    for attempt in range(1, max_retries + 1):
        try:
            request = ecmwf_module.Request(
                request={
                    "stream": "oper",
                    "type": "fc",
                    "levtype": "sfc",
                    "param": ["10si", "tp"],
                    "date": f"{date_str}",
                    "time": f"{cycle:02d}",
                    "step": f"{fhour}",
                    "target": output_path,
                },
            )
            request.execute()
            if os.path.exists(output_path):
                return True, None
            return False, "文件未生成"
        except Exception as e:
            wait_time = min(2 ** (attempt - 1), 60)
            if attempt < max_retries:
                logger.warning(f"  第 {attempt}/{max_retries} 次尝试失败: {e}")
                logger.warning(f"  {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"  ECMWF下载失败: {e}")
                return False, str(e)
    return False, "达到最大重试次数"


def download_via_herbie(
    date_str: str,
    cycle: int,
    fhour: int,
    output_path: str,
    source_config: Dict,
    force: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    使用herbie-data库下载GFS等数据
    """
    if Herbie is None:
        return False, "herbie-data 库未安装"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not force:
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            logger.debug(f"  缓存命中，跳过: {output_path}")
            return True, None

    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        full_date = f"{date_obj.strftime('%Y-%m-%d')} {cycle:02d}"

        source_name = source_config.get("name", "")
        if source_name == "NCEP":
            model = "gfs"
        elif source_name in ("EC",):
            model = "ecmwf"
        else:
            model = "gfs"

        H = Herbie(
            full_date,
            model=model,
            fxx=fhour,
            verbose=False,
            save_dir=os.path.dirname(output_path),
            verbose_errors=False,
        )

        # 下载降水和风速变量
        ds = H.xarray(":APCP:|:TP:|:10 si:|:GUST:")
        if ds is not None:
            # 将数据保存为GRIB2
            # 复制herbie下载的原始grib2文件到目标路径
            local_path = H.get_local_path()
            if local_path and os.path.exists(local_path):
                import shutil
                shutil.copy2(local_path, output_path)
                return True, None

        # 如果xarray下载未找到，直接尝试复制原始grib文件
        local_path = H.get_local_path()
        if local_path and os.path.exists(local_path):
            import shutil
            shutil.copy2(local_path, output_path)
            return True, None

        return False, "未找到数据"
    except Exception as e:
        logger.error(f"  Herbie下载失败: {e}")
        return False, str(e)


def download_via_cma(
    url: str,
    output_path: str,
    max_retries: int = 5,
    timeout: int = 180,
    force: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    从中国气象局(CMA)数据源下载，当前通过HTTP实现
    """
    return http_download_with_retry(
        url=url,
        output_path=output_path,
        max_retries=max_retries,
        timeout=timeout,
        force=force,
    )


# ============================================================================
# URL构建
# ============================================================================
def build_url(source_config: Dict, date_str: str,
              cycle: int, fhour: int,
              backend: Optional[str] = None) -> Tuple[str, str]:
    """
    根据配置构建下载URL和确定后端

    返回:
        (url, backend)
    """
    source_name = source_config.get("name", "")

    # 优先使用配置中的 url_pattern
    url_pattern = source_config.get("url_pattern")
    effective_backend = backend or source_config.get(
        "backend"
    ) or DEFAULT_URL_TEMPLATES.get(source_name, {}).get("backend", "http")

    if url_pattern is None:
        # 使用内置模板
        if source_name in DEFAULT_URL_TEMPLATES:
            template_info = DEFAULT_URL_TEMPLATES[source_name]
            url_pattern = template_info["url_pattern"]
        else:
            # 未知数据源，无法下载
            return "", effective_backend

    # 格式化URL
    try:
        url = url_pattern.format(
            date=date_str,
            cycle=cycle,
            fhour=fhour,
            stream=source_config.get("stream", "oper"),
            type=source_config.get("type", "fc"),
            name=source_name,
        )
    except (KeyError, IndexError) as e:
        logger.warning(f"  URL模板格式化失败 ({source_name}): {e}")
        url = url_pattern

    return url, effective_backend


def build_output_path(source_config: Dict, date_str: str,
                       cycle: int, fhour: int,
                       output_dir: str) -> str:
    """
    根据配置构建输出文件路径（使用path_pattern作为相对output_dir内的路径）
    """
    path_pattern = source_config.get(
        "path_pattern",
        "./{name}/{date}/{name}_{date}{cycle:02d}_{fhour:03d}.grb"
    )
    relative_path = path_pattern.format(
        name=source_config.get("name", "UNKNOWN"),
        date=date_str,
        cycle=cycle,
        fhour=fhour,
    )

    # 如果相对路径以 "./" 开头，去掉它以便和output_dir拼接
    if relative_path.startswith("./"):
        relative_path = relative_path[2:]

    return os.path.join(output_dir, relative_path)


# ============================================================================
# 下载单个任务
# ============================================================================
def download_single_task(
    source_config: Dict,
    date_str: str,
    cycle: int,
    fhour: int,
    output_dir: str,
    force: bool,
    max_retries: int,
    timeout: int,
    backend_override: Optional[str],
) -> Tuple[bool, str]:
    """
    下载单个 (source, cycle, fhour) 组合

    返回:
        (是否成功, 输出路径或错误信息)
    """
    source_name = source_config["name"]
    output_path = build_output_path(source_config, date_str, cycle, fhour,
                                    output_dir)

    # 构建URL
    url, effective_backend = build_url(source_config, date_str, cycle, fhour,
                                      backend_override)

    logger.info(f"  [{source_name}] 起报{cycle:02d}时 +{fhour:03d}h "
                 f"(后端: {effective_backend})")
    logger.debug(f"    URL: {url}")
    logger.debug(f"    输出: {output_path}")

    if not url and effective_backend not in ("ecmwf", "herbie"):
        msg = f"未配置URL模板，跳过"
        logger.warning(f"    {msg}")
        return False, msg

    # 根据后端选择下载方式
    success = False
    error = None

    if effective_backend == "ecmwf":
        success, error = download_via_ecmwf_opendata(
            date_str=date_str,
            cycle=cycle,
            fhour=fhour,
            output_path=output_path,
            max_retries=max_retries,
            force=force,
        )
    elif effective_backend == "herbie":
        success, error = download_via_herbie(
            date_str=date_str,
            cycle=cycle,
            fhour=fhour,
            output_path=output_path,
            source_config=source_config,
            force=force,
        )
    elif effective_backend == "cma":
        success, error = download_via_cma(
            url=url,
            output_path=output_path,
            max_retries=max_retries,
            timeout=timeout,
            force=force,
        )
    else:  # http
        success, error = http_download_with_retry(
            url=url,
            output_path=output_path,
            max_retries=max_retries,
            timeout=timeout,
            force=force,
        )

    return success, output_path if success else (error or "未知错误")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """程序入口"""
    args = parse_args()

    # 日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 校验日期格式
    try:
        datetime.strptime(args.date, "%Y%m%d")
        date_str = args.date
    except ValueError:
        logger.error(f"日期格式错误: {args.date}，请使用YYYYMMDD格式")
        sys.exit(1)

    # 选择数据源
    if args.sources:
        selected_sources = []
        for s in args.sources:
            matched = [cfg for cfg in DATA_SOURCES
                       if cfg.get("name") == s]
            if matched:
                selected_sources.append(matched[0])
            else:
                logger.warning(f"未找到数据源: {s}，跳过")
    else:
        selected_sources = [cfg for cfg in DATA_SOURCES
                         if cfg.get("enabled", True)]

    if not selected_sources:
        logger.error("没有可下载的数据源")
        sys.exit(1)

    # 输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 打印任务信息
    logger.info("=" * 60)
    logger.info("GRIB2数据批量下载器启动")
    logger.info(f"起报日期: {date_str}")
    logger.info(f"起报时刻: {args.cycle}")
    logger.info(f"预报时效: {args.fhours}")
    logger.info(f"数据源: {[s['name'] for s in selected_sources]}")
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")
    logger.info(f"后端覆盖: {args.backend or '使用配置默认'}")
    logger.info(f"强制下载: {'是' if args.force else '否'}")
    logger.info(f"最大重试: {args.max_retries}")
    total_tasks = len(selected_sources) * len(args.cycle) * len(args.fhours)
    logger.info(f"总任务数: {total_tasks}")
    logger.info("=" * 60)

    # 执行下载
    success_count = 0
    fail_count = 0
    fail_details: List[str] = []

    for source_cfg in selected_sources:
        source_name = source_cfg["name"]
        logger.info(f"\n[{source_name}] 开始下载...")

        # 检查是否为实况观测数据（无起报时刻/时效不适用）
        if source_name == "OBS":
            logger.info(f"  [{source_name}] 为实况观测数据，"
                         f"跳过（无GRIB2下载")
            continue

        for cycle in args.cycle:
            for fhour in args.fhours:
                success, info = download_single_task(
                    source_config=source_cfg,
                    date_str=date_str,
                    cycle=cycle,
                    fhour=fhour,
                    output_dir=output_dir,
                    force=args.force,
                    max_retries=args.max_retries,
                    timeout=args.timeout,
                    backend_override=args.backend,
                )
                if success:
                    success_count += 1
                    logger.info(f"    ✓ 成功: {info}")
                else:
                    fail_count += 1
                    fail_details.append(
                        f"[{source_name}] cycle={cycle:02d} "
                        f"fhour={fhour:03d} - {info}"
                    )
                    logger.info(f"    ✗ 失败: {info}")

    # 汇总结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("下载任务完成")
    logger.info(f"成功: {success_count} | 失败: {fail_count} | "
                f"总计: {success_count + fail_count}")
    if fail_count > 0:
        logger.info("失败详情:")
        for detail in fail_details:
            logger.info(f"  - {detail}")
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")

    # 列出已下载的文件
    downloaded_files = glob.glob(
        os.path.join(output_dir, "**", "*.grb*"),
        recursive=True,
    )
    logger.info(f"已存在GRIB/GRIB2文件数: {len(downloaded_files)}")
    logger.info("=" * 60)

    return {
        "success": success_count,
        "fail": fail_count,
        "fail_details": fail_details,
    }


if __name__ == "__main__":
    main()
