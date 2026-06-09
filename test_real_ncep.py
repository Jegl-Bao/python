"""测试 NCEP Reader 真实数据下载+解析流程"""
import logging, time, os
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')

from data_reader import create_reader
from config import DATA_SOURCES, INNER_MONGOLIA_CITIES

cfg = [s for s in DATA_SOURCES if s['name'] == 'NCEP'][0]
reader = create_reader(cfg, 'bilinear')
print(f'Reader: {type(reader).__name__} | product: {reader.product} | model: {reader.model}')
print(f'URL template: {reader.url_template[:80]}...')

cities = [{'name_short': c['name'], 'lon': c['lon'], 'lat': c['lat']}
          for c in INNER_MONGOLIA_CITIES]

filepath = './data/ncep/20260609/NCEP_2026060900_012.grb'

print(f'\n=== 测试降水 (tp) ===')
t0 = time.time()
res = reader.read_variable(filepath, 'tp', cities)
print(f'  用时: {time.time() - t0:.1f}s')
for name, val in res.items():
    print(f'  {name}: {val}')

print(f'\n=== 测试风速 (10si) ===')
res2 = reader.read_variable(filepath, '10si', cities)
for name, val in res2.items():
    print(f'  {name}: {val}')

# 列出缓存目录
cache_dir = os.path.dirname(filepath)
if os.path.isdir(cache_dir):
    print(f'\n缓存目录 ({cache_dir}):')
    for f in sorted(os.listdir(cache_dir)):
        fp = os.path.join(cache_dir, f)
        print(f'  {f} ({os.path.getsize(fp)//1024} KB)')
