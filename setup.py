from pathlib import Path
from setuptools import find_packages, setup


share = 'share/naro'
data = [(share, ['package.xml', 'README.md', 'LICENSE', 'NOTICE']),
        ('share/ament_index/resource_index/packages', ['resource/naro'])]
for folder in ('launch', 'config', 'rviz', 'examples', 'third_party'):
    for path in sorted(Path(folder).rglob('*')):
        if path.is_file():
            data.append((str(Path(share) / path.parent), [str(path)]))

setup(
    name='naro',
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    data_files=data,
    install_requires=['setuptools', 'numpy==1.26.4', 'small_gicp==1.0.1'],
    python_requires='>=3.10',
    description='Nano Relocalization: lightweight LiDAR relocalization',
    maintainer='Zhiyou Guan',
    maintainer_email='contact@zhiyouguan.com',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'node = naro.node:main',
        'prepare_map = naro.map_tools:main',
        'relocalize = naro.cli:main',
        'replay = naro.replay:main',
        'play_scan = naro.player:main',
        'naro-map = naro.map_tools:main',
        'naro-reloc = naro.cli:main',
        'naro-replay = naro.replay:main',
    ]},
)
