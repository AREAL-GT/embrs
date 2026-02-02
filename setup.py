from setuptools import setup, find_packages

setup(
    name='embrs',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'shapely',
        'tqdm',
        'matplotlib',
        'requests',
        'requests-cache',
        'retry-requests',
        'geopandas',
        'pyproj',
        'utm',
        'openmeteo-requests',
        'alphashape',
        'tkcalendar',
        'msgpack',
        'rasterio',
        'PyQt5',
        'timezonefinder',
        'pyarrow',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'ruff>=0.1.0',
            'memory-profiler>=0.60',
        ],
    },
    python_requires='>=3.9',
)