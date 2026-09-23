"""Compatibility entry point for pip versions that predate full PEP 621 support."""
from pathlib import Path

from setuptools import setup


setup(
    name='seedplane',
    version='0.12.1',
    description='Run language models as independent shards across CPU cores, GPUs and machines',
    long_description=Path(__file__).with_name('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=['seedplane'],
    python_requires='>=3.9',
    install_requires=['torch>=2.2', 'numpy', 'transformers>=4.45', 'safetensors'],
    entry_points={'console_scripts': ['seedplane=seedplane.cli:main']},
    license='MIT',
    url='https://github.com/robson-apk/SeedPlane',
)
