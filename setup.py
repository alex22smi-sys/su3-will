from setuptools import setup, find_packages

setup(
    name="su3-will",
    version="3.4",
    description="SU(3) Gauge-Covariant Resonance Graph Layer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="alex22smi-sys",
    license="MIT",
    packages=find_packages(),
    install_requires=["torch>=2.0.0", "numpy>=1.24.0"],
    python_requires=">=3.9",
)
