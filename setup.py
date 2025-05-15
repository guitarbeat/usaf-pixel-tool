#!/usr/bin/env python3
"""Setup script for USAF target analysis package."""

from setuptools import setup, find_packages

setup(
    name="usaf_analyzer",
    version="1.0.0",
    description="USAF 1951 Resolution Target Analysis Tool",
    author="USAF Analyzer Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "opencv-python>=4.2.0",
    ],
    entry_points={
        "console_scripts": [
            "usaf-analyzer=usaf_package.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
) 