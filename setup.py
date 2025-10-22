"""
Setup script for ES Futures VPOC Strategy Backtester.
A professional algorithmic trading strategy backtesting system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]

setup(
    name="futures-vpoc-backtest",
    version="1.0.0",
    author="Lukas Rueda",
    author_email="your.email@example.com",  # Update with actual email
    description="Advanced algorithmic trading strategy for E-mini S&P 500 futures with VPOC analysis and ML enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lrud/futures_vpoc_backtest",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "distributed": [
            "mpi4py>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "futures-train=src.ml.train:main",
            "futures-backtest=src.analysis.run_ml_backtest:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.png", "*.jpg"],
    },
    zip_safe=False,
    keywords="trading, finance, algorithmic-trading, futures, backtesting, machine-learning, volume-profile, vpoc",
    project_urls={
        "Bug Reports": "https://github.com/lrud/futures_vpoc_backtest/issues",
        "Source": "https://github.com/lrud/futures_vpoc_backtest",
    },
)