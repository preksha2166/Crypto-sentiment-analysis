"""
Setup script for the cryptocurrency sentiment analysis system.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="crypto_sentiment",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced sentiment analysis system for cryptocurrency trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-sentiment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "plotly",
        "streamlit"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.7b0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-sentiment=crypto_sentiment.main:main",
            "crypto-sentiment-dashboard=crypto_sentiment.dashboard.app:main",
        ],
    },
) 