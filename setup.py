from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies that are essential for basic functionality
core_deps = [
    'streamlit>=1.22.0,<2.0.0',
    'pandas>=1.5.3,<2.0.0',
    'numpy>=1.23.5,<2.0.0',
    'yfinance>=0.2.18,<0.3.0',
    'plotly>=5.13.0,<6.0.0',
    'scikit-learn>=1.0.2,<2.0.0',
    'scipy>=1.9.3,<2.0.0',
    'pytz>=2022.7,<2024.0',
    'ta>=0.10.2,<1.0.0',
    'python-dateutil>=2.8.2,<3.0.0',
    'requests>=2.28.0,<3.0.0',
    'protobuf>=4.21.0,<5.0.0'
]

# Optional dependencies for specific features
ml_deps = [
    'xgboost>=1.7.0,<2.0.0',
    'lightgbm>=3.3.0,<4.0.0',
    'catboost>=1.0.0,<2.0.0',
]

web_deps = [
    'gunicorn>=20.1.0,<21.0.0',
    'flask>=2.0.0,<3.0.0',
]

# All optional dependencies
all_deps = ml_deps + web_deps

setup(
    name="stock-market-predictor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive stock market prediction tool with advanced analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-market-predictor",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    install_requires=core_deps,
    extras_require={
        'ml': ml_deps,
        'web': web_deps,
        'all': all_deps,
    },
    python_requires='>=3.9,<3.12',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='stock market prediction finance machine learning',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/stock-market-predictor/issues',
        'Source': 'https://github.com/yourusername/stock-market-predictor',
    },
)
