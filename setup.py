from setuptools import setup, find_packages

setup(
    name="stock-market-predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.22.0',
        'pandas>=1.5.3',
        'numpy>=1.23.5',
        'yfinance>=0.2.18',
        'plotly>=5.13.0',
        'scikit-learn>=1.0.2',
        'scipy>=1.9.3',
        'pytz>=2022.7',
        'ta>=0.10.2',
        'xgboost>=1.7.0',
        'lightgbm>=3.3.0',
        'catboost>=1.0.0',
        'gunicorn>=20.1.0',
        'requests>=2.28.0',
        'python-dateutil>=2.8.2',
        'protobuf>=4.21.0'
    ],
    python_requires='>=3.9',
)
