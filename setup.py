"""
This setup.py is intentionally left minimal to avoid conflicts with requirements.txt.
All dependencies are managed through requirements.txt for Streamlit Cloud deployment.
"""
from setuptools import setup

setup(
    name="stock-market-predictor",
    version="1.0.0",
    install_requires=[],  # All dependencies are managed through requirements.txt
    python_requires='>=3.9',
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
