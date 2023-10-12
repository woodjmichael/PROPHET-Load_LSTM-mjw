"""
Copyright (C) 2020 Eugenio Gianniti

All rights reserved.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(include=["PROPHET_LOAD_LSTM", "PROPHET_LOAD_LSTM.*"])
    )
