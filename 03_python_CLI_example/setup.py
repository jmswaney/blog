"""Starfox Python Package

This setup.py file uses setuptools to install the `starfox` package
"""

from setuptools import setup, find_packages

setup(
    name="starfox",
    version="0.1.0",
    description="Do a barrel roll!",
    packages=find_packages(),
    install_requires=[
        "click",
        "questionary"
    ],
    entry_points={
        'console_scripts': ['starfox=starfox.main:main']
    },
    author="Justin Swaney",
    license="MIT"
)
