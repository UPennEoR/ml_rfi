# -*- mode: python; coding: utf-8
# Copyright (c) 2019 The HERA Team
# Licensed under the 2-clause BSD License

from setuptools import setup
import io
import glob

__version = "0.0.1"

with io.open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_args = {
    "name": "ml_rfi",
    "author": "The HERA Collaboration",
    "url": "https://github.com/UPennEoR/ml_rfi",
    "license": "BSD",
    "description": "a package for identifying RFI using machine learning",
    "long_description": readme,
    "long_description_content_type": "text/markdown",
    "package_dir": {"ml_rfi": "ml_rfi"},
    "packages": ["ml_rfi"],
    "scripts": glob.glob("scripts/*"),
    "version": __version,
    "include_package_data": True,
    "install_requires": ["tensorflow"],
    "classifiers": ["Development Status :: 3 - Alpha",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: BSD License",
                    "Programming Language :: Python :: 3.7",
                    "Topic :: Scientific/Engineering :: Astronomy"],
    "keywords": "radio astronomy machine learning"
}

if __name__ == "__main__":
    setup(**setup_args)
