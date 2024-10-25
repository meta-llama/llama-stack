# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from setuptools import find_packages, setup


# Function to read the requirements.txt file
def read_requirements():
    with open("requirements.txt") as req:
        content = req.readlines()
    return [line.strip() for line in content]


setup(
    name="llama_stack",
    version="0.0.45",
    author="Meta Llama",
    author_email="llama-oss@meta.com",
    description="Llama Stack",
    entry_points={
        "console_scripts": [
            "llama = llama_stack.cli.llama:main",
            "install-wheel-from-presigned = llama_stack.cli.scripts.run:install_wheel_from_presigned",
        ]
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/meta-llama/llama-stack",
    # workaround to include distributions/*/build.yaml files in the package
    # https://stackoverflow.com/questions/62550952/including-package-data-python-from-top-level-when-package-is-in-subdirectory
    packages=find_packages() + ["llama_stack.cli.distributions"],
    package_dir={"llama_stack.cli.distributions": "distributions"},
    package_data={"llama_stack.cli.distributions": ["distributions/*/build.yaml"]},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    include_package_data=True,
)
