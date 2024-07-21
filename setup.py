from setuptools import setup

setup(
    name="llama_toolchain",
    version="0.0.0.1",
    author="Meta Llama",
    author_email="rsm@meta.com",
    description="Llama toolchain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/meta-llama/llama-toolchain",
    package_dir={ "llama_toolchain": "toolchain"},
    classifiers=[
    ],
    python_requires=">=3.10",
    install_requires=[],
    include_package_data=True
)
