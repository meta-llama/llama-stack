from setuptools import find_packages, setup


# Function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt') as req:
        content = req.readlines()
    return [line.strip() for line in content]


setup(
    name="llama_toolchain",
    version="0.0.0.1",
    author="Meta Llama",
    author_email="rsm@meta.com",
    description="Llama toolchain",
    entry_points={
        "console_scripts": [
            'llama = llama_toolchain.cli.llama:main'
        ]
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/meta-llama/llama-toolchain",
    packages=find_packages(),
    classifiers=[],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    include_package_data=True
)
