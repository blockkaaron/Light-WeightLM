from setuptools import setup, find_packages

setup(
    name="bloxslm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0.1",
        "omegaconf>=2.3.0",
        "safetensors>=0.4.2",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "sentencepiece>=0.2.0",
        "regex>=2024.1.0",
    ],
)
