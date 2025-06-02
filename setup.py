from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="superbowl-ad-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for analyzing Super Bowl advertisements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/superbowl-ad-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "superbowl-collect=src.data_collection.main:main",
            "superbowl-analyze=src.analysis.main:main",
            "superbowl-visualize=src.visualization.main:main",
        ],
    },
) 