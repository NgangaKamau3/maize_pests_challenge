from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="fall_armyworm_detection",
    version="0.1.0",
    author="Fall Armyworm Detection Team",
    author_email="example@example.com",
    description="A deep learning system for detecting fall armyworm infestation in maize plants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fall_armyworm_detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)