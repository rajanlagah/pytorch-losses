from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name="pytorchLosses",
    version='0.0.1',
    classifiers=[
        "Programming Langugae :: Python :: 3",
        "Programming Langugae :: Python :: 3.6",
        "Programming Langugae :: Python :: 3.7",
        "License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    description='Say hello',
    py_modules=['LabelSmoothingCrossEntropy'],
    package_dir={'':'src'},
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch >=1.6.0"
    ],
    extras_require = {
        "dev": [
            "pytest >= 3.7"
        ]
    },
    url="https://github.com/rajanlagah/pytorch-losses.git",
    author="Rajan Lagah",
    author_email="rajanlagah@gmail.com"
)