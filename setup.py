from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name="pytorchLosses",
    version='0.3.2',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    description='Pytorch loss functions',
    py_modules=['pytorchLosses'],
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