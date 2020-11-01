import setuptools
from noisytest import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="noisytest-xifle",
    version=__version__,
    author="Felix Sygulla",
    author_email="felix.sygulla@tum.de",
    description="A noise-based failure detection tool for robot testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xifle/noisytest",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
