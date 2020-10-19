import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DI2-JupitersMight",
    version="0.0.1",
    author="Leonardo Alexandre",
    author_email="leonardoalexandre@tecnico.ulisboa.pt",
    description="A library used to discretize data according to a distribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JupitersMight/DI2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)