import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DI2",
    version="1.0.2",
    author="L. Alexandre, R.S. Costa, R. Henriques",
    author_email="leonardoalexandre@tecnico.ulisboa.pt",
    description="An an unsupervised discretization method, DI2, for variables with arbitrarily skewed distributions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["multi-item discretization", "prior-free discretization", "heterogeneous biological data", "data mining"],
    project_urls={
        "Bug Tracker": "https://github.com/JupitersMight/DI2/issues",
    },
    url="https://github.com/JupitersMight/DI2",
    packages=['DI2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)