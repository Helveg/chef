import setuptools, os, sys

with open("README.md", "r") as fh:
    long_description = fh.read()

# Get the version from the patch module without importing it.
with open(os.path.join(os.path.dirname(__file__), "pesticide", "__init__.py"), "r") as f:
    for line in f:
        if "__version__ = " in line:
            exec(line.strip())
            break

setuptools.setup(
    name="pesticide",
    version=__version__,
    author="Robin De Schepper",
    author_email="robingilbert.deschepper@unipv.it",
    description="Inspect your models and get rid of bugs in your Arboretum.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/helveg/pesticide",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=["plotly", "arborize>=2.0.1"],
    extras_require={"dev": ["sphinx", "pre-commit", "black>=20.8b1", "sphinxcontrib-contentui"]},
)
