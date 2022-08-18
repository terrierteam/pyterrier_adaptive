from glob import glob
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyterrier_adaptive",
    version="0.0.1",
    author="Sean MacAvaney",
    author_email="sean.macavaney@glasgow.ac.uk",
    description="PyTerrier implementation of Adaptive Re-Ranking using a Corpus Graph (CIKM 2022)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terrierteam/pyterrier_adaptive",
    packages=setuptools.find_packages(include=['pyterrier_adaptive']),
    install_requires=list(open('requirements.txt')),
    classifiers=[],
    python_requires='>=3.6',
    package_data={
        '': ['requirements.txt'],
    },
)
