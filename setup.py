from glob import glob
import setuptools

def get_version(path):
    for line in open(path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError(f"Unable to find __version__ in {path}")


def get_requirements(path):
    res = []
    for line in open(path):
        line = line.split('#')[0].strip()
        if line:
            res.append(line)
    return res

setuptools.setup(
    name="pyterrier-adaptive",
    version=get_version('pyterrier_adaptive/__init__.py'),
    author="Sean MacAvaney",
    author_email="sean.macavaney@glasgow.ac.uk",
    description="PyTerrier implementation of Adaptive Re-Ranking using a Corpus Graph (CIKM 2022)",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/terrierteam/pyterrier_adaptive",
    packages=setuptools.find_packages(include=['pyterrier_adaptive']),
    install_requires=list(open('requirements.txt')),
    classifiers=[],
    python_requires='>=3.8',
    entry_points={
        'pyterrier.artifact': [
            'corpus_graph.np_topk = pyterrier_adaptive.corpus_graph:NpTopKCorpusGraph',
        ],
    },
    package_data={
        '': ['requirements.txt'],
    },
)
