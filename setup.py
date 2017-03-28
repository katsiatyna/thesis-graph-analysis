import uuid
from setuptools import setup
import os
from pip.req import parse_requirements

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# reqs_file = os.path.join(BASE_DIR, 'requirements.txt')
#install_reqs = parse_requirements(reqs_file, session=uuid.uuid1())

setup(
    name="thesis-graph-analysis",
    version="0.1",
    author="Lorena Mesa",
    author_email="me@lorenamesa.com",
    description=("Thesis IT4BI Katsiaryna Krasnashchok"),
    license="MIT",
    keywords="subgraph mining, kafka, spark",
    install_requires=["pynauty", "networkx", "peakutils",
                      "numpy", "matplotlib", "pyqt_fit",
                      "statsmodels", "sklearn"],
    long_description=read('README.md'),
    url="http://example.com",
    py_modules=['upc.bsc.grahanalysis.model.SubgraphCollection',
                  'upc.bsc.grahanalysis.model.VsigramGraph'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7"
    ]
)
