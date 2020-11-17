from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf8") as f:
    readme = f.read()

setup(
    name="cogdl",
    version="0.1.2",
    description="An Extensive Research Toolkit for Deep Learning on Graphs",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/THUDM/cogdl",

    license="MIT",

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",

        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # What does your project relate to?
    keywords="network embedding, graph representation learning, graph neural networks",

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "torch",
        "networkx",
        "matplotlib",
        "tqdm",
        "numpy",
        "scipy",
        "six",
        "gensim",
        "grave",
        "scikit_learn",
        "tabulate",
        "optuna",
        "texttable",
        "gensim",
        "ogb",
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extra_require={"test": ["pytest",]}
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
