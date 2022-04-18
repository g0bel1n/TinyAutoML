from setuptools import setup

setup(
    name="TinyAutoML",
    version="0.2.3.3",
    packages=[
        "TinyAutoML",
        "TinyAutoML.Preprocessing",
        "TinyAutoML.constants",
        "TinyAutoML.Models",
        "TinyAutoML.support",
    ],
    url="https://github.com/g0bel1n/TinyAutoML/tree/pooling-opt",
    license="MIT",
    author="g0bel1n",
    author_email="lucas.saban@ensae.fr",
    install_requires=[
        "pandas",
        "scikit-learn",
        "tqdm",
        "numpy",
        "statsmodels",
        "matplotlib",
        "xgboost",
    ],
    description="Combinaison of ML models for binary classification. Academic Project.",
    download_url="https://github.com/g0bel1n/TinyAutoML/archive/refs/tags/v0.2.3.3.tar.gz",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
    ],
)
