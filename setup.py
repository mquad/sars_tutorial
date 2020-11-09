import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sars_tutorial", # Replace with your own username
    version="0.0.1",
    author="Massimo Quadrana",
    author_email="max.square@gmail.com",
    description="Sars tutorial",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mquad/sars_tutorial",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "scipy==1.1.0",
        "theano==1.0.3",
        "tqdm==4.25.0",
        "pandas==0.23.4",
        "gensim==3.4.0",
        "matplotlib",
        "mkl-service",
        "numba==0.39.0",
        "numpy==1.15.1",
        "treelib<=1.5.5",
        "pymining"
    ]
)
