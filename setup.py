from setuptools import setup, find_packages

setup(
    name="clusterdc",  # Unique name for your package
    version="0.2.0",  # Version of your package
    author="Samer Hmoud & Maximilien Meyrieux",
    author_email="geo.samer.hmoud@gmail.com",
    description = (
    "ClusterDC is a fast and robust density-based clustering algorithm for identifying clusters in "
    "two-dimensional embeddings. Initially developed for geochemists, it is applicable across natural sciences "
    "and engineering fields, aiding in the analysis of complex datasets by identifying meaningful patterns in data."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically finds packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
    install_requires=[
        "numpy>=1.22.4",
        "pandas>=2.0.3",
        "scipy>=1.10.1",
        "seaborn>=0.11.2",
        "matplotlib>=3.7.1",
        "shapely>=2.0.4",
        "networkx>=3.1",
    ],
)
