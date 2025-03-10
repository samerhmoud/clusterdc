from setuptools import setup, find_packages, Command
from setuptools.command.install import install
import subprocess
import sys
import os

class CustomInstallCommand(install):
    """Custom install command to install compatible dependencies automatically."""
    def run(self):
        # Run the standard install
        install.run(self)
        
        # Install compatible versions of critical packages
        try:
            # Check if we're in conda environment
            in_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
            
            if in_conda:
                # For conda environments, suggest conda install
                print("\nConda environment detected.")
                print("For best compatibility, you may want to run:")
                print("conda install -y numpy=1.24.3 matplotlib=3.7.1")
                
                # Also try pip with --ignore-installed which is safer in conda
                try:
                    print("Attempting to install compatible dependencies with pip...")
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install',
                        'numpy<2.0.0', 'matplotlib>=3.3.0,<3.8.0',
                        '--user', '--quiet'
                    ])
                    print("Successfully installed compatible dependencies.")
                except Exception as e:
                    print(f"Note: Could not install with pip: {str(e)}")
                    print("Dependencies will be handled at import time if needed.")
            else:
                # Use pip for non-conda environments
                print("\nInstalling compatible dependencies...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install',
                    'numpy<2.0.0', 'matplotlib>=3.3.0,<3.8.0',
                    '--force-reinstall', '--quiet'
                ])
                print("Successfully installed compatible versions of NumPy and matplotlib.")
                
        except Exception as e:
            print(f"\nNote: Could not automatically install dependencies: {str(e)}")
            print("Package will attempt to fix dependencies when imported if needed.")

# Read the long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = (
        "ClusterDC: A powerful density-based clustering library for "
        "discovering patterns in 2D data using kernel density estimation."
    )

setup(
    name="clusterdc",  # Unique name for your package
    version="0.0.0.2",  # Version of your package
    author="Samer Hmoud & Maximilien Meyrieux",
    author_email="geo.samer.hmoud@gmail.com",
    description=(
        "A powerful density-based clustering library for discovering patterns in 2D data"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically finds packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.6",  # Minimum Python version
    install_requires=[
        # Core dependencies with more flexible version ranges
        "numpy>=1.19.0,<2.0.0",  # Stay with 1.x for maximum compatibility
        "pandas>=1.1.0",
        "matplotlib>=3.3.0,<3.8.0",  # Added upper bound for compatibility  
        "scipy>=1.5.0",  # No upper bound to avoid conflicts with gensim
        "scikit-learn>=1.0.0",
        "scikit-optimize>=0.8.0", 
        "shapely>=1.7.0",
        "networkx>=2.5",
        "tqdm>=4.50.0",
        "joblib>=1.0.0",
        "pillow>=7.1.0",  # Lower minimum to be compatible with streamlit
    ],
    keywords="clustering, density, contour, machine learning, data science, geochemistry, analysis",
    project_urls={
        "Bug Reports": "https://github.com/samerhmoud/clusterdc/issues",
        "Source": "https://github.com/samerhmoud/clusterdc",
        "Documentation": "https://github.com/samerhmoud/clusterdc#readme",
    },
    cmdclass={
        'install': CustomInstallCommand,  # Use our custom install command
    },
)