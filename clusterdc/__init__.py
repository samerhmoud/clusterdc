# __init__.py
"""
ClusterDC: Density-Contour Clustering with Advanced Kernel Density Estimation.

This package provides tools for density-based clustering using kernel density 
estimation and contour analysis to identify clusters in 2D data.
"""

__version__ = "0.0.0.2"

import sys
import subprocess
import importlib.util
import os
import warnings

# Set up a flag to track if we've already attempted fixes
_attempted_fix = False

def _fix_dependencies_pip():
    """Fix dependencies using pip (faster approach)."""
    try:
        # Check if we're in conda environment
        in_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
        
        if in_conda:
            # For conda, use --ignore-installed to avoid uninstall errors
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'numpy<2.0.0', 'matplotlib>=3.3.0,<3.8.0',
                '--ignore-installed', '--quiet'
            ])
        else:
            # For regular Python, use --force-reinstall
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'numpy<2.0.0', 'matplotlib>=3.3.0,<3.8.0',
                '--force-reinstall', '--quiet'
            ])
        
        # Clear module cache
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith(('numpy.', 'matplotlib.')):
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
        
        if 'numpy' in sys.modules:
            del sys.modules['numpy']
        if 'matplotlib' in sys.modules:
            del sys.modules['matplotlib']
            
        # Try to import numpy to verify
        import numpy
        print(f"Successfully installed compatible dependencies with pip (NumPy {numpy.__version__}).")
        return True
        
    except Exception as e:
        print(f"Pip installation failed: {str(e)}")
        return False

def _fix_dependencies():
    """Automatically install compatible dependencies efficiently."""
    global _attempted_fix
    
    # Avoid trying multiple times
    if _attempted_fix:
        return False
    
    _attempted_fix = True
    
    try:
        # Check if we're having import issues
        needs_fixing = False
        
        # Check numpy version
        try:
            import numpy
            numpy_version = numpy.__version__
            numpy_major = int(numpy_version.split('.')[0])
            if numpy_major >= 2:
                needs_fixing = True
        except ImportError:
            needs_fixing = True
            
        # Check matplotlib
        if not needs_fixing:
            try:
                import matplotlib.pyplot
            except ImportError:
                needs_fixing = True
                
        if needs_fixing:
            # Check if we're in conda environment
            in_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
            
            if in_conda:
                # For conda, try pip with --ignore-installed (safer for conda)
                print("Conda environment detected. Installing compatible dependencies...")
                if _fix_dependencies_pip():
                    return True
                
                # If pip approach fails, suggest conda command
                print("\nFor a more complete fix, please run manually:")
                print("conda install -y numpy=1.24.3 matplotlib=3.7.1")
                print("Then restart your Python session.")
                return False
            else:
                # For regular Python, use pip
                print("Installing compatible dependencies...")
                return _fix_dependencies_pip()
                
    except Exception as e:
        print(f"Note: Auto-dependency fix attempt failed ({str(e)})")
        print("If you experience import errors, manually run:")
        print("pip install numpy<2.0.0 matplotlib>=3.3.0,<3.8.0 --force-reinstall")
    
    return False

# Try to fix dependencies - just once per session
if _fix_dependencies():
    # If we fixed dependencies, we need to restart the import process
    import importlib
    importlib.invalidate_caches()
else:
    # Continue with normal imports
    try:
        from .main import *
        
        try:
            from .datasets import *
        except ImportError:
            pass
            
    except ImportError as e:
        error_msg = str(e)
        if "matplotlib.backends.registry" in error_msg or "compiled using NumPy" in error_msg:
            print("\nImport Error: Dependencies compatibility issue detected.")
            
            # Check if we're in conda environment
            in_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
            if in_conda:
                print("For ClusterDC to work properly, please run:")
                print("    conda install -y numpy=1.24.3 matplotlib=3.7.1")
                print("Or alternatively:")
                print("    pip install numpy<2.0.0 matplotlib>=3.3.0,<3.8.0 --ignore-installed")
            else:
                print("For ClusterDC to work properly, please run:")
                print("    pip install numpy<2.0.0 matplotlib>=3.3.0,<3.8.0 --force-reinstall")
                
            print("Then restart your Python session before importing clusterdc again.\n")
        raise


def check_environment():
    """
    Check if the current environment is properly set up for ClusterDC.
    
    Returns:
        dict: Status of dependencies with version information
    """
    import importlib
    
    dependencies = {
        "numpy": "1.19.0",
        "matplotlib": "3.3.0",
        "pandas": "1.1.0",
        "scipy": "1.5.0",
        "scikit-learn": "1.0.0",
        "shapely": "1.7.0",
        "networkx": "2.5",
    }
    
    status = {}
    all_ok = True
    
    print("ClusterDC Environment Check:")
    print("----------------------------")
    
    # Check if we're in conda environment
    in_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if in_conda:
        print("Conda environment detected")
    else:
        print("Standard Python environment detected")
    
    for package, min_version in dependencies.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            
            # Very simple version comparison - just check major.minor
            min_ver_parts = min_version.split(".")[:2]
            cur_ver_parts = version.split(".")[:2]
            
            is_ok = True
            # Only mark as problematic if major version is too low
            if cur_ver_parts[0] < min_ver_parts[0]:
                is_ok = False
                all_ok = False
            
            status[package] = {
                "installed": True,
                "version": version,
                "min_version": min_version,
                "ok": is_ok
            }
            
            if is_ok:
                print(f"✓ {package}: {version} (min: {min_version})")
            else:
                print(f"✗ {package}: {version} (min: {min_version}) - UPDATE RECOMMENDED")
                
        except ImportError:
            status[package] = {
                "installed": False,
                "version": None,
                "min_version": min_version,
                "ok": False
            }
            all_ok = False
            print(f"✗ {package}: NOT FOUND (min: {min_version})")
    
    if all_ok:
        print("\nEnvironment looks good! ClusterDC should work properly.")
    else:
        print("\nSome dependencies may need updating. For best results, run:")
        if in_conda:
            print("conda install -y numpy=1.24.3 matplotlib=3.7.1")
            print("Or alternatively:")
            print("pip install numpy<2.0.0 matplotlib>=3.3.0,<3.8.0 --ignore-installed")
        else:
            print("pip install numpy<2.0.0 matplotlib>=3.3.0,<3.8.0 --force-reinstall")
    
    return status