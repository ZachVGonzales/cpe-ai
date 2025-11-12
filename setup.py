from setuptools import setup, find_packages
from src.cpe_ai import __version__ as version

setup(
    name="cpe_ai",
    version=version,
    packages=find_packages(where="src"),
    package_dir={"":"src"},    
    install_requires=
    [
        open('requirements.txt').read()
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
)
