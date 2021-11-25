# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="picarro",
    version="0.1.0",
    description="A package to analyze data from a Picarro G2308. YMMV.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6, <4",
    install_requires=[
        "pandas >= 1.3",
        "matplotlib >= 3.5",
        "scipy >= 1.7",
    ],
    extras_require={
        "dev": [
            "pytest==6.2.5",
        ],
    },
    package_data={
        "picarro": ["matplotlib-style"],
    },
    entry_points={
        "console_scripts": [
            # "sample=sample:main",
        ],
    },
)
