from setuptools import setup, find_packages

setup(
    name="hplot",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="A spatial heterogeneity plot inspired by Kaplan-Meier for tumor edge analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hplot",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.0",
        "pandas>=1.0",
        "scipy>=1.6",
        "numpy>=1.18"
    ],
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License"
    ],
    python_requires='>=3.7',
    include_package_data=True,
)