import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlqa",
    version="0.1.0.post1",
    author="Dogan Askan",
    author_email="doganaskan@gmail.com",
    description="A Package to perform QA on data flows in Machine Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ddaskan/mlqa",
    download_url="https://pypi.python.org/pypi/mlqa",
    project_urls={
        "Bug Tracker": "https://github.com/ddaskan/mlqa/issues",
        "Documentation": "http://www.doganaskan.com/mlqa/",
        "Source Code": "https://github.com/ddaskan/mlqa",
    },
    license='MIT',
    keywords='qa ml ai data analysis machine learning quality assurance',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
    ],
    python_requires='>=3.5',
    install_requires=['numpy', 'pandas'],
)