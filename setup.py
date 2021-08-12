from setuptools import setup, find_packages

setup(
    name='diffaqua',
    version='0.1',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'scipy'
    ],
)
