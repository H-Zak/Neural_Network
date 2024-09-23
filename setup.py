from setuptools import setup, find_packages

setup(
    name='Neural_netwrok',
    version='0.1',
    description='Creation of a Neural Network to detect cancer',
    author='Zhamdouc dnieto-c',
    author_email='',
    url='https://github.com/H-Zak/Neural_Network',
    packages=find_packages(),  # Recherche automatiquement les packages Python dans le projet
    install_requires=[
        'numpy>=1.18.0',
        'requests>=2.22.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Version minimale de Python
)
