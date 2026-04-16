from setuptools import setup, find_packages

setup(
    name='Neural_network',
    version='0.1',
    description='Multilayer Perceptron for breast cancer classification',
    author='Zhamdouc dnieto-c',
    url='https://github.com/H-Zak/Neural_Network',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'scikit-learn>=0.22.0',
        'matplotlib>=3.0.0',
        'joblib>=0.14.0',
    ],
    python_requires='>=3.6',
)
