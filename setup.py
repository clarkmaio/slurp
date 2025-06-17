from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='slurp', # Il nome del tuo pacchetto su PyPI
    version='0.1.0', # La versione attuale del tuo pacchetto
    author='andrea maioli',
    author_email='maioliandrea0@gmail.com',
    description='Package to build spline neuralnetwork with syntax similar to GAM packages',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/clarkmaio/slurp',
    packages=find_packages(), 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.7',
    install_requires=[
            "torch",
            "polars",
            "scipy",
            "altair",
    ],
)