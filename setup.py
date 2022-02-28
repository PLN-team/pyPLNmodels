from setuptools import setup, find_packages

from PLNpy import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setup(
    name='PLNpy',
    version=__version__,
    description = 'Package implementing PLN models', 
    url='https://github.com/PLN-team/PLNpy/tree/master/PLNpy',
    author='Bastien Batardière, Julien Chiquet, Joon Kwon',
    author_email='bastien.batardiere@gmail.com',
    license_files = ('LICENSE.txt',),
    long_description=long_description,
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    py_modules=['PLNpy/utils','PLNpy/VRA','PLNpy/models'],
)
