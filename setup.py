from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='translation',
    version='0.1.0',
    description='Chinese to english machine traslation.',
    long_description=readme,
    author='Samu',
    author_email='samu@mail.mail',
    packages=find_packages(exclude=('tests'))
)
