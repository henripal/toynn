from setuptools import setup, find_packages

setup(name='toynn',
        version='0.0001',
        description='A basic but extendable nn library',
        url='http://github.com/henripal/toynn',
        author='Henri Palacci',
        author_email='henri.palacci@gmail.com',
        license='MIT',
        packages= find_packages(),
        install_requires=['numpy']
        )
