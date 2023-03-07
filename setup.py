from setuptools import setup
setup(
    name='occlusion',
    version='v0.1',
    description='Functions to occlude objects in images.',
    author='Tim Maniquet',
    author_email='timotheemaniquet@gmail.com',
    url='https://github.com/TimManiquet/occlusion',
    packages=['occlusion'],
    install_requires=[
        'opencv-python',
        'numpy',
        'setuptools'
    ],
)