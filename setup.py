from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.rst')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

"""version = {}
with open(os.path.join(_here, 'somepackage', 'version.py')) as f:
    exec(f.read(), version)"""

setup(
    name='swag',
    version='0.0',
    description=('SWA-Gaussian repo'),
    long_description=long_description,
    author='Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson',
    author_email='wm326@cornell.edu',
    url='https://github.com/wjmaddox/private_swa_uncertainties',
    license='MPL-2.0',
    packages=['swag'],
   install_requires=[
       'torch>=1.0.0',
       'gpytorch'
   ],
#   no scripts in this example
#   scripts=['bin/a-script'],
    include_package_data=True,
    classifiers=[
        'Development Status :: 0',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6'],
)