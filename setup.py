"""
propertyestimator
Property calculation toolkit from the Open Forcefield Consortium.
"""
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except IOError:
    long_description = "\n".join(short_description[2:]),


setup(
    # Self-descriptive entries which should always be present
    name='propertyestimator',
    author='Open Force Field Consortium',
    author_email='john.chodera@choderalab.org',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    # packages=['propertyestimator', "propertyestimator.tests"],

    packages=find_packages(),

    # Optional include package data to ship with your package
    # Comment out this line to prevent the files from being packaged with your software
    # Extend/modify the list to include/exclude other items as need be
    # package_data={'propertyestimator': ["data/*.dat"]
    #              },
    # UPDATE -> Use MANIFEST.in and set include_package_data=True
    # (https://blog.ionelmc.ro/presentations/packaging/#slide:13)
    include_package_data=True,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # author_email='me@place.org',      # Author email
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
