from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = "Flights' arrival schedule generator"
LONG_DESCRIPTION = ''

# Setting up
setup(
    name="ScheduleMaker",
    version=VERSION,
    author="Andrea Gasparin, Fulvio Vascotto",
    author_email="<andrea.gasparin@phd.units.it>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='git@github.com:andygaspar/SchedulePackage.git',
    packages=find_packages(),
    license='MIT',
    install_requires=['pandas'],
    keywords=['flights', 'schedule', 'regulations', 'delay'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix",
    ],
    include_package_data=True
)
