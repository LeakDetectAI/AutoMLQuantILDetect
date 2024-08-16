from os.path import realpath, dirname, join

from setuptools import setup, find_packages

DISTNAME = "autoqild"
DESCRIPTION = "Information Leakage Detection Techniques"
MAINTAINER = "Pritha Gupta"
MAINTAINER_EMAIL = "prithag@mail.uni-paderborn.de"
VERSION = "1.0"
LICENSE = "Apache"
DOWNLOAD_URL = "https://github.com/prithagupta/automl_quant_il_detect/"
PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        packages=find_packages(),
        install_requires=install_reqs,
        url=DOWNLOAD_URL,
        include_package_data=True,
    )
