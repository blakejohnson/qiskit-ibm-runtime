# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"ntc-ibm-programs setup file."

import os

from setuptools import find_packages, setup

# pylint: disable=unspecified-encoding

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name="ntc-ibm-programs",
    version="0.0.1",
    description="Programs for internal runtime server.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/IBM-Q-Software/ntc-ibm-programs",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(exclude=["test*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.7",
    project_urls={
        "Bug Tracker": "https://github.ibm.com/IBM-Q-Software/ntc-ibm-programs/issues",
        "Source Code": "https://github.ibm.com/IBM-Q-Software/ntc-ibm-programs",
    },
    zip_safe=False,
)
