import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
setup(name="octilab",
      version="0.0.0",
      description="An isaac sim reinforcement learning algorithm, environment library on top of isaac lab",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      url="https://github.com/takuseno/d3rlpy",
      author="Zhengyu Zhang",
      author_email="zzyoctopus@gmail.com",
      license="MIT License",
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Education",
                   "Intended Audience :: Science/Research",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Programming Language :: Python :: 3.10",
                   "Operating System :: POSIX :: Linux"],
      install_requires=["d3rlpy"],
      packages=find_packages(exclude=["test*"]),
      python_requires=">=3.10.0",
      zip_safe=False)
