from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym_pcgil',
      version='0.4.0',
      install_requires=['gym==0.18.0', 'numpy>=1.17', 'pillow', 'tensorflow==1.15.0', 'stable_baselines==2.10.1',
                        'pandas'],
      author="Matthew Siper",
      author_email="matt.quantheads.io@gmail.com",
      description="A package for \"PCGIL: Procedural Content Generation via Immitation Learning\".",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/matt-quant-heads-io/pcgil_gym",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ]
)
