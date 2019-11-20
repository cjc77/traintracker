from setuptools import setup
import os

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as fp:
    README = fp.read()

with open(os.path.join(HERE, "requirements.txt")) as fp:
    REQUIREMENTS = list(fp.read().replace(os.linesep, '\n').strip().split('\n'))


setup(
    name="traintracker",
    version="0.0.1",
    description="Evaluate and monitor model training.",
    long_description=README,
    url="https://github.com/cjc77/train_tracker",
    author="Carson Cook",
    maintainer_email="carsonjamescook@gmail.com",
    license="MIT",
    packages=["traintracker", "tests"],
    python_requires=">=3.5",
    install_requires=REQUIREMENTS
)
