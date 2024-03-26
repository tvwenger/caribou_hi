from setuptools import setup
import re


def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="caribou",
    version=get_property("__version__", "caribou"),
    description="A Model of the Diffuse Neutral Interstellar Medium",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["caribou"],
    install_requires=install_requires,
    python_requires=">=3.9",
    license="GNU GPLv3",
    url="https://github.com/tvwenger/caribou",
)
