from setuptools import find_packages, setup
import versioneer

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="caribou_hi",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A Bayesian Model of the Diffuse Neutral Interstellar Medium",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.11",
    license="GNU General Public License v3 (GPLv3)",
    url="https://github.com/tvwenger/caribou_hi",
)
