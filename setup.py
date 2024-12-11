from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]

setup(
    name="dac-sim",
    version="0.0.3",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    package_data={"dac_sim": ["models/*"]},
    entry_points={
        "console_scripts": [
            "dac-sim = dac_sim.cli:cli",
        ],
    },
)
