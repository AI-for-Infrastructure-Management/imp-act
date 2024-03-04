from setuptools import find_packages, setup

setup(name='imp_rl_challenge',
    packages=find_packages("imp_rl_challenge"),
    package_dir={"": "imp_rl_challenge"},
    package_data={'': ['*.yaml']},  # Packages should include any .yaml files
)