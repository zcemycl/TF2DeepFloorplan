from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

if __name__ == "__main__":
    setup(use_scm_version=True, install_requires=required, test_suite="tests")
