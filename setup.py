import os

from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

if __name__ == "__main__":
    use_scm_version = not os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)
    setup(
        use_scm_version=use_scm_version,
        install_requires=required,
        test_suite="tests",
    )
