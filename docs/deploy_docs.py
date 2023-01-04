"""Script to deploy the latest version to github pages using mike"""
import logging
import subprocess
from typing import List, Union

from packaging import version

logger = logging.getLogger(__file__)


def run_cmd(cmd: str) -> List[str]:
    "Run a command in a subprocess."
    return subprocess.check_output(cmd.split()).decode("utf-8").splitlines()


def get_current_version() -> version.Version:
    "Get current version specified in pyproject.toml."
    for line in run_cmd("poetry version"):
        if line.startswith("norfair"):
            return version.parse(line.split()[-1])
    raise Exception("Could not read current version from toml")


def get_latest_version() -> Union[version.Version, None]:
    "Get the deployed version number tagged as `latest` in mike."
    for line in run_cmd("mike list"):
        if line.endswith("[latest]"):
            return version.parse(line.split()[0])
    logger.warning("Could not read the latest version deployed")


def truncate_version(v: version.Version):
    "Truncates a version as `major.minor`"
    return version.parse(f"{v.major}.{v.minor}")


if __name__ == "__main__":
    logging.basicConfig()

    current_version = get_current_version()
    latest_version = get_latest_version()

    target_version = truncate_version(current_version)
    alias = ""

    if latest_version is None or current_version >= latest_version:
        alias = "latest"

    logger.debug(f"{latest_version=}")
    logger.debug(f"{current_version=}")
    logger.debug(f"{target_version=}")
    logger.debug(f"{alias=}")

    logger.info("Running mike...")

    logger.info("\n".join(run_cmd(f"mike deploy --push -u {target_version} {alias}")))
