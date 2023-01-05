import subprocess
import sys

from packaging import version


def run_cmd(cmd):
    return subprocess.check_output(cmd.split()).decode("utf-8").splitlines()


def get_current_version():
    for line in run_cmd("poetry version"):
        if line.startswith("norfair"):
            return version.parse(line.split()[-1])


def get_latest_version():
    for line in run_cmd("mike list"):
        if line.endswith("[latest]"):
            return version.parse(line.split()[0])


def is_higher(va, vb):
    return va < vb


def truncate_version(v):
    return version.parse(f"{v.major}.{v.minor}")


if __name__ == "__main__":
    current_version = get_current_version()
    target_version = truncate_version(current_version)
    latest_version = get_latest_version()

    print(f"{latest_version=}")
    print(f"{current_version=}")
    print(f"{target_version=}")

    if is_higher(target_version, latest_version):
        aliases = "latest"
    else:
        aliases = ""

    print(f"{aliases=}")
    # print("\n".join(run_cmd(f"mike deploy -u {target_version} {aliases}")))
