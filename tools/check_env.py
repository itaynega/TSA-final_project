"""Report the environment, and check it against what this project needs.

The brief asks the shipped README to pin package versions, and asks the report to
explain any difference between our numbers and the paper's. Both are unanswerable
after the fact if nobody wrote down what the run actually executed on, so this script
is meant to be run *before* the reconstruction and its output kept.

It also front-loads the two failures that otherwise appear a minute into training:
a missing dataset, and a torch build with no usable device.

Usage, from the repository root:

    python3 tools/check_env.py                 # print a report
    python3 tools/check_env.py --json results/environment.json
"""

import argparse
import importlib
import json
import os
import platform
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Versions this project was developed and validated against. These are *observed*,
# not requirements: TQNet's own requirements.txt pins no versions at all, so there is
# no upstream constraint to satisfy, only a record to keep.
VALIDATED_AGAINST = {
    "python": "3.13.5",
    "torch": "2.9.1",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "sklearn": "1.8.0",
    "matplotlib": "3.10.7",
}

# The paper's environment, from Appendix A.2 and upstream's README. Recorded because
# every difference here is a candidate explanation for a numeric gap.
PAPER_ENVIRONMENT = {
    "gpu": "1x NVIDIA GeForce RTX 4090, 24 GB",
    "framework": "PyTorch (version unstated)",
    "python": "3.8 per upstream README (now end-of-life)",
    "seed": 2024,
}

REQUIRED = ("torch", "numpy", "pandas", "sklearn", "matplotlib")


def module_version(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, "{}: {}".format(type(exc).__name__, exc)
    return getattr(module, "__version__", "unknown"), None


def collect():
    report = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {},
        "import_errors": {},
    }

    for name in REQUIRED:
        version, error = module_version(name)
        if error:
            report["import_errors"][name] = error
        else:
            report["packages"][name] = version

    # Device availability. `accelerator` is what run.py's `--accelerator auto` will
    # resolve to, so this line predicts where training will actually happen.
    devices = {"cuda": False, "mps": False, "cpu": True}
    if "torch" in report["packages"]:
        import torch

        devices["cuda"] = bool(torch.cuda.is_available())
        devices["mps"] = bool(torch.backends.mps.is_available())
        if devices["cuda"]:
            devices["cuda_device_name"] = torch.cuda.get_device_name(0)
            devices["cuda_version"] = torch.version.cuda
        devices["threads"] = torch.get_num_threads()
    report["devices"] = devices
    report["accelerator"] = "cuda" if devices["cuda"] else ("mps" if devices["mps"] else "cpu")

    # Dataset presence and identity.
    dataset = {"present": False}
    try:
        from common import data as data_mod

        dataset["path"] = data_mod.DEFAULT_CSV
        if os.path.exists(data_mod.DEFAULT_CSV):
            from tools import get_data

            dataset["present"] = True
            dataset["sha256"] = data_mod.data_sha256()
            dataset["matches_expected"] = dataset["sha256"] == get_data.EXPECTED_SHA256
    except Exception as exc:
        dataset["error"] = "{}: {}".format(type(exc).__name__, exc)
    report["dataset"] = dataset

    report["validated_against"] = dict(VALIDATED_AGAINST)
    report["paper_environment"] = dict(PAPER_ENVIRONMENT)
    return report


def drifted(report):
    """Packages whose version differs from the one this project was validated on."""
    out = {}
    observed = dict(report["packages"])
    observed["python"] = report["python"]
    for name, expected in VALIDATED_AGAINST.items():
        actual = observed.get(name)
        if actual is not None and actual != expected:
            out[name] = (expected, actual)
    return out


def render(report):
    lines = []
    add = lines.append

    add("Environment")
    add("-" * 60)
    add("python      : {}".format(report["python"]))
    add("platform    : {}".format(report["platform"]))
    add("machine     : {}".format(report["machine"]))
    add("")

    add("Packages")
    add("-" * 60)
    for name in REQUIRED:
        if name in report["packages"]:
            add("{:<12}: {}".format(name, report["packages"][name]))
        else:
            add("{:<12}: MISSING ({})".format(name, report["import_errors"].get(name, "")))
    add("")

    add("Compute")
    add("-" * 60)
    devices = report["devices"]
    add("cuda        : {}{}".format(
        devices["cuda"],
        "  ({})".format(devices.get("cuda_device_name")) if devices.get("cuda_device_name") else "",
    ))
    add("mps         : {}".format(devices["mps"]))
    add("torch threads: {}".format(devices.get("threads", "?")))
    add("--accelerator auto will select: {}".format(report["accelerator"]))
    if report["accelerator"] == "cpu":
        add("")
        add("  CPU only. The target cell is small (661,640 parameters, 8,449 training")
        add("  windows) and trains in minutes rather than hours, so this is workable.")
        add("  The paper used an RTX 4090; record that as an environment deviation.")
    add("")

    add("Dataset")
    add("-" * 60)
    dataset = report["dataset"]
    if not dataset.get("present"):
        add("ETTh1.csv   : MISSING")
        add("              run: python3 tools/get_data.py")
    else:
        add("ETTh1.csv   : present")
        add("sha256      : {}".format(dataset["sha256"]))
        add("as expected : {}".format(dataset.get("matches_expected")))
    add("")

    drift = drifted(report)
    add("Drift from the validated environment")
    add("-" * 60)
    if not drift:
        add("none")
    else:
        for name, (expected, actual) in sorted(drift.items()):
            add("{:<12}: validated {} / running {}".format(name, expected, actual))
        add("")
        add("Version drift is not an error. It is a reportable deviation: note it in")
        add("the README and treat it as a candidate explanation for any numeric gap.")
    add("")

    add("Paper's environment, for the deviation table")
    add("-" * 60)
    for key, value in report["paper_environment"].items():
        add("{:<12}: {}".format(key, value))

    blocking = []
    if report["import_errors"]:
        blocking.append("missing packages: {}".format(", ".join(sorted(report["import_errors"]))))
    if not report["dataset"].get("present"):
        blocking.append("ETTh1.csv absent")
    elif report["dataset"].get("matches_expected") is False:
        blocking.append("ETTh1.csv does not match the expected digest")

    add("")
    if blocking:
        add("NOT READY: " + "; ".join(blocking))
    else:
        add("READY: environment and dataset are usable.")

    return "\n".join(lines), bool(blocking)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    args = parser.parse_args(argv)

    report = collect()
    text, blocking = render(report)
    print(text)

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(REPO_ROOT, args.json)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("\nwrote {}".format(path))

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
