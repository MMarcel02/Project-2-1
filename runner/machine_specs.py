"""Hardware specs of this machine, for the database.

Values are detected automatically. If configs/machine_overrides.yaml exists,
its values replace the detected ones
"""

import platform
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import get_type_hints

import psutil
from yaml import safe_load as yaml_safe_load

OVERRIDES_PATH = Path(__file__).resolve().parent / "configs" / "machine_overrides.yaml"


@dataclass(frozen=True)
class HardwareSpecs:
    """This machine's hardware; the field names are the database's column names"""

    os_name: str
    cpu_model: str
    cpu_physical_cores_total: int
    cpu_logical_cores_total: int
    cpu_p_cores_allocated: int
    cpu_e_cores_allocated: int
    cpu_p_core_max_ghz: float
    cpu_e_core_max_ghz: float
    ram_total_mb: int
    ram_allocated_limit_mb: int | None = None  # None = no limit
    ram_speed_mhz: int | None = None  # None = unknown
    ram_type: str | None = None  # e.g. "DDR5"; None = unknown


def read_cpu_model() -> str:
    """Retrieve the CPU's name"""
    os_name = platform.system()
    if os_name == "Windows":
        try:
            import winreg  # note: moved to windows-specific logic, never deleted

            root_key = winreg.HKEY_LOCAL_MACHINE
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            access = winreg.KEY_READ
            with winreg.OpenKey(root_key, path, access=access) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os_name == "Linux":
        try:
            with open("/proc/cpuinfo") as file:
                for line in file:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def read_max_ghz() -> float:
    """The max CPU clock speed the system reports (GHz).
    If unknown use 0.0 as no CPU runs at 0.0 GHz
    """
    try:
        freq = psutil.cpu_freq()
    except (OSError, NotImplementedError):
        freq = None
    if not freq:
        return 0.0

    return round((freq.max or freq.current) / 1000, 2)


def get_detected_specs() -> HardwareSpecs:
    """Return every spec that can be read from this machine. P- and E-cores
    can't be detected reliably, so all logical cores count as P-cores.

    Only total RAM amount can be measured, RAM type and RAM speed cannot be recorded.

    Consider using a machine_overrides.yaml for recording actual values
    """
    logical = psutil.cpu_count(logical=True) or 1
    return HardwareSpecs(
        os_name=platform.system(),
        cpu_model=read_cpu_model(),
        cpu_physical_cores_total=psutil.cpu_count(logical=False) or logical,
        cpu_logical_cores_total=logical,
        cpu_p_cores_allocated=logical,  # without manual values: every core counts as a P-core
        cpu_e_cores_allocated=0,
        cpu_p_core_max_ghz=read_max_ghz(),
        cpu_e_core_max_ghz=0.0,
        ram_total_mb=round(psutil.virtual_memory().total / 1024**2),
    )


def load_overrides(path: Path = OVERRIDES_PATH) -> dict:
    """Retrieve manually recorded hardware values from a specific path.
    - By default reads from machine_overrides.yaml.
    - returns empty if the file doesn't exist
    """
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as file:
        overrides = yaml_safe_load(file) or {}  # an empty file gives None
    if not isinstance(overrides, dict):
        raise TypeError(f"{path} must contain 'name: value' lines")
    return overrides


def apply_overrides(specs: HardwareSpecs, overrides: dict) -> HardwareSpecs:
    """Override auto-recorded specs from the machine.
    Also checks the types read from the override against the database's types and
    raises an exception when they don't match
    """
    field_types = get_type_hints(HardwareSpecs)
    unknown = set(overrides) - set(field_types)
    if unknown:
        raise ValueError("Unknown hardware setting(s): " + ", ".join(sorted(unknown)))

    # type check the values from the read file against the types of the database
    checked = {}
    for name, value in overrides.items():
        expected = field_types[name]
        if expected is float and isinstance(value, int):
            value = float(value)
        if not isinstance(value, expected):
            # use !r to show the `true type` of a value in the terminal
            raise TypeError(
                f"Hardware setting {name}: {value!r} has the wrong type (expected {expected})"
            )
        checked[name] = value
    return replace(specs, **checked)


def read_specs() -> HardwareSpecs:
    """Detects hardware specs, and applies override(s) if such exist"""
    return apply_overrides(get_detected_specs(), load_overrides())


def get_summary(specs: HardwareSpecs) -> str:
    """Format the hardware specifications so they are easier to read (table-like)"""
    # max field name length so values can be properly padded
    width = max(len(field.name) for field in fields(specs))
    lines = []
    for field in fields(specs):
        value = getattr(specs, field.name)
        lines.append(
            # `value if value` doesn't work as some values are recorded as 0, which is interpreted as false
            f"{field.name:<{width}}  {value if value is not None else 'unknown'}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_summary(read_specs()))
