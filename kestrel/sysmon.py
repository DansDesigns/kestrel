"""System monitoring.

Running a model locally is a resource negotiation, and the numbers that matter —
is the GPU actually being used, is memory about to run out, is this falling back
to CPU — are otherwise in a different window.

psutil is used when present and is the better source. When it is not, the same
figures are read from /proc on Linux and from the Win32 API through ctypes on
Windows, because requiring a dependency to show a memory bar would be a poor
trade. GPUs are read from the vendor tools that ship with the drivers.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


def _module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE_PSUTIL = _module("psutil")


INTEGRATED_HINTS = ("iris", "uhd", "hd graphics", "integrated", "apple",
                    "i915", "intel", "radeon graphics", "vega ", "gfx",
                    "adreno", "mali", "llvmpipe")
# Only these are integrated: a Vega 56/64 is a card with its own memory, and
# treating it as integrated would size an offload against system RAM it cannot
# reach. The trailing space in "vega " matches "Vega 8 Graphics" while leaving
# "RX Vega 64" alone.
DISCRETE_HINTS = ("rx vega", "radeon rx", "radeon pro", "geforce", "quadro",
                  "tesla", "radeon vii", "arc(tm)", "arc a", " arc ", "battlemage",
                  "firepro", "instinct", "titan")


@dataclass
class GpuSample:
    name: str = ""
    utilisation: float = -1.0      # percent, -1 when unknown
    mem_used_mb: int = 0
    mem_total_mb: int = 0          # dedicated, as the adapter reports it
    shared_mb: int = 0             # system memory the GPU may borrow
    temperature: float = -1.0
    vendor: str = ""
    meta_note: str = ""

    @property
    def integrated(self) -> bool:
        """Integrated parts have no memory of their own worth the name.

        Win32_VideoController reports AdapterRAM in a 32-bit field, so an Iris
        Xe with 15 GB of shared system memory available to it comes back as
        1 GB — and sizing an offload from that number puts nothing on the GPU.
        """
        low = self.name.lower()
        if any(hint in low for hint in DISCRETE_HINTS):
            return False
        return any(hint in low for hint in INTEGRATED_HINTS)

    # Win32_VideoController.AdapterRAM is a 32-bit field, so anything with more
    # than 4 GB reports exactly 4 GB. An 8 GB card looking like a 4 GB one is
    # the same bug as a 24 GB one looking like 4 GB.
    CAPPED_MB = 4096

    @property
    def capped(self) -> bool:
        return self.mem_total_mb in (self.CAPPED_MB, self.CAPPED_MB - 1)

    @property
    def budget_mb(self) -> int:
        """What can actually be allocated for a model."""
        return max(self.shared_mb, self.mem_total_mb)

    def memory_summary(self) -> str:  # noqa: D401
        """Both figures, because the small one is the one that gets quoted.

        An adapter reporting 1 GB alongside 15.9 GB of shared system memory is
        not a 1 GB device for the purpose of loading a model, and showing only
        the dedicated number is what makes it look like one.
        """
        bits = []
        if self.mem_total_mb:
            more = " (at least — the driver caps its report here)" if self.capped else ""
            bits.append(f"{self.mem_total_mb / 1024:.1f} GB dedicated{more}")
        if self.shared_mb:
            bits.append(f"{self.shared_mb / 1024:.1f} GB shared with system RAM")
        return "  +  ".join(bits) or "memory unknown"

    @property
    def mem_percent(self) -> float:
        return 100.0 * self.mem_used_mb / self.mem_total_mb if self.mem_total_mb else 0.0


@dataclass
class Sample:
    gpus_scanned: bool = False     # False while the first scan is still running
    cpu_temp: float = -1.0         # -1 when the platform will not say
    cpu_percent: float = 0.0
    per_core: list[float] = field(default_factory=list)
    mem_used_mb: int = 0
    mem_total_mb: int = 0
    swap_used_mb: int = 0
    swap_total_mb: int = 0
    gpus: list[GpuSample] = field(default_factory=list)
    source: str = ""

    @property
    def mem_percent(self) -> float:
        return 100.0 * self.mem_used_mb / self.mem_total_mb if self.mem_total_mb else 0.0


class Monitor:
    def __init__(self, gpu_interval: float = 3.0):
        self._prev_cpu: tuple[int, int] | None = None
        self._gpu_interval = gpu_interval
        self._gpu_at = 0.0
        self._gpu_cache: list[GpuSample] = []
        self._gpu_busy = False
        self._gpu_scanned = False
        self._gpu_tool = _find_gpu_tool()
        if HAVE_PSUTIL:
            try:
                import psutil
                psutil.cpu_percent(interval=None)     # prime the delta
            except Exception:
                pass

    # -- cpu and memory -------------------------------------------------------
    def sample(self) -> Sample:
        s = Sample()
        if HAVE_PSUTIL:
            try:
                import psutil
                s.cpu_percent = float(psutil.cpu_percent(interval=None))
                s.per_core = [float(x) for x in psutil.cpu_percent(interval=None, percpu=True)]
                vm = psutil.virtual_memory()
                s.mem_used_mb = int((vm.total - vm.available) / 1048576)
                s.mem_total_mb = int(vm.total / 1048576)
                sw = psutil.swap_memory()
                s.swap_used_mb = int(sw.used / 1048576)
                s.swap_total_mb = int(sw.total / 1048576)
                s.source = "psutil"
            except Exception:
                s.source = ""
        if not s.source:
            s.cpu_percent = self._cpu_fallback()
            used, total = self._mem_fallback()
            s.mem_used_mb, s.mem_total_mb = used, total
            s.source = "built-in"
        s.gpus = self.gpus()
        s.gpus_scanned = self._gpu_scanned
        s.cpu_temp = _cpu_temperature()
        return s

    def _cpu_fallback(self) -> float:
        if platform.system() == "Linux":
            try:
                fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
                values = [int(x) for x in fields[:8]]
                idle = values[3] + values[4]
                total = sum(values)
                if self._prev_cpu:
                    d_total = total - self._prev_cpu[0]
                    d_idle = idle - self._prev_cpu[1]
                    self._prev_cpu = (total, idle)
                    if d_total > 0:
                        return max(0.0, min(100.0, 100.0 * (d_total - d_idle) / d_total))
                self._prev_cpu = (total, idle)
            except (OSError, ValueError, IndexError):
                return 0.0
            return 0.0
        if os.name == "nt":
            try:
                idle, kernel, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(),
                                      ctypes.c_ulonglong())
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle),
                                                      ctypes.byref(kernel),
                                                      ctypes.byref(user))
                total = kernel.value + user.value
                if self._prev_cpu:
                    d_total = total - self._prev_cpu[0]
                    d_idle = idle.value - self._prev_cpu[1]
                    self._prev_cpu = (total, idle.value)
                    if d_total > 0:
                        return max(0.0, min(100.0, 100.0 * (d_total - d_idle) / d_total))
                self._prev_cpu = (total, idle.value)
            except Exception:
                return 0.0
        return 0.0

    def _mem_fallback(self) -> tuple[int, int]:
        if platform.system() == "Linux":
            try:
                info = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, _, rest = line.partition(":")
                    info[key] = int(rest.strip().split()[0])
                total = info.get("MemTotal", 0) // 1024
                available = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
                return total - available, total
            except (OSError, ValueError, IndexError):
                return 0, 0
        if os.name == "nt":
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            try:
                status = MemStatus()
                status.dwLength = ctypes.sizeof(MemStatus)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                total = int(status.ullTotalPhys / 1048576)
                avail = int(status.ullAvailPhys / 1048576)
                return total - avail, total
            except Exception:
                return 0, 0
        if platform.system() == "Darwin":
            try:
                total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                           capture_output=True, text=True,
                                           timeout=5).stdout.strip()) // 1048576
                return 0, total
            except Exception:
                return 0, 0
        return 0, 0

    # -- gpus -----------------------------------------------------------------
    def gpus(self) -> list[GpuSample]:
        """The cached reading, refreshed in the background.

        Reading GPU counters spawns a process — PowerShell takes the better part
        of a second — so the caller is never made to wait for it. The panel
        shows the previous sample until the new one lands.
        """
        now = time.time()
        if now - self._gpu_at >= self._gpu_interval and not self._gpu_busy:
            self._gpu_busy = True
            self._gpu_at = now
            threading.Thread(target=self._refresh_gpus, daemon=True).start()
        return self._gpu_cache

    def _refresh_gpus(self) -> None:
        try:
            self._gpu_cache = _read_gpus(self._gpu_tool)
        except Exception:
            pass
        finally:
            self._gpu_scanned = True
            self._gpu_busy = False


def _find_gpu_tool() -> str:
    """Vendor tools first, then the platform's own interface.

    nvidia-smi and rocm-smi only exist for discrete cards from those two
    vendors. An integrated Intel GPU — which is what most laptops actually run
    llama.cpp on — is reported by neither, and returning "no GPU" for a machine
    that plainly has one is simply wrong.
    """
    for name in ("nvidia-smi", "rocm-smi"):
        if shutil.which(name):
            return name
    if os.name == "nt":
        return "windows"
    if platform.system() == "Darwin":
        return "apple"
    if Path("/sys/class/drm").is_dir():
        return "sysfs"
    return ""


def _read_gpus(tool: str) -> list[GpuSample]:
    if tool == "nvidia-smi":
        return _nvidia()
    if tool == "rocm-smi":
        return _rocm()
    if tool == "windows":
        return _windows()
    if tool == "sysfs":
        return _sysfs()
    if tool == "apple":
        return [GpuSample(name="Apple GPU", vendor="apple")]
    return []


PS_GPU = (
    "$ErrorActionPreference='SilentlyContinue';"
    "Get-CimInstance Win32_VideoController | ForEach-Object "
    "{ 'ADP|' + $_.Name + '|' + $_.AdapterRAM };"
    "$u=(Get-Counter '\\GPU Engine(*engtype_3D)\\Utilization Percentage')"
    ".CounterSamples | Measure-Object -Property CookedValue -Sum;"
    "'UTL|' + [math]::Round($u.Sum,1);"
    "$m=(Get-Counter '\\GPU Process Memory(*)\\Local Usage')"
    ".CounterSamples | Measure-Object -Property CookedValue -Sum;"
    "'MEM|' + [math]::Round($m.Sum/1MB,0)"
)


def _windows() -> list[GpuSample]:
    """Every adapter Windows knows about, including integrated ones, with
    utilisation from the GPU performance counters that Task Manager uses."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", PS_GPU],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    adapters: list[GpuSample] = []
    utilisation = -1.0
    used_mb = 0
    for line in (out.stdout or "").splitlines():
        parts = line.strip().split("|")
        if parts[0] == "ADP" and len(parts) >= 2 and parts[1]:
            ram = 0
            try:
                ram = int(int(parts[2]) / 1048576) if len(parts) > 2 and parts[2] else 0
            except ValueError:
                ram = 0
            adapters.append(GpuSample(name=parts[1].strip(), mem_total_mb=ram,
                                      vendor=_vendor_of(parts[1])))
        elif parts[0] == "UTL" and len(parts) > 1:
            try:
                utilisation = float(parts[1])
            except ValueError:
                pass
        elif parts[0] == "MEM" and len(parts) > 1:
            try:
                used_mb = int(float(parts[1]))
            except ValueError:
                pass
    if adapters:
        # The counters are machine-wide rather than per adapter, so they are
        # attributed to the first one rather than reported for each.
        adapters[0].utilisation = utilisation
        adapters[0].mem_used_mb = used_mb
    shared = _shared_budget_mb()
    real = _windows_dedicated_mb()
    for adapter in adapters:
        # The registry figure wins whenever it is larger: it is the one that
        # is not truncated at 4 GB.
        exact = real.get(adapter.name.strip().lower())
        if exact and exact > adapter.mem_total_mb:
            adapter.mem_total_mb = exact
        if adapter.integrated:
            adapter.shared_mb = shared
        elif adapter.capped and not exact:
            # A discrete card still reporting exactly 4 GB is almost certainly
            # larger; say so rather than sizing an offload against a wrong number.
            adapter.meta_note = "reported 4 GB; the driver may have more"
    return adapters


def _windows_dedicated_mb() -> dict:
    """Real VRAM per adapter, from the driver's own registry entry.

    `qwMemorySize` is a 64-bit value and is what Task Manager reports; the WMI
    field everything else reads is 32 bits and therefore useless above 4 GB.
    """
    if os.name != "nt":
        return {}
    key = ("HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
           "{4d36e968-e325-11ce-bfc1-08002be10318}")
    script = (
        f"Get-ChildItem '{key}' -EA SilentlyContinue | ForEach-Object {{ "
        "$p = Get-ItemProperty $_.PSPath -EA SilentlyContinue; "
        "if ($p.'HardwareInformation.qwMemorySize') { "
        "Write-Output ('VRAM|' + $p.DriverDesc + '|' + "
        "$p.'HardwareInformation.qwMemorySize') } }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, text=True, timeout=8,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        return {}
    found = {}
    for line in (out.stdout or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and parts[0] == "VRAM":
            try:
                found[parts[1].strip().lower()] = int(parts[2]) // (1024 * 1024)
            except ValueError:
                continue
    return found


def _cpu_temperature() -> float:
    """CPU temperature, if the machine will give one.

    Linux exposes it through psutil or /sys; Windows generally does not without
    a driver, so it is simply left out there rather than guessed at.
    """
    try:
        import psutil
        readings = psutil.sensors_temperatures()          # type: ignore[attr-defined]
    except Exception:
        readings = {}
    for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz"):
        entries = readings.get(key) or []
        for entry in entries:
            if entry.current and entry.current > 0:
                return float(entry.current)
    for entry in [e for group in readings.values() for e in group]:
        if entry.current and 20 < entry.current < 120:
            return float(entry.current)
    try:
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            kind = (zone / "type").read_text().strip().lower()
            if "cpu" in kind or "x86" in kind or "soc" in kind:
                return int((zone / "temp").read_text().strip()) / 1000.0
    except (OSError, ValueError):
        pass
    return -1.0


def _shared_budget_mb() -> int:
    """How much system memory the graphics driver may hand to an integrated GPU.

    Windows offers WDDM drivers up to half of physical RAM as shared memory,
    which is where an integrated GPU's working set actually lives. That is the
    figure to size an offload against, not the token amount the adapter claims
    as dedicated.
    """
    try:
        total = Monitor()._mem_fallback()[1]
    except Exception:
        total = 0
    return int(total * 0.5) if total else 0


def _vendor_of(name: str) -> str:
    low = name.lower()
    for needle, vendor in (("nvidia", "nvidia"), ("geforce", "nvidia"),
                           ("radeon", "amd"), ("amd", "amd"),
                           ("intel", "intel"), ("iris", "intel"), ("arc", "intel"),
                           ("apple", "apple")):
        if needle in low:
            return vendor
    return ""


PCI_VENDORS = {"0x8086": "intel", "0x1002": "amd", "0x10de": "nvidia"}


def _sysfs() -> list[GpuSample]:
    """Linux without a vendor tool. Utilisation is not exposed for integrated
    parts, so the clock is reported instead of inventing a percentage."""
    out: list[GpuSample] = []
    try:
        cards = sorted(Path("/sys/class/drm").glob("card[0-9]"))
    except OSError:
        return out
    for card in cards:
        device = card / "device"
        try:
            vendor_id = (device / "vendor").read_text().strip().lower()
        except OSError:
            continue
        vendor = PCI_VENDORS.get(vendor_id, "")
        name = vendor.capitalize() + " GPU" if vendor else f"GPU ({card.name})"
        try:
            for line in (device / "uevent").read_text().splitlines():
                if line.startswith("DRIVER="):
                    name = f"{name} ({line.split('=', 1)[1]})"
                    break
        except OSError:
            pass
        sample = GpuSample(name=name, vendor=vendor)
        if vendor == "intel" or "integrated" in name.lower():
            sample.shared_mb = _shared_budget_mb()
        for clock in ("gt_act_freq_mhz", "gt_cur_freq_mhz"):
            try:
                sample.temperature = -1.0
                mhz = int((card / clock).read_text().strip())
                sample.name = f"{name} · {mhz} MHz"
                break
            except (OSError, ValueError):
                continue
        out.append(sample)
    return out


def _nvidia() -> list[GpuSample]:
    query = ("name,utilization.gpu,memory.used,memory.total,temperature.gpu")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in (out.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(GpuSample(
                name=parts[0], utilisation=float(parts[1]),
                mem_used_mb=int(float(parts[2])), mem_total_mb=int(float(parts[3])),
                temperature=float(parts[4]) if len(parts) > 4 and parts[4] else -1.0,
                vendor="nvidia"))
        except ValueError:
            continue
    return gpus


def _rocm() -> list[GpuSample]:
    try:
        out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse"],
                             capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return []
    text = out.stdout or ""
    gpus: dict[int, GpuSample] = {}
    for line in text.splitlines():
        m = re.search(r"GPU\[(\d+)\].*?GPU use \(%\)\s*:?\s*(\d+)", line)
        if m:
            g = gpus.setdefault(int(m.group(1)), GpuSample(vendor="amd"))
            g.utilisation = float(m.group(2))
            g.name = g.name or f"AMD GPU {m.group(1)}"
        m = re.search(r"GPU\[(\d+)\].*?(?:Memory use|GPU memory use) \(%\)\s*:?\s*(\d+)", line)
        if m:
            g = gpus.setdefault(int(m.group(1)), GpuSample(vendor="amd"))
            g.mem_total_mb = g.mem_total_mb or 100
            g.mem_used_mb = int(float(m.group(2)))
            g.name = g.name or f"AMD GPU {m.group(1)}"
    return [gpus[k] for k in sorted(gpus)]


def describe() -> str:
    bits = [f"{platform.system()} {platform.machine()}"]
    if HAVE_PSUTIL:
        bits.append("psutil")
    tool = _find_gpu_tool()
    bits.append({"windows": "Windows GPU counters", "sysfs": "Linux sysfs",
                 "apple": "Apple", "": "no GPU interface"}.get(tool, tool))
    return " · ".join(bits)
