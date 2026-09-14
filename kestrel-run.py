"""Entry point for the built Kestrel.exe.

Two jobs, both of which exist because of how Windows treats a program's
identity.

The first is the icon. A .py file launched through the Python interpreter is,
as far as Windows is concerned, Python running — the taskbar shows Python's
icon and groups Kestrel's window with any other Python program. Nothing inside
the application can correct that, because the identity belongs to the process,
not the window. A real executable with its own icon is the only reliable fix.

The second is updating. A frozen executable normally carries its own copy of
every module, so pulling a new version of the source would change nothing until
the exe was rebuilt. The import hook below prefers files on disk beside the
executable, so an update is a git pull as it always was, and the exe is only
rebuilt when the launcher itself changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _install_dir() -> Path:
    """Where Kestrel's own files live, frozen or not."""
    if getattr(sys, "frozen", False):
        # The folder holding the exe, not the temporary unpack directory: the
        # point is to find the source the user can edit and update.
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


class _DiskFirst:
    """Import Kestrel's modules from disk in preference to the bundle.

    Registered ahead of PyInstaller's own finder. Only `kestrel` and its
    submodules are affected — everything else, PySide6 included, still comes
    from the bundle, which is what keeps the executable self-contained.
    """

    def __init__(self, root: Path):
        self.root = root

    def find_module(self, name, path=None):        # legacy API, still called
        return None

    def find_spec(self, name, path=None, target=None):
        if name != "kestrel" and not name.startswith("kestrel."):
            return None
        import importlib.machinery as machinery
        import importlib.util as util

        relative = name.split(".")
        candidate = self.root.joinpath(*relative)
        package = candidate / "__init__.py"
        module = candidate.with_suffix(".py")
        if package.is_file():
            spec = util.spec_from_file_location(
                name, package,
                submodule_search_locations=[str(candidate)],
                loader=machinery.SourceFileLoader(name, str(package)))
            return spec
        if module.is_file():
            return util.spec_from_file_location(
                name, module,
                loader=machinery.SourceFileLoader(name, str(module)))
        return None


def main() -> int:
    root = _install_dir()
    if (root / "kestrel" / "__init__.py").is_file():
        sys.meta_path.insert(0, _DiskFirst(root))
        os.chdir(root)

    # Windows groups taskbar buttons by application identity, and a program
    # that does not claim one inherits the interpreter's. Set before any window
    # exists, and only ever to Kestrel's own — stamping an identity onto
    # anything else is how other applications' pinned shortcuts get broken.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "AlterniTech.Kestrel")
        except Exception:
            pass

    from kestrel.__main__ import main as kestrel_main
    return kestrel_main()


if __name__ == "__main__":
    raise SystemExit(main())
