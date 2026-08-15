from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    if "--cli" in argv or "--headless" in argv:
        from .cli import main as cli_main
        return cli_main([a for a in argv if a not in ("--headless",)])
    if argv and argv[0] == "node":
        from .node import main as node_main
        return node_main(argv[1:])
    try:
        from .ui.app import main as gui_main
    except ImportError as e:
        print(f"The interface needs PySide6, which isn't installed ({e}).\n"
              "Run install.sh / install.bat, or use --cli for the headless mode.")
        return 1
    return gui_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
