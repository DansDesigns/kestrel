#!/usr/bin/env python3
"""Turn per-device memory figures into a --tensor-split argument.

    python3 split.py 24576 8192 16384
    --tensor-split 0.500,0.167,0.333
"""
import sys

def main(argv):
    try:
        mem = [float(a) for a in argv]
    except ValueError:
        print(__doc__)
        return 1
    if not mem or min(mem) <= 0:
        print("Give one positive memory figure (MB) per device, in device order:")
        print("local devices first, then each --rpc host in the order you list them.")
        return 1
    total = sum(mem)
    print("--tensor-split " + ",".join(f"{m / total:.3f}" for m in mem))
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
