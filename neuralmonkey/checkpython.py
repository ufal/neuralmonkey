import sys

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    print("Error:", file=sys.stderr)
    print("Neural Monkey must use Python >= 3.6", file=sys.stderr)
    print("Your Python is", sys.version, sys.executable, file=sys.stderr)
    sys.exit(1)
