import sys

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    raise Exception("Must be using Python >= 3.5")
