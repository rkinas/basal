"""What the command-line tools share: counts that must be positive."""
import argparse


def positive(s):
    """An argparse type: an integer >= 1 (a count of 0 would wait for ever or divide by zero)."""
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f"{s}: must be at least 1")
    return v
