"""Backward-compatible script entry point.

Use ``python main.py`` to behave the same as ``python -m code_radar``.
"""

from code_radar.__main__ import main


if __name__ == "__main__":
    main()
