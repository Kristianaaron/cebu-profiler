"""SM121 hardware-friendly width vocabulary (blueprint §14.2).

Atlas should only emit expert widths from this supported set; it can change
after CUTLASS/SM121 profiling, but never silently.
"""

from __future__ import annotations

SM121_WIDTH_BUCKETS: list[int] = [2048, 1920, 1792, 1664, 1536, 1408, 1280]
