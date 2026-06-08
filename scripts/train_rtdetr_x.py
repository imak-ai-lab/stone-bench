"""Convenience wrapper for local RT-DETR-X training."""

from __future__ import annotations

import sys

import train_supervised


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--config", "configs/supervised/rtdetr_x.yaml", *sys.argv[1:]]
    raise SystemExit(train_supervised.main())
