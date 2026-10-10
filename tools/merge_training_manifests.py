"""CLI wrapper for the WebUI training dataset composer."""

from maichart.training.dataset_catalog import *  # noqa: F401,F403
from maichart.training.dataset_catalog import main


if __name__ == "__main__":
    raise SystemExit(main())
