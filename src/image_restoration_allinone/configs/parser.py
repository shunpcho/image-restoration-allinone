import argparse
from pathlib import Path

from fvcore.common.config import CfgNode

from image_restoration_allinone.configs.default import get_default_cfg


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Path to config file.")
    parser.add_argument("options", nargs=argparse.REMAINDER, help="Override config options using the command line.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> CfgNode:
    """Load the configuration from a file and override with command line options."""
    cfg = get_default_cfg()

    if args.config is not None:
        cfg.merge_from_file(args.config)
    if args.options:
        cfg.merge_from_list(args.options)

    cfg.freeze()

    return cfg
