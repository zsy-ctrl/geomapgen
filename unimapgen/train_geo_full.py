import argparse

from unimapgen.geo.errors import run_with_geo_error_boundary
from unimapgen.train_geo_model import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_training(config_path=args.config, mode_override="full")


if __name__ == "__main__":
    run_with_geo_error_boundary(main, default_code="GEO-1000")
