from argparse import ArgumentParser
from pathlib import Path

from dvf_maintenance.clean import clean
from dvf_maintenance.store import store


def main() -> None:
    parser = ArgumentParser(
        description="Provides usefull tools to transform DVFs datasets"
    )
    subparsers = parser.add_subparsers(required=True, dest="command")

    clean_parser = subparsers.add_parser("clean", help="Clean DVF dataset")
    clean_parser.add_argument(
        "path", type=str, help="Path to the CSV for the DVF dataset"
    )
    clean_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Path to the new parquet dataset",
    )
    clean_parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="csv",
        help="Input format : csv, parquet or pickle, default : csv",
    )

    store_parser = subparsers.add_parser("store", help="Store DVF dataset in parquet")
    store_parser.add_argument(
        "path", type=str, help="Path to the CSV for the DVF dataset"
    )
    store_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Path to the new parquet dataset",
    )
    store_parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="csv",
        help="Input format : csv, parquet or pickle, default : csv",
    )

    args = parser.parse_args()
    match args.command:
        case "clean":
            if args.output is None:
                args.output = (
                    Path(args.path)
                    .with_stem(Path(args.path).stem + "_cleaned")
                    .with_suffix(".parquet")
                )
            elif Path(args.output).is_dir():
                args.output = Path(args.output) / Path(args.path).name
                args.output = args.output.with_stem(
                    args.output.stem + "_cleaned"
                ).with_suffix(".parquet")
            clean(args.path, args.output, args.format)
        case "store":
            if args.output is None:
                args.output = Path(args.path).with_suffix(".parquet")
            elif Path(args.output).is_dir():
                args.output = Path(args.output) / Path(args.path).name
                args.output = args.output.with_suffix(".parquet")
            store(args.path, args.output, args.format)
