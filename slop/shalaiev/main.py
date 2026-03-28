"""Butler — live camera preview with face detection and recognition."""

import argparse

from butler import settings
from butler.pipeline import Pipeline


def _parse_resolution(value: str) -> tuple[int, int]:
    width, height = (int(v) for v in value.split("x"))
    return width, height


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both 'run' and 'enroll' subcommands."""
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Capture resolution WxH (default: from settings)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["scrfd", "ultraface", "none"],
        default=None,
        help="Face detector backend (default: from settings)",
    )


def main():
    parser = argparse.ArgumentParser(description="Butler face recognition system")
    subparsers = parser.add_subparsers(dest="command")

    # 'run' subcommand — live preview with detection/recognition.
    run_parser = subparsers.add_parser("run", help="Live preview with detection/recognition")
    _add_common_args(run_parser)
    run_parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Window title (default: from settings)",
    )
    run_parser.add_argument(
        "--no-recognize",
        action="store_true",
        default=False,
        help="Disable face recognition (detection only)",
    )

    # 'enroll' subcommand — enroll a face into the database.
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a face into the database")
    _add_common_args(enroll_parser)
    enroll_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Person's name to enroll",
    )

    args = parser.parse_args()

    # Default to 'run' when no subcommand given.
    if args.command is None:
        args.command = "run"
        # Set defaults that 'run' would have.
        args.resolution = None
        args.detector = None
        args.title = None
        args.no_recognize = False

    # Resolution: CLI flag → settings.
    resolution = (
        _parse_resolution(args.resolution)
        if args.resolution is not None
        else settings.CAPTURE_RESOLUTION
    )

    # Detector: CLI flag → settings.
    detect = True
    if args.detector == "none":
        detect = False
    elif args.detector is not None:
        settings.FACE_DETECTOR = args.detector

    if args.command == "enroll":
        Pipeline(
            resolution=resolution,
            detect=detect,
            recognize=True,
        ).enroll(args.name)
    else:
        title = args.title or settings.WINDOW_TITLE
        recognize = detect and not args.no_recognize
        Pipeline(
            resolution=resolution,
            window_name=title,
            detect=detect,
            recognize=recognize,
        ).run()


if __name__ == "__main__":
    main()
