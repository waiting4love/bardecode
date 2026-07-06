import argparse
import sys
from . import __version__
from .pipeline import process_image
from .image_io import format_report_json
from .schemas import Report, ImageResult


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bardecode", description="Detect and decode barcodes from photos.")
    p.add_argument("images", nargs="+", help="image file path(s)")
    p.add_argument("--conf-threshold", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.45)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--model", type=str, default=None, help="ONNX model path override")
    p.add_argument("--formats", type=str, default=None, help="comma-separated format whitelist")
    p.add_argument("--no-fallback", action="store_true", help="disable full-image decode fallback")
    p.add_argument("-q", "--quiet", action="store_true", help="print only barcode values")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose to stderr")
    p.add_argument("--version", action="version", version=f"bardecode {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    formats = [s.strip() for s in args.formats.split(",")] if args.formats else None

    from .detector import BarcodeDetector
    detector = BarcodeDetector(
        model_path=args.model,
        img_size=args.img_size,
        conf_thres=args.conf_threshold,
        iou_thres=args.iou_threshold,
    )

    results = []
    exit_code = 0
    for path in args.images:
        try:
            ir = process_image(
                path,
                formats=formats,
                fallback=not args.no_fallback,
                detector=detector,
            )
        except Exception as e:
            ir = ImageResult(image=path, width=0, height=0, barcodes=[],
                             undecoded=[], error=f"unexpected: {e}")
        results.append(ir)
        if args.verbose:
            print(f"[bardecode] {path}: {len(ir.barcodes)} decoded, {len(ir.undecoded)} undecoded, error={ir.error}", file=sys.stderr)
        if ir.error == "file not found":
            exit_code = max(exit_code, 2)
        elif ir.error:
            exit_code = max(exit_code, 1)

    report = Report(tool="bardecode", version=__version__, results=results)
    out = format_report_json(report, quiet=args.quiet)
    if out:
        print(out)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
