from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "Missing computer-vision dependencies. Install them with `pip install -r requirements.txt`."
    ) from exc

try:
    from .gemini_keyboard_localizer import (
        API_IMAGE_JPEG_QUALITY,
        API_IMAGE_MAX_DIM,
        GEMINI_BACKENDS,
        GeminiLocalizationResult,
        ValidationResult,
        build_skipped_validation_result,
        call_gemini,
        classical_validation,
        draw_overlay,
        parse_gemini_response,
        parse_target_letters,
        preprocess_for_gemini,
    )
except ImportError:
    from gemini_keyboard_localizer import (
        API_IMAGE_JPEG_QUALITY,
        API_IMAGE_MAX_DIM,
        GEMINI_BACKENDS,
        GeminiLocalizationResult,
        ValidationResult,
        build_skipped_validation_result,
        call_gemini,
        classical_validation,
        draw_overlay,
        parse_gemini_response,
        parse_target_letters,
        preprocess_for_gemini,
    )


CAMERA_DIR = Path("camera")
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
# DEFAULT_MODEL = "gemini-3-flash-preview"
FAST_MODEL = "gemini-2.5-flash"
FAST_API_IMAGE_MAX_DIM = 960
FAST_API_IMAGE_JPEG_QUALITY = 55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize keyboard keys with Gemini, validate them, and save an annotated image."
    )
    parser.add_argument(
        "--letter",
        required=True,
        help="Target keyboard letter(s) to localize, for example A or Q,K,L.",
    )
    parser.add_argument(
        "--image",
        help="Optional image filename or path. If omitted, the newest image from camera/ is used.",
    )
    parser.add_argument(
        "--camera-dir",
        default=str(CAMERA_DIR),
        help="Directory that contains the input image(s). Default: camera/",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for the annotated image. Default: camera/annotated_<image>_<letters>.jpg",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--fallback-models",
        default="gemini-2.5-flash-lite",
        help="Comma-separated fallback Gemini models tried after --model if the API is unavailable.",
    )
    parser.add_argument(
        "--gemini-backend",
        choices=sorted(GEMINI_BACKENDS),
        default="standard",
        help=(
            "Vertex AI Gemini request mode: standard PayGo, Priority PayGo, "
            "or Provisioned Throughput. Default: standard."
        ),
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project for Vertex AI. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        help="Google Cloud location for Vertex AI. Defaults to GOOGLE_CLOUD_LOCATION or global.",
    )
    parser.add_argument(
        "--api-max-dim",
        type=int,
        default=API_IMAGE_MAX_DIM,
        help=(
            "Maximum image dimension sent to Gemini. Lower values are faster but can reduce "
            f"accuracy. Default: {API_IMAGE_MAX_DIM}"
        ),
    )
    parser.add_argument(
        "--api-jpeg-quality",
        type=int,
        default=API_IMAGE_JPEG_QUALITY,
        help=(
            "JPEG quality used for the image sent to Gemini, in [1,100]. Lower values are "
            f"smaller/faster but more lossy. Default: {API_IMAGE_JPEG_QUALITY}"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Speed-oriented preset: uses a smaller image, stronger JPEG compression, and "
            f"switches the default model to {FAST_MODEL}."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the classical OpenCV validation step for faster local post-processing.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Do not save the annotated output image.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print a more detailed timing breakdown of local and Gemini steps.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert the image to grayscale before sending it to Gemini.",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply light CLAHE contrast enhancement before sending it to Gemini.",
    )
    return parser.parse_args()


def resolve_image_path(camera_dir: Path, image_arg: str | None) -> Path:
    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")

    if image_arg:
        candidate = Path(image_arg)
        if candidate.is_file():
            return candidate.resolve()

        candidate_in_camera = camera_dir / image_arg
        if candidate_in_camera.is_file():
            return candidate_in_camera.resolve()

        raise FileNotFoundError(f"Image not found: {image_arg}")

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        path
        for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    ]
    if not image_files:
        raise FileNotFoundError(f"No supported image files were found in {camera_dir}")

    newest = max(image_files, key=lambda path: path.stat().st_mtime)
    return newest.resolve()


def load_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not load image: {image_path}")
    return image


def build_output_path(image_path: Path, target_letters: list[str], output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).resolve()

    target_letters_suffix = "_".join(target_letters)
    return (image_path.parent / f"annotated_{image_path.stem}_{target_letters_suffix}.jpg").resolve()


def save_image(output_path: Path, image) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise IOError(f"Failed to save annotated image to {output_path}")


def parse_fallback_models(fallback_models_arg: str) -> list[str]:
    return [model.strip() for model in fallback_models_arg.split(",") if model.strip()]


def print_results(results: list[GeminiLocalizationResult], validations: list[ValidationResult]) -> None:
    for index, (result, validation) in enumerate(zip(results, validations), start=1):
        if index > 1:
            print()
        print(f"Gemini localization result ({result.target_letter}):")
        print(
            json.dumps(
                {
                    "center": result.center,
                    "bounding_box": result.bounding_box,
                },
                indent=2,
            )
        )
        print()
        print(f"Classical validation result ({result.target_letter}):")
        print(json.dumps(asdict(validation), indent=2))


def main() -> None:
    try:
        total_start_time = time.perf_counter()
        args = parse_args()
        if args.fast:
            if args.model == DEFAULT_MODEL:
                args.model = FAST_MODEL
            args.api_max_dim = min(args.api_max_dim, FAST_API_IMAGE_MAX_DIM)
            args.api_jpeg_quality = min(args.api_jpeg_quality, FAST_API_IMAGE_JPEG_QUALITY)

        target_letters = parse_target_letters(args.letter)

        io_start_time = time.perf_counter()
        camera_dir = Path(args.camera_dir).resolve()
        image_path = resolve_image_path(camera_dir=camera_dir, image_arg=args.image)
        image = load_image(image_path)
        image_height, image_width = image.shape[:2]
        fallback_models = parse_fallback_models(args.fallback_models)
        gemini_image = preprocess_for_gemini(
            image,
            use_grayscale=args.grayscale,
            use_clahe=args.clahe,
        )
        io_elapsed_seconds = time.perf_counter() - io_start_time

        if args.grayscale or args.clahe:
            enabled_steps: list[str] = []
            if args.grayscale:
                enabled_steps.append("grayscale")
            if args.clahe:
                enabled_steps.append("clahe")
            print(f"Gemini preprocessing enabled: {', '.join(enabled_steps)}")

        gemini_call = call_gemini(
            image=gemini_image,
            image_width=image_width,
            image_height=image_height,
            target_letters=target_letters,
            model=args.model,
            fallback_models=fallback_models,
            project=args.project,
            location=args.location,
            api_max_dim=args.api_max_dim,
            api_jpeg_quality=args.api_jpeg_quality,
            gemini_backend=args.gemini_backend,
        )
        print(f"Gemini response received from model: {gemini_call.model_used}")
        print(
            "Gemini API image: "
            f"{gemini_call.api_image_width}x{gemini_call.api_image_height}, "
            f"{gemini_call.api_image_bytes / 1024.0:.1f} KB"
        )
        print(f"Gemini request time: {gemini_call.request_elapsed_seconds:.2f} seconds")
        print(f"Gemini total call time: {gemini_call.elapsed_seconds:.2f} seconds")

        postprocess_start_time = time.perf_counter()
        localizations = parse_gemini_response(
            gemini_call.response_text,
            image_width=image_width,
            image_height=image_height,
            expected_letters=target_letters,
        )
        if args.skip_validation:
            validations = [build_skipped_validation_result(localization) for localization in localizations]
        else:
            validations = [classical_validation(image, localization) for localization in localizations]

        annotated = image.copy()
        for index, (localization, validation) in enumerate(zip(localizations, validations)):
            annotated = draw_overlay(
                annotated,
                localization,
                validation,
                text_y0=24 + (index * 120),
                copy_image=False,
            )
        postprocess_elapsed_seconds = time.perf_counter() - postprocess_start_time

        output_path: Path | None = None
        save_elapsed_seconds = 0.0
        if not args.skip_save:
            save_start_time = time.perf_counter()
            output_path = build_output_path(
                image_path=image_path,
                target_letters=target_letters,
                output_arg=args.output,
            )
            save_image(output_path, annotated)
            save_elapsed_seconds = time.perf_counter() - save_start_time

        print_results(localizations, validations)
        print()
        if output_path is not None:
            print(f"Annotated image saved to: {output_path}")
        else:
            print("Annotated image saving skipped.")

        if args.profile:
            total_elapsed_seconds = time.perf_counter() - total_start_time
            print()
            print("Timing breakdown:")
            print(f"- Local image load: {io_elapsed_seconds:.3f} s")
            print(f"- Gemini preprocess (resize/encode): {gemini_call.preprocess_elapsed_seconds:.3f} s")
            print(f"- Gemini request: {gemini_call.request_elapsed_seconds:.3f} s")
            print(f"- Local parse/validate/draw: {postprocess_elapsed_seconds:.3f} s")
            print(f"- Save image: {save_elapsed_seconds:.3f} s")
            print(f"- End-to-end total: {total_elapsed_seconds:.3f} s")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
