from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing computer-vision dependencies. Install them with `pip install -r requirements.txt`."
    ) from exc

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
except ImportError as exc:
    raise SystemExit(
        "Missing Gemini SDK. Install it with `pip install -r requirements.txt`."
    ) from exc


CAMERA_DIR = Path("camera")
DEFAULT_MODEL = "gemini-2.5-flash"
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080
API_IMAGE_MAX_DIM = 1920
API_IMAGE_JPEG_QUALITY = 100
THINKING_BUDGET = 0
FAST_MODEL = "gemini-2.5-flash-lite"
FAST_API_IMAGE_MAX_DIM = 960
FAST_API_IMAGE_JPEG_QUALITY = 55


@dataclass
class GeminiLocalizationResult:
    target_letter: str
    found: bool
    center: dict[str, int] | None
    bounding_box: list[int] | None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    passed: bool
    score: int
    total_checks: int
    checks: dict[str, bool]
    metrics: dict[str, float | int | str | bool]
    reasons: list[str]


@dataclass
class GeminiCallResult:
    response_text: str
    model_used: str
    elapsed_seconds: float
    request_elapsed_seconds: float
    preprocess_elapsed_seconds: float
    api_image_width: int
    api_image_height: int
    api_image_bytes: int


def build_skipped_validation_result(result: GeminiLocalizationResult) -> ValidationResult:
    return ValidationResult(
        passed=True,
        score=0,
        total_checks=0,
        checks={},
        metrics={"skipped": True, "found": result.found},
        reasons=["Classical validation was skipped."],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize a keyboard key with Gemini, validate it heuristically, and save an annotated image."
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
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project for Vertex AI. Defaults to the GOOGLE_CLOUD_PROJECT environment variable.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        help="Google Cloud location for Vertex AI. Defaults to GOOGLE_CLOUD_LOCATION or 'global'.",
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
    image_files = [path for path in camera_dir.iterdir() if path.is_file() and path.suffix.lower() in supported]
    if not image_files:
        raise FileNotFoundError(f"No supported image files were found in {camera_dir}")

    newest = max(image_files, key=lambda path: path.stat().st_mtime)
    return newest.resolve()


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not load image: {image_path}")
    return image


def infer_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type:
        return mime_type
    return "image/jpeg"


def prepare_api_image_part(
    image: np.ndarray,
    *,
    api_max_dim: int = API_IMAGE_MAX_DIM,
    api_jpeg_quality: int = API_IMAGE_JPEG_QUALITY,
) -> tuple[types.Part, int, int, int]:
    if api_max_dim <= 0:
        raise ValueError("--api-max-dim must be a positive integer.")
    if not 1 <= api_jpeg_quality <= 100:
        raise ValueError("--api-jpeg-quality must be within [1, 100].")

    api_image = image
    api_image_height, api_image_width = image.shape[:2]
    max_dim = max(api_image_width, api_image_height)

    if max_dim > api_max_dim:
        scale = api_max_dim / float(max_dim)
        api_image_width = max(1, round(api_image_width * scale))
        api_image_height = max(1, round(api_image_height * scale))
        api_image = cv2.resize(
            image,
            (api_image_width, api_image_height),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded_image = cv2.imencode(
        ".jpg",
        api_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), api_jpeg_quality],
    )
    if not success:
        raise ValueError("OpenCV could not encode the API image.")

    image_bytes = encoded_image.tobytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    return image_part, api_image_width, api_image_height, len(image_bytes)


def parse_target_letters(letter_arg: str) -> list[str]:
    target_letters = [letter.strip().upper() for letter in letter_arg.split(",") if letter.strip()]
    if not target_letters:
        raise ValueError("At least one target letter is required.")

    invalid_letters = [letter for letter in target_letters if len(letter) != 1 or not letter.isalpha()]
    if invalid_letters:
        raise ValueError(
            "Each target letter must be a single alphabetic character, for example A or K."
        )

    deduplicated_letters: list[str] = []
    seen_letters: set[str] = set()
    for letter in target_letters:
        if letter in seen_letters:
            continue
        seen_letters.add(letter)
        deduplicated_letters.append(letter)

    return deduplicated_letters


@lru_cache(maxsize=1)
def build_single_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "center": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            "bounding_box": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
            },
        },
        "required": ["center", "bounding_box"],
    }


@lru_cache(maxsize=16)
def build_response_schema(expected_results: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": build_single_result_schema(),
            },
        },
        "required": ["results"],
    }


def build_gemini_prompt(target_letters: list[str], image_width: int, image_height: int) -> str:
    target_letters_text = ", ".join(target_letters)
    return f"""
Localize keyboard keys in one image.

Target letters: {target_letters_text}
Image size: {image_width}x{image_height}

Return strict JSON only.
- Top-level object: {{"results": [...]}}
- Exactly {len(target_letters)} results, in this exact order: {target_letters_text}
- For each result return only: center, bounding_box
- Coordinates must be integers in [0,1000] over the full image extent, never pixels
- bbox format must be [xmin, ymin, xmax, ymax]
- If a key is not visible: center=null, bounding_box=null
""".strip()


@lru_cache(maxsize=8)
def _get_vertex_client(project: str, location: str) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


def call_gemini(
    image: np.ndarray,
    image_width: int,
    image_height: int,
    target_letters: list[str],
    model: str,
    fallback_models: list[str],
    project: str | None,
    location: str,
    api_max_dim: int = API_IMAGE_MAX_DIM,
    api_jpeg_quality: int = API_IMAGE_JPEG_QUALITY,
) -> GeminiCallResult:
    if not project:
        raise ValueError("Missing Google Cloud project. Set GOOGLE_CLOUD_PROJECT or pass --project.")

    start_time = time.perf_counter()
    preprocess_start_time = time.perf_counter()
    image_part, api_image_width, api_image_height, api_image_bytes = prepare_api_image_part(
        image,
        api_max_dim=api_max_dim,
        api_jpeg_quality=api_jpeg_quality,
    )
    prompt = build_gemini_prompt(
        target_letters=target_letters,
        image_width=image_width,
        image_height=image_height,
    )
    preprocess_elapsed_seconds = time.perf_counter() - preprocess_start_time

    client = _get_vertex_client(project, location)

    model_candidates = [model, *fallback_models]
    seen_models: set[str] = set()
    last_error: Exception | None = None
    response_schema = build_response_schema(expected_results=len(target_letters))

    for index, candidate_model in enumerate(model_candidates, start=1):
        candidate_model = candidate_model.strip()
        if not candidate_model or candidate_model in seen_models:
            continue
        seen_models.add(candidate_model)

        try:
            print(f"Calling Gemini with model {candidate_model} (attempt {index}/{len(model_candidates)})...")
            request_start_time = time.perf_counter()
            response = client.models.generate_content(
                model=candidate_model,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
                ),
            )
            if not response.text:
                raise RuntimeError(f"Gemini model {candidate_model} returned an empty response.")
            return GeminiCallResult(
                response_text=response.text,
                model_used=candidate_model,
                elapsed_seconds=time.perf_counter() - start_time,
                request_elapsed_seconds=time.perf_counter() - request_start_time,
                preprocess_elapsed_seconds=preprocess_elapsed_seconds,
                api_image_width=api_image_width,
                api_image_height=api_image_height,
                api_image_bytes=api_image_bytes,
            )
        except genai_errors.ServerError as exc:
            last_error = exc
            if _is_retryable_unavailable_error(exc) and index < len(model_candidates):
                print(
                    f"Model {candidate_model} is temporarily overloaded (503). "
                    f"Retrying with the next configured model..."
                )
                time.sleep(1.0)
                continue
            raise RuntimeError(_format_gemini_api_error(candidate_model, exc)) from exc
        except genai_errors.APIError as exc:
            last_error = exc
            raise RuntimeError(_format_gemini_api_error(candidate_model, exc)) from exc

    raise RuntimeError("Gemini call failed without a usable response.") from last_error

def _is_retryable_unavailable_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 503:
        return True

    details = f"{getattr(error, 'message', '')} {error}".upper()
    return "503" in details or "UNAVAILABLE" in details or "HIGH DEMAND" in details


def _format_gemini_api_error(model_name: str, error: Exception) -> str:
    status_code = getattr(error, "status_code", "unknown")
    details = getattr(error, "message", None) or str(error)
    return (
        f"Gemini request failed for model `{model_name}` "
        f"(status: {status_code}). Details: {details}"
    )


def _parse_single_gemini_result(
    payload: dict[str, Any],
    image_width: int,
    image_height: int,
    expected_letter: str,
) -> GeminiLocalizationResult:
    required_keys = {"center", "bounding_box"}
    missing = required_keys.difference(payload.keys())
    if missing:
        raise ValueError(f"Gemini JSON is missing required keys: {sorted(missing)}")

    center = payload["center"]
    if center is not None:
        if not isinstance(center, dict) or {"x", "y"} - set(center.keys()):
            raise ValueError("Gemini JSON field 'center' must be an object with x and y.")
        center = {"x": int(center["x"]), "y": int(center["y"])}

    bounding_box = payload["bounding_box"]
    if bounding_box is not None:
        if not isinstance(bounding_box, list) or len(bounding_box) != 4:
            raise ValueError("Gemini JSON field 'bounding_box' must be a list of four integers.")
        bounding_box = [int(value) for value in bounding_box]

    if (center is None) != (bounding_box is None):
        raise ValueError("Gemini must return center and bounding_box together, or both null.")

    found = center is not None and bounding_box is not None

    result = GeminiLocalizationResult(
        target_letter=expected_letter,
        found=found,
        center=center,
        bounding_box=bounding_box,
        raw_response=payload,
    )

    result = _convert_normalized_to_pixel_coordinates(
        result,
        image_width=image_width,
        image_height=image_height,
    )
    _validate_localization_payload(result, image_width=image_width, image_height=image_height)
    return result


def parse_gemini_response(
    raw_text: str,
    image_width: int,
    image_height: int,
    expected_letters: list[str],
) -> list[GeminiLocalizationResult]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini response is not valid JSON: {exc}") from exc

    if isinstance(payload, dict) and "results" in payload:
        raw_results = payload["results"]
        if not isinstance(raw_results, list):
            raise ValueError("Gemini JSON field 'results' must be a list.")
    elif isinstance(payload, dict):
        raw_results = [payload]
    else:
        raise ValueError("Gemini response JSON must be an object.")

    if len(raw_results) != len(expected_letters):
        raise ValueError(
            f"Gemini returned {len(raw_results)} result(s), expected {len(expected_letters)}."
        )

    return [
        _parse_single_gemini_result(
            payload=result_payload,
            image_width=image_width,
            image_height=image_height,
            expected_letter=expected_letter,
        )
        for result_payload, expected_letter in zip(raw_results, expected_letters)
    ]


def _convert_normalized_to_pixel_coordinates(
    result: GeminiLocalizationResult,
    image_width: int,
    image_height: int,
) -> GeminiLocalizationResult:
    if not result.found:
        return result

    if result.center is None or result.bounding_box is None:
        raise ValueError("Gemini marked the key as found but center or bounding_box is null.")

    _validate_normalized_localization_payload(result)

    xmin, ymin, xmax, ymax = result.bounding_box
    return GeminiLocalizationResult(
        target_letter=result.target_letter,
        found=result.found,
        center={
            "x": _scale_normalized_center(result.center["x"], image_width),
            "y": _scale_normalized_center(result.center["y"], image_height),
        },
        bounding_box=[
            _scale_normalized_min(xmin, image_width),
            _scale_normalized_min(ymin, image_height),
            _scale_normalized_max(xmax, image_width),
            _scale_normalized_max(ymax, image_height),
        ],
        raw_response=result.raw_response,
    )


def _validate_normalized_localization_payload(result: GeminiLocalizationResult) -> None:
    if result.center is None or result.bounding_box is None:
        raise ValueError("Gemini localization is missing center or bounding_box.")

    x = result.center["x"]
    y = result.center["y"]
    xmin, ymin, xmax, ymax = result.bounding_box

    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError("Gemini center must use normalized coordinates in [0, 1000].")
    if not all(0 <= value <= 1000 for value in result.bounding_box):
        raise ValueError("Gemini bounding_box must use normalized coordinates in [0, 1000].")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Gemini bounding_box must be ordered as [xmin, ymin, xmax, ymax].")
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        raise ValueError("Gemini center must fall inside the normalized bounding_box.")


def _scale_normalized_center(value: int, axis_size: int) -> int:
    return max(0, min(axis_size - 1, round((value / 1000.0) * axis_size)))


def _scale_normalized_min(value: int, axis_size: int) -> int:
    return max(0, min(axis_size - 1, round((value / 1000.0) * axis_size)))


def _scale_normalized_max(value: int, axis_size: int) -> int:
    return max(1, min(axis_size, round((value / 1000.0) * axis_size)))


def _validate_localization_payload(
    result: GeminiLocalizationResult,
    image_width: int,
    image_height: int,
) -> None:
    if not result.found:
        return

    if result.center is None or result.bounding_box is None:
        raise ValueError("Gemini marked the key as found but center or bounding_box is null.")

    x = result.center["x"]
    y = result.center["y"]
    xmin, ymin, xmax, ymax = result.bounding_box

    if not (0 <= x < image_width and 0 <= y < image_height):
        raise ValueError("Gemini center is outside image bounds.")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Gemini bounding box has invalid ordering.")


def classical_validation(image: np.ndarray, result: GeminiLocalizationResult) -> ValidationResult:
    image_height, image_width = image.shape[:2]

    if not result.found or result.center is None or result.bounding_box is None:
        return ValidationResult(
            passed=False,
            score=0,
            total_checks=0,
            checks={},
            metrics={"found": result.found},
            reasons=["Gemini did not return a usable key localization."],
        )

    xmin, ymin, xmax, ymax = result.bounding_box
    center_x, center_y = result.center["x"], result.center["y"]

    box_in_bounds = 0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height
    box_width = max(0, xmax - xmin)
    box_height = max(0, ymax - ymin)
    area = box_width * box_height
    area_fraction = area / float(image_width * image_height)
    aspect_ratio = box_width / float(box_height) if box_height else float("inf")
    center_inside_box = xmin <= center_x <= xmax and ymin <= center_y <= ymax

    checks: dict[str, bool] = {
        "box_in_bounds": box_in_bounds,
        "center_inside_box": center_inside_box,
        "area_plausible": 0.0005 <= area_fraction <= 0.12,
        "aspect_ratio_plausible": 0.35 <= aspect_ratio <= 4.0,
    }
    reasons: list[str] = []
    metrics: dict[str, float | int | str | bool] = {
        "box_width": box_width,
        "box_height": box_height,
        "area_fraction": round(area_fraction, 5),
        "aspect_ratio": round(aspect_ratio, 4) if np.isfinite(aspect_ratio) else "inf",
    }

    if not box_in_bounds:
        reasons.append("Bounding box falls outside the image.")
    if box_in_bounds and area > 0:
        crop = image[ymin:ymax, xmin:xmax]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        intensity_std = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        checks["texture_contrast"] = intensity_std >= 10.0
        checks["edge_density_plausible"] = 0.01 <= edge_density <= 0.45
        checks["sharpness_plausible"] = laplacian_var >= 20.0

        metrics.update(
            {
                "intensity_std": round(intensity_std, 3),
                "edge_density": round(edge_density, 4),
                "laplacian_variance": round(laplacian_var, 3),
            }
        )

        if not checks["texture_contrast"]:
            reasons.append("The predicted crop has very low intensity variation.")
        if not checks["edge_density_plausible"]:
            reasons.append("The predicted crop edge density looks unlike a distinct key region.")
        if not checks["sharpness_plausible"]:
            reasons.append("The predicted crop is too flat or blurry for a key-sized object.")
    else:
        reasons.append("The predicted crop could not be extracted for heuristic checks.")

    hard_checks = [
        "box_in_bounds",
        "center_inside_box",
        "area_plausible",
        "aspect_ratio_plausible",
    ]
    soft_checks = [name for name in checks if name not in hard_checks]
    hard_pass = all(checks.get(name, False) for name in hard_checks)
    soft_score = sum(1 for name in soft_checks if checks.get(name, False))
    total_checks = len(checks)
    score = sum(1 for passed in checks.values() if passed)
    passed = hard_pass and soft_score >= 2

    if not checks.get("area_plausible", False):
        reasons.append("The bounding box area is implausible for a single key.")
    if not checks.get("aspect_ratio_plausible", False):
        reasons.append("The bounding box aspect ratio is implausible for a key.")
    if not checks.get("center_inside_box", False):
        reasons.append("The returned center does not fall inside the bounding box.")

    if passed:
        reasons.append("Heuristic validation passed.")

    return ValidationResult(
        passed=passed,
        score=score,
        total_checks=total_checks,
        checks=checks,
        metrics=metrics,
        reasons=reasons,
    )


def draw_overlay(
    image: np.ndarray,
    result: GeminiLocalizationResult,
    validation: ValidationResult,
    *,
    text_y0: int = 24,
    copy_image: bool = True,
) -> np.ndarray:
    validation_skipped = bool(validation.metrics.get("skipped", False))
    annotated = image.copy() if copy_image else image
    if validation_skipped:
        status_color = (0, 220, 255)
        cv_status = "SKIP"
    else:
        status_color = (0, 200, 0) if validation.passed else (0, 0, 255)
        cv_status = "PASS" if validation.passed else "FAIL"

    if result.found and result.bounding_box is not None:
        xmin, ymin, xmax, ymax = result.bounding_box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), status_color, 2)

    if result.found and result.center is not None:
        cv2.circle(annotated, (result.center["x"], result.center["y"]), 5, (255, 140, 0), -1)

    text_lines = [
        f"Letter: {result.target_letter}",
        f"Found: {result.found}",
        f"CV check: {cv_status}",
    ]

    y0 = text_y0
    for line in text_lines:
        cv2.putText(
            annotated,
            line,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
            cv2.LINE_AA,
        )
        y0 += 24

    return annotated


def build_output_path(image_path: Path, target_letters: list[str], output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).resolve()

    target_letters_suffix = "_".join(target_letters)
    return (image_path.parent / f"annotated_{image_path.stem}_{target_letters_suffix}.jpg").resolve()


def save_image(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise IOError(f"Failed to save annotated image to {output_path}")


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


def parse_fallback_models(fallback_models_arg: str) -> list[str]:
    return [model.strip() for model in fallback_models_arg.split(",") if model.strip()]


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
        io_elapsed_seconds = time.perf_counter() - io_start_time

        gemini_call = call_gemini(
            image=image,
            image_width=image_width,
            image_height=image_height,
            target_letters=target_letters,
            model=args.model,
            fallback_models=fallback_models,
            project=args.project,
            location=args.location,
            api_max_dim=args.api_max_dim,
            api_jpeg_quality=args.api_jpeg_quality,
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