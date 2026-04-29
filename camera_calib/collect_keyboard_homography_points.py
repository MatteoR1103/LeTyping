from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError as exc:
    raise SystemExit("Missing Gemini SDK. Install google-genai in your conda environment.") from exc


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODELS = "gemini-2.5-flash-lite"
DEFAULT_OUTPUT = Path("camera_calib/letter_points_pixel_world.json")
DEFAULT_ANNOTATED_OUTPUT = Path("camera_calib/letter_homography_points.jpg")
DEFAULT_HOMOGRAPHY_DIR = Path("camera_calib/calibrations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemini to localize keyboard letter centers in pixels, match them to known "
            "world letter positions, and fit a planar homography."
        )
    )
    parser.add_argument(
        "--world-points",
        type=Path,
        required=True,
        help="JSON with known world positions for keyboard letters.",
    )
    parser.add_argument(
        "--letters",
        help="Optional comma-separated subset of letters to use, e.g. Q,W,E,A,S,D,Z,X,C.",
    )
    parser.add_argument("--image", type=Path, help="Optional image path. If omitted, capture from --camera.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index. Default: 0.")
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="OpenCV camera backend. Default: auto.",
    )
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud project ID.")
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        help="Google Cloud location. Default: GOOGLE_CLOUD_LOCATION or global.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model. Default: {DEFAULT_MODEL}.")
    parser.add_argument(
        "--fallback-models",
        default=DEFAULT_FALLBACK_MODELS,
        help="Comma-separated fallback models. Default: gemini-2.5-flash-lite.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output correspondences JSON.")
    parser.add_argument(
        "--annotated-output",
        type=Path,
        default=DEFAULT_ANNOTATED_OUTPUT,
        help="Annotated image output path.",
    )
    parser.add_argument(
        "--homography-dir",
        type=Path,
        default=DEFAULT_HOMOGRAPHY_DIR,
        help="Directory for homography .npy/.txt outputs.",
    )
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=5.0,
        help="RANSAC threshold in pixels for homography. Default: 5.",
    )
    parser.add_argument("--min-points", type=int, default=4, help="Minimum detected letter pairs required.")
    parser.add_argument("--api-max-dim", type=int, default=1280, help="Max image dimension sent to Gemini.")
    parser.add_argument("--api-jpeg-quality", type=int, default=90, help="JPEG quality sent to Gemini.")
    parser.add_argument("--skip-homography", action="store_true", help="Only save correspondences.")
    return parser.parse_args()


def load_world_points(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"World points file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("letters"), dict):
        raw_points = data["letters"]
    elif isinstance(data, dict) and isinstance(data.get("points"), list):
        raw_points = rows_to_letter_points(data["points"])
    elif isinstance(data, list):
        raw_points = rows_to_letter_points(data)
    elif isinstance(data, dict):
        raw_points = data
    else:
        raise ValueError("Unsupported world point JSON format.")

    world_points: dict[str, list[float]] = {}
    for key, value in raw_points.items():
        letter = normalize_letter(key)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError(f"World point for {letter} must be [x, y] or [x, y, z].")
        world_points[letter] = [
            float(value[0]),
            float(value[1]),
            float(value[2]) if len(value) >= 3 else 0.0,
        ]

    if len(world_points) < 4:
        raise ValueError(f"Need at least 4 world letter points, got {len(world_points)}.")
    return dict(sorted(world_points.items()))


def rows_to_letter_points(rows: list[Any]) -> dict[str, Any]:
    points: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Point rows must be objects/dicts.")
        letter = row.get("letter") or row.get("name") or row.get("key")
        if letter is None:
            raise ValueError(f"Missing letter/name/key in row: {row}")
        value = row.get("world") or row.get("xyz") or row.get("xy")
        if value is None and all(k in row for k in ("x", "y")):
            value = [row["x"], row["y"], row.get("z", 0.0)]
        if value is None:
            raise ValueError(f"Missing world point in row: {row}")
        points[normalize_letter(letter)] = value
    return points


def normalize_letter(value: Any) -> str:
    letter = str(value).strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError(f"Expected a single alphabetic letter, got {value!r}")
    return letter


def parse_letters_arg(letters_arg: str | None, world_points: dict[str, list[float]]) -> list[str]:
    if letters_arg is None:
        return list(world_points.keys())

    letters = [normalize_letter(part) for part in letters_arg.split(",") if part.strip()]
    if not letters:
        raise ValueError("--letters was provided but no valid letters were found.")

    missing = [letter for letter in letters if letter not in world_points]
    if missing:
        raise ValueError(f"These letters are not present in --world-points: {missing}")
    return list(dict.fromkeys(letters))


def resolve_capture_backend(name: str) -> int:
    normalized = name.strip().lower()
    if normalized in {"auto", "any"}:
        return cv.CAP_ANY
    if normalized == "dshow":
        return cv.CAP_DSHOW
    if normalized == "msmf":
        return cv.CAP_MSMF
    raise ValueError(f"Unsupported backend: {name}")


def load_or_capture_image(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.image is not None:
        image = cv.imread(str(args.image), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {args.image}")
        return image, str(args.image)

    cap = cv.VideoCapture(args.camera, resolve_capture_backend(args.backend))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}.")
        print("Camera preview open. Press SPACE to capture, or q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera returned no frame.")
            preview = frame.copy()
            put_lines(
                preview,
                [
                    "Letter homography capture",
                    "SPACE: capture image for Gemini",
                    "q: quit",
                ],
            )
            cv.imshow("letter homography capture", preview)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit("Cancelled before capture.")
            if key in (ord(" "), 13):
                return frame, f"camera:{args.camera}"
    finally:
        cap.release()
        cv.destroyAllWindows()


def put_lines(frame: np.ndarray, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv.putText(frame, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv.LINE_AA)
        y += 30


def prepare_image_part(
    image: np.ndarray,
    *,
    max_dim: int,
    jpeg_quality: int,
) -> tuple[types.Part, int, int, int]:
    if max_dim <= 0:
        raise ValueError("--api-max-dim must be positive.")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--api-jpeg-quality must be in [1, 100].")

    api_image = image
    height, width = image.shape[:2]
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        api_image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

    ok, encoded = cv.imencode(".jpg", api_image, [int(cv.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise ValueError("OpenCV could not encode image for Gemini.")

    image_bytes = encoded.tobytes()
    return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), width, height, len(image_bytes)


def build_prompt(letters: list[str], image_width: int, image_height: int) -> str:
    letters_text = ", ".join(letters)
    return f"""
Localize the center pixel of each requested physical keyboard keycap.

Target letters: {letters_text}
Image size: {image_width}x{image_height}

For each target letter, return the pixel coordinate of the center of that
letter's physical keycap on the keyboard. Do not return the center of the
printed glyph/ink; return the center of the whole key surface that contains
that letter. If a letter is not visible, mark it found=false.

Return strict JSON only:
{{"results": [{{"letter": "A", "found": true, "x": 0, "y": 0}}, ...]}}

Rules:
- Return exactly one result for each requested target letter.
- Use the same order as the requested target letters.
- x and y must be image pixel coordinates, not normalized coordinates.
- x must be an integer in [0,{image_width - 1}], y in [0,{image_height - 1}].
- If found=false, set x=null and y=null.
""".strip()


def build_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "letter": {"type": "string"},
                        "found": {"type": "boolean"},
                        "x": {"type": ["integer", "null"]},
                        "y": {"type": ["integer", "null"]},
                    },
                    "required": ["letter", "found", "x", "y"],
                },
            },
        },
        "required": ["results"],
    }


def call_gemini_for_letters(
    image: np.ndarray,
    *,
    letters: list[str],
    project: str | None,
    location: str,
    model: str,
    fallback_models: list[str],
    api_max_dim: int,
    api_jpeg_quality: int,
) -> dict[str, list[float]]:
    if not project:
        raise ValueError("Missing Google Cloud project. Pass --project or set GOOGLE_CLOUD_PROJECT.")

    image_height, image_width = image.shape[:2]
    image_part, api_w, api_h, api_bytes = prepare_image_part(
        image,
        max_dim=api_max_dim,
        jpeg_quality=api_jpeg_quality,
    )
    prompt = build_prompt(letters, image_width, image_height)
    client = genai.Client(vertexai=True, project=project, location=location)

    candidates = [model, *fallback_models]
    last_error: Exception | None = None
    for attempt, candidate in enumerate(candidates, start=1):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            print(
                f"Calling Gemini model {candidate} "
                f"(attempt {attempt}/{len(candidates)}, image {api_w}x{api_h}, {api_bytes / 1024:.1f} KB)..."
            )
            response = client.models.generate_content(
                model=candidate,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=build_response_schema(),
                    temperature=0,
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return parse_gemini_results(response.text, letters, image_width, image_height)
        except genai_errors.ServerError as exc:
            last_error = exc
            if getattr(exc, "status_code", None) == 503 and attempt < len(candidates):
                print("Gemini returned 503; retrying fallback model...")
                time.sleep(1.0)
                continue
            raise
        except genai_errors.APIError as exc:
            last_error = exc
            raise

    raise RuntimeError("Gemini call failed.") from last_error


def parse_gemini_results(
    response_text: str,
    requested_letters: list[str],
    image_width: int,
    image_height: int,
) -> dict[str, list[float]]:
    payload = json.loads(response_text)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Gemini response must contain a results list.")

    requested = set(requested_letters)
    by_letter: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            letter = normalize_letter(row.get("letter"))
        except ValueError:
            print(f"Warning: invalid Gemini row, skipping it: {row}")
            continue
        if letter not in requested:
            continue
        found = bool(row.get("found"))
        if not found:
            print(f"Warning: Gemini did not find letter {letter}; skipping it.")
            continue

        if row.get("x") is None or row.get("y") is None:
            print(f"Warning: Gemini marked {letter} found but returned null coordinates; skipping it.")
            continue

        u = float(row["x"])
        v = float(row["y"])
        if not (0.0 <= u <= image_width - 1 and 0.0 <= v <= image_height - 1):
            print(
                f"Warning: Gemini pixel {letter} outside image bounds "
                f"{image_width}x{image_height}: {(u, v)}; skipping it."
            )
            continue

        by_letter[letter] = [u, v]

    missing = [letter for letter in requested_letters if letter not in by_letter]
    if missing:
        print(f"Warning: missing detected pixels for letters: {missing}")
    return by_letter


def save_correspondences(
    output_path: Path,
    *,
    letters: list[str],
    world_points: dict[str, list[float]],
    pixel_points: dict[str, list[float]],
    image_source: str,
) -> list[dict[str, Any]]:
    rows = []
    for letter in letters:
        if letter not in pixel_points:
            continue
        rows.append(
            {
                "letter": letter,
                "name": letter,
                "pixel": pixel_points[letter],
                "world": world_points[letter],
                "image_source": image_source,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def draw_letters(image: np.ndarray, rows: list[dict[str, Any]], output_path: Path) -> None:
    annotated = image.copy()
    for row in rows:
        u, v = row["pixel"]
        point = (int(round(u)), int(round(v)))
        cv.circle(annotated, point, 6, (0, 0, 255), -1)
        cv.circle(annotated, point, 10, (255, 255, 255), 2)
        cv.putText(
            annotated,
            row["letter"],
            (point[0] + 8, point[1] - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), annotated)


def apply_homography(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points_xy, np.ones(len(points_xy))])
    mapped_h = points_h @ H.T
    return mapped_h[:, :2] / mapped_h[:, 2:3]


def compute_and_save_homography(
    rows: list[dict[str, Any]],
    output_dir: Path,
    threshold_px: float,
    min_points: int,
) -> None:
    if len(rows) < min_points:
        raise RuntimeError(f"Need at least {min_points} detected letter pairs, got {len(rows)}.")

    pixels = np.asarray([row["pixel"] for row in rows], dtype=np.float64)
    world_xy = np.asarray([row["world"][:2] for row in rows], dtype=np.float64)

    H_world_to_pixel, mask = cv.findHomography(
        world_xy,
        pixels,
        method=cv.RANSAC,
        ransacReprojThreshold=threshold_px,
    )
    if H_world_to_pixel is None or mask is None:
        raise RuntimeError("cv.findHomography failed. Check that points are not collinear.")

    H_pixel_to_world = np.linalg.inv(H_world_to_pixel)
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < 4:
        raise RuntimeError(f"Need at least 4 RANSAC inliers, got {int(inliers.sum())}.")

    reproj = apply_homography(H_world_to_pixel, world_xy)
    pixel_errors = np.linalg.norm(reproj - pixels, axis=1)

    back_projected = apply_homography(H_pixel_to_world, pixels)
    world_errors = np.linalg.norm(back_projected - world_xy, axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "homography_pixel_to_world.npy", H_pixel_to_world)
    np.savetxt(output_dir / "homography_pixel_to_world.txt", H_pixel_to_world, fmt="%.12g")
    np.save(output_dir / "homography_world_to_pixel.npy", H_world_to_pixel)
    np.savetxt(output_dir / "homography_world_to_pixel.txt", H_world_to_pixel, fmt="%.12g")

    print(f"RANSAC inliers: {int(inliers.sum())}/{len(inliers)}")
    print(f"Pixel reprojection RMS: {np.sqrt(np.mean(pixel_errors[inliers] ** 2)):.3f} px")
    print(f"World back-projection RMS: {np.sqrt(np.mean(world_errors[inliers] ** 2)):.6f} m")
    print("H_pixel_to_world:")
    print(H_pixel_to_world)


def main() -> None:
    args = parse_args()
    world_points = load_world_points(args.world_points)
    letters = parse_letters_arg(args.letters, world_points)
    image, image_source = load_or_capture_image(args)
    fallback_models = [model.strip() for model in args.fallback_models.split(",") if model.strip()]

    pixel_points = call_gemini_for_letters(
        image,
        letters=letters,
        project=args.project,
        location=args.location,
        model=args.model,
        fallback_models=fallback_models,
        api_max_dim=args.api_max_dim,
        api_jpeg_quality=args.api_jpeg_quality,
    )
    rows = save_correspondences(
        args.output,
        letters=letters,
        world_points=world_points,
        pixel_points=pixel_points,
        image_source=image_source,
    )
    if len(rows) < args.min_points:
        raise RuntimeError(f"Need at least {args.min_points} detected letter pairs, got {len(rows)}.")

    draw_letters(image, rows, args.annotated_output)

    print(f"Requested letters: {letters}")
    print(f"Detected letter pairs: {len(rows)}")
    print(f"Saved correspondences: {args.output}")
    print(f"Saved annotated image: {args.annotated_output}")
    for row in rows:
        print(f"{row['letter']}: pixel={np.round(row['pixel'], 2).tolist()} world={row['world']}")

    if not args.skip_homography:
        compute_and_save_homography(rows, args.homography_dir, args.ransac_threshold_px, args.min_points)


if __name__ == "__main__":
    main()
