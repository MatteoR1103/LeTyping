from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np

from gemini_keyboard_localizer import (
    REFERENCE_HEIGHT,
    REFERENCE_WIDTH,
    GeminiLocalizationResult,
    call_gemini,
    classical_validation,
    parse_fallback_models,
    parse_gemini_response,
    parse_target_letters,
)
from triangulation import CAMERA_NO, K, convert_to_ray, find_intersection, trackForward


WINDOW_NAME = "live keyboard tracker"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_LIVE_MODEL = "gemini-3-flash-preview"
DEFAULT_CAMERA_BACKEND = "dshow" if os.name == "nt" else "auto"
DEFAULT_T_WC = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 21.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
DEFAULT_PLANE_NORMAL = np.array([0.0, 0.0, 1.0])
DEFAULT_PLANE_POINT = np.array([0.0, 0.0, 0.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemini once to localize a keyboard key, then track it live and "
            "estimate its 3D position on the keyboard plane."
        )
    )
    parser.add_argument(
        "--letter",
        help="Single target keyboard letter to localize and track, for example X.",
    )
    parser.add_argument(
        "--camera",
        "--camera-index",
        dest="camera_index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=(
            f"OpenCV camera index. Default: {DEFAULT_CAMERA_INDEX} "
            f"(integrated webcam on most laptops). triangulation.py uses {CAMERA_NO}."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe common OpenCV camera indices and exit.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default=DEFAULT_CAMERA_BACKEND,
        help=(
            "OpenCV camera backend. On Windows, dshow is usually best for USB cameras. "
            f"Default: {DEFAULT_CAMERA_BACKEND}"
        ),
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=REFERENCE_WIDTH,
        help=(
            "Requested capture width. Defaults to the calibration/reference "
            f"width used by the current K matrix: {REFERENCE_WIDTH}."
        ),
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=REFERENCE_HEIGHT,
        help=(
            "Requested capture height. Defaults to the calibration/reference "
            f"height used by the current K matrix: {REFERENCE_HEIGHT}."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help=f"Gemini model used for the initial localization. Default: {DEFAULT_LIVE_MODEL}",
    )
    parser.add_argument(
        "--fallback-models",
        default="",
        help="Optional comma-separated fallback Gemini models. Default: none.",
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
        "--save-initial-frame",
        nargs="?",
        const="camera/live_initial_frame.jpg",
        help=(
            "Optional path for saving the Gemini-initialized frame overlay. "
            "If no path is provided, saves to camera/live_initial_frame.jpg."
        ),
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Skip the live preview and call Gemini immediately on the first camera frame.",
    )
    parser.add_argument(
        "--rerun-on-loss",
        action="store_true",
        help="If tracking is lost, wait for r to re-run Gemini on the current frame.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame tracked pixel and 3D point information.",
    )
    return parser.parse_args()


def parse_single_letter(letter_arg: str) -> str:
    if not letter_arg:
        raise ValueError("Missing required --letter, for example --letter X.")

    target_letters = parse_target_letters(letter_arg)
    if len(target_letters) != 1:
        raise ValueError("live tracking expects exactly one target letter, for example --letter X.")
    return target_letters[0]


def resolve_capture_backend(backend_name: str) -> int:
    normalized = backend_name.strip().lower()
    if normalized in {"auto", "any"}:
        return cv.CAP_ANY
    if normalized == "dshow":
        return cv.CAP_DSHOW
    if normalized == "msmf":
        return cv.CAP_MSMF
    raise ValueError(f"Unsupported camera backend: {backend_name}")


def open_camera(camera_index: int, frame_width: int, frame_height: int, backend_name: str) -> cv.VideoCapture:
    backend = resolve_capture_backend(backend_name)
    cap = cv.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index} with backend `{backend_name}`.")

    if frame_width > 0:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    if frame_height > 0:
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    return cap


def list_cameras(max_index: int = 10, backend_name: str = DEFAULT_CAMERA_BACKEND) -> None:
    backend = resolve_capture_backend(backend_name)
    print(f"Probing camera indices 0..{max_index - 1} with backend `{backend_name}`...")
    found_any = False
    for camera_index in range(max_index):
        cap = cv.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            print(f"camera {camera_index}: available ({width}x{height})")
            found_any = True
        else:
            print(f"camera {camera_index}: opened but no frame returned")
            found_any = True
        cap.release()

    if not found_any:
        print("No cameras were found by OpenCV in the probed index range.")


def read_frame(cap: cv.VideoCapture, *, error_message: str) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(error_message)
    return frame


def put_status_lines(
    frame: np.ndarray,
    lines: list[str],
    *,
    color: tuple[int, int, int] = (0, 220, 255),
) -> None:
    y = 30
    for line in lines:
        cv.putText(
            frame,
            line,
            (12, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv.LINE_AA,
        )
        y += 30


def capture_initial_frame_with_preview(cap: cv.VideoCapture, letter: str) -> np.ndarray | None:
    print("Camera preview is open. Press SPACE to run Gemini on the current frame, or q to quit.")

    while True:
        frame = read_frame(cap, error_message="Camera stream ended during preview.")
        preview = frame.copy()
        put_status_lines(
            preview,
            [
                "Live camera preview",
                f"Target letter: {letter}",
                "SPACE: localize with Gemini",
                "q: quit",
            ],
        )
        cv.imshow(WINDOW_NAME, preview)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key in (ord(" "), 13):
            return frame


def show_gemini_busy_frame(frame: np.ndarray, letter: str) -> None:
    busy_frame = frame.copy()
    put_status_lines(
        busy_frame,
        [
            f"Calling Gemini for letter {letter}...",
            "Please wait. The video will continue after initialization.",
        ],
        color=(0, 220, 255),
    )
    cv.imshow(WINDOW_NAME, busy_frame)
    cv.waitKey(1)


def warn_if_intrinsics_mismatch(frame: np.ndarray) -> None:
    frame_height, frame_width = frame.shape[:2]
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if not (0 <= cx < frame_width and 0 <= cy < frame_height):
        print(
            "Warning: camera intrinsics K look incompatible with this frame size. "
            f"K principal point is ({cx:.1f}, {cy:.1f}), frame is {frame_width}x{frame_height}. "
            "2D tracking still works, but the 3D estimate may be wrong."
        )
        return

    expected_width = max(1.0, 2.0 * cx)
    expected_height = max(1.0, 2.0 * cy)
    width_error = abs(frame_width - expected_width) / expected_width
    height_error = abs(frame_height - expected_height) / expected_height
    if width_error > 0.15 or height_error > 0.15:
        print(
            "Warning: frame size differs from the apparent K calibration size. "
            f"Frame is {frame_width}x{frame_height}; K suggests roughly "
            f"{expected_width:.0f}x{expected_height:.0f}. 3D estimates may be inaccurate."
        )


def localize_with_gemini(
    frame: np.ndarray,
    *,
    letter: str,
    model: str,
    fallback_models: list[str],
    project: str | None,
    location: str,
) -> GeminiLocalizationResult:
    image_height, image_width = frame.shape[:2]
    gemini_call = call_gemini(
        image=frame,
        image_width=image_width,
        image_height=image_height,
        target_letters=[letter],
        model=model,
        fallback_models=fallback_models,
        project=project,
        location=location,
    )
    print(f"Gemini response received from model: {gemini_call.model_used}")
    print(
        "Gemini API image: "
        f"{gemini_call.api_image_width}x{gemini_call.api_image_height}, "
        f"{gemini_call.api_image_bytes / 1024.0:.1f} KB"
    )
    print(f"Gemini request time: {gemini_call.request_elapsed_seconds:.2f} seconds")

    result = parse_gemini_response(
        gemini_call.response_text,
        image_width=image_width,
        image_height=image_height,
        expected_letters=[letter],
    )[0]

    if not result.found or result.center is None:
        raise RuntimeError(f"Gemini did not find the target letter `{letter}`.")

    validation = classical_validation(frame, result)
    print(
        f"Initial localization: letter={result.target_letter}, "
        f"center=({result.center['x']}, {result.center['y']}), "
        f"cv_check={'PASS' if validation.passed else 'FAIL'}"
    )
    if not validation.passed:
        print("Warning: initial OpenCV heuristic validation failed; tracking will still start.")

    return result


def point_from_result(result: GeminiLocalizationResult) -> np.ndarray:
    if result.center is None:
        raise ValueError("Cannot initialize tracking without a Gemini center point.")
    return np.array([result.center["x"], result.center["y"]], dtype=np.float32)


def estimate_3d_point(pixel: np.ndarray) -> tuple[np.ndarray | None, str]:
    ray_o, ray_d = convert_to_ray(pixel, T_WC=DEFAULT_T_WC)
    point_3d, _, status = find_intersection(
        plane_n=DEFAULT_PLANE_NORMAL,
        plane_p0=DEFAULT_PLANE_POINT,
        ray_o=ray_o,
        ray_d=ray_d,
    )
    return point_3d, status


def draw_initial_bbox(frame: np.ndarray, result: GeminiLocalizationResult) -> None:
    if result.bounding_box is None:
        return
    xmin, ymin, xmax, ymax = result.bounding_box
    cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 255), 1)
    cv.putText(
        frame,
        "initial Gemini bbox",
        (xmin, max(18, ymin - 8)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 255),
        1,
        cv.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    *,
    letter: str,
    pixel: np.ndarray | None,
    point_3d: np.ndarray | None,
    tracking_status: str,
    intersection_status: str,
    initial_result: GeminiLocalizationResult | None,
) -> None:
    color = (0, 220, 0) if tracking_status == "tracking" else (0, 0, 255)

    if initial_result is not None:
        draw_initial_bbox(frame, initial_result)

    if pixel is not None:
        pixel_int = tuple(np.round(pixel).astype(int))
        cv.circle(frame, pixel_int, 6, color, -1)
        cv.circle(frame, pixel_int, 12, color, 2)

    lines = [
        f"Letter: {letter}",
        f"Tracking: {tracking_status}",
    ]
    if pixel is not None:
        lines.append(f"Pixel: ({pixel[0]:.1f}, {pixel[1]:.1f})")
    if point_3d is not None:
        lines.append(f"3D: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})")
    else:
        lines.append(f"3D: unavailable ({intersection_status})")
    lines.append("Keys: q quit, r re-run Gemini")

    y = 26
    for line in lines:
        cv.putText(
            frame,
            line,
            (12, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv.LINE_AA,
        )
        y += 28


def save_initial_frame(path_arg: str, frame: np.ndarray, result: GeminiLocalizationResult) -> None:
    output_path = Path(path_arg).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = frame.copy()
    draw_initial_bbox(annotated, result)
    if result.center is not None:
        center = (result.center["x"], result.center["y"])
        cv.circle(annotated, center, 6, (0, 220, 255), -1)
    if not cv.imwrite(str(output_path), annotated):
        raise IOError(f"Failed to save initial frame to {output_path}")
    print(f"Initial frame saved to: {output_path}")


def run_gemini_initialization(
    frame: np.ndarray,
    *,
    args: argparse.Namespace,
    letter: str,
    fallback_models: list[str],
) -> GeminiLocalizationResult:
    print(f"Localizing `{letter}` with Gemini on the current frame...")
    return localize_with_gemini(
        frame,
        letter=letter,
        model=args.model,
        fallback_models=fallback_models,
        project=args.project,
        location=args.location,
    )


def main() -> None:
    cap: cv.VideoCapture | None = None
    try:
        args = parse_args()
        if args.list_cameras:
            list_cameras(backend_name=args.backend)
            return

        letter = parse_single_letter(args.letter)
        fallback_models = parse_fallback_models(args.fallback_models)

        cap = open_camera(
            camera_index=args.camera_index,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            backend_name=args.backend,
        )

        if args.auto_start:
            initial_frame = read_frame(
                cap,
                error_message="Camera opened, but no initial frame was available.",
            )
        else:
            initial_frame = capture_initial_frame_with_preview(cap, letter)
            if initial_frame is None:
                return

        warn_if_intrinsics_mismatch(initial_frame)

        show_gemini_busy_frame(initial_frame, letter)
        initial_result = run_gemini_initialization(
            initial_frame,
            args=args,
            letter=letter,
            fallback_models=fallback_models,
        )
        if args.save_initial_frame:
            save_initial_frame(args.save_initial_frame, initial_frame, initial_result)

        current_pixel = point_from_result(initial_result)
        last_gray = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)
        print(
            f"Tracking initialized at pixel ({current_pixel[0]:.1f}, {current_pixel[1]:.1f}). "
            "Press q to quit, r to re-run Gemini on the current frame."
        )

        while True:
            frame = read_frame(cap, error_message="Camera stream ended or returned no frame.")
            raw_frame = frame.copy()
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            new_points, status = trackForward(
                pixel_coord=current_pixel,
                prevImg=last_gray,
                nextImg=gray_frame,
            )
            status_flat = np.asarray(status).reshape(-1)
            tracking_ok = bool(status_flat.size and status_flat[0] == 1)

            point_3d = None
            intersection_status = "not computed"
            tracking_status = "tracking"
            if tracking_ok:
                current_pixel = np.asarray(new_points, dtype=np.float32).reshape(-1, 2)[0]
                point_3d, intersection_status = estimate_3d_point(current_pixel)
                if args.verbose:
                    if point_3d is None:
                        print(
                            f"pixel=({current_pixel[0]:.2f}, {current_pixel[1]:.2f}), "
                            f"3d unavailable: {intersection_status}"
                        )
                    else:
                        print(
                            f"pixel=({current_pixel[0]:.2f}, {current_pixel[1]:.2f}), "
                            f"3d=({point_3d[0]:.4f}, {point_3d[1]:.4f}, {point_3d[2]:.4f})"
                        )
            else:
                tracking_status = "lost"
                intersection_status = "tracking lost"
                print("Tracking lost: KLT could not track the keypoint into the current frame.")

            draw_overlay(
                frame,
                letter=letter,
                pixel=current_pixel if tracking_ok else None,
                point_3d=point_3d,
                tracking_status=tracking_status,
                intersection_status=intersection_status,
                initial_result=initial_result,
            )
            cv.imshow(WINDOW_NAME, frame)

            if tracking_ok:
                last_gray = gray_frame

            wait_delay = 1 if tracking_ok or not args.rerun_on_loss else 0
            key = cv.waitKey(wait_delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                show_gemini_busy_frame(raw_frame, letter)
                initial_result = run_gemini_initialization(
                    raw_frame,
                    args=args,
                    letter=letter,
                    fallback_models=fallback_models,
                )
                current_pixel = point_from_result(initial_result)
                last_gray = gray_frame
                print(
                    f"Tracking re-initialized at pixel "
                    f"({current_pixel[0]:.1f}, {current_pixel[1]:.1f})."
                )
                continue
            if not tracking_ok and not args.rerun_on_loss:
                break

    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
