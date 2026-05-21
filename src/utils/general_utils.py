import argparse
import os
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WINDOW_NAME = "track to world"


def load_pipeline_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Main pipeline config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Main pipeline config must contain a YAML mapping: {config_path}")
    return config


def config_value(config: dict, dotted_key: str, fallback=None):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return fallback
        value = value[key]
    return fallback if value is None else value


def resolve_urdf_path(path: str) -> str:
    """
    Resolves paths to the URDF file that is needed for FK computation
    """
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(REPO_ROOT, path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "SO101 URDF not found. Copy `so101_new_calib.urdf` from the SO-ARM100 repo into "
        f"`{os.path.join(REPO_ROOT, 'SO101')}` or pass `--urdf-path /absolute/path/to/so101_new_calib.urdf`."
    )


def resolve_capture_backend(backend_name: str) -> int:
    """
    Sets the camera backend for cv
    """
    normalized = backend_name.strip().lower()
    if normalized in {"auto", "any"}:
        return cv.CAP_ANY
    if normalized == "dshow":
        return cv.CAP_DSHOW
    if normalized == "msmf":
        return cv.CAP_MSMF
    raise ValueError(f"Unsupported camera backend: {backend_name}")


def parse_typing_targets(word_args: list[str]) -> list[str]:
    normalized_tokens = [token.strip().upper() for token in word_args if token.strip()]
    if normalized_tokens and all(
        token in {"SPACE", "ENTER"} or (len(token) == 1 and token.isalpha())
        for token in normalized_tokens
    ):
        return normalized_tokens

    text = " ".join(word_args)
    targets: list[str] = []
    for char in text:
        if char.isspace():
            targets.append("SPACE")
        elif char.isalpha():
            targets.append(char.upper())
    return targets


def read_sentence_list(list_path: Path) -> list[str]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Sentence list file not found: {list_path}")

    sentences = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sentences:
        raise ValueError(f"Sentence list file is empty: {list_path}")
    return sentences


def build_typing_runs(
    args: argparse.Namespace,
    *,
    task1_targets: list[str],
) -> list[tuple[str, list[str]]]:
    if getattr(args, "task_1", False) or getattr(args, "task", None) == 1:
        return [("Task 1", task1_targets.copy())]

    if args.word is not None:
        text = " ".join(args.word)
        return [(text, parse_typing_targets(args.word))]

    if args.list_path is not None:
        return [
            (sentence, parse_typing_targets(sentence.split()))
            for sentence in read_sentence_list(args.list_path)
        ]

    raise ValueError("Pass --word, --task-1, --task 1, or --list-path.")


def read_frame(cap: cv.VideoCapture, *, error_message: str) -> np.ndarray:
    """
    Reads an image out of a frame of cv2
    """
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
    """
    Writes onto the cv2 video capture frame
    """
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


def capture_initial_frame_with_preview(
    cap: cv.VideoCapture,
    letter: str,
) -> tuple[np.ndarray | None, float | None]:
    """
    Waits for the user to press ENTER or SPACE to run key localization.
    """
    print("Camera preview is open. Press ENTER to localize on the current frame, or q to quit.")

    while True:
        frame = read_frame(cap, error_message="Camera stream ended during preview.")
        preview = frame.copy()
        put_status_lines(
            preview,
            [
                "Live camera preview",
                f"Target letter: {letter}",
                "ENTER: localize",
                "q: quit",
            ],
        )
        cv.imshow(WINDOW_NAME, preview)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            return None, None
        if key in (ord(" "), 13):
            return frame, time.perf_counter()


def show_gemini_busy_frame(frame: np.ndarray, letter: str) -> None:
    """
    Shows that Gemini VLM is looking for the key
    """
    busy_frame = frame.copy()
    put_status_lines(
        busy_frame,
        [
            f"Calling Gemini for letter {letter}...",
            "Please wait. Tracking will start after initialization.",
        ],
    )
    cv.imshow(WINDOW_NAME, busy_frame)
    cv.waitKey(1)
