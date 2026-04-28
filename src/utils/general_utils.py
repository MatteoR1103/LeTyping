import os
import cv2 as cv
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WINDOW_NAME = "track to world"


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


def capture_initial_frame_with_preview(cap: cv.VideoCapture, letter: str) -> np.ndarray | None:
    """
    Waits for the user to press SPACE to run Gemini VLM key localization
    """
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



