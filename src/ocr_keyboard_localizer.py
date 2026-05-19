from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .gemini_keyboard_localizer import GeminiLocalizationResult
except ImportError:
    from gemini_keyboard_localizer import GeminiLocalizationResult


EASYOCR_DEBUG_DIR = Path("camera")
EASYOCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
EASYOCR_KEYBOARD_ROWS = ("QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM")
EASYOCR_LAYOUT_BY_LETTER = {
    letter: (float(column) + 0.5 * row_index, float(row_index))
    for row_index, row_letters in enumerate(EASYOCR_KEYBOARD_ROWS)
    for column, letter in enumerate(row_letters)
}
EASYOCR_LAYOUT_BY_SYMBOL_KEY = {
    "LEFT_BRACKET": (10.0, 0.0),
    "RIGHT_BRACKET": (11.0, 0.0),
    "SEMICOLON": (9.5, 1.0),
    "QUOTE": (10.5, 1.0),
}
EASYOCR_LAYOUT_BY_KEY = {
    **EASYOCR_LAYOUT_BY_LETTER,
    **EASYOCR_LAYOUT_BY_SYMBOL_KEY,
    "SPACE": (4.5, 3.25),
    "ENTER": (12.0, 0.0),
}
EASYOCR_ANCHOR_MIN_PROBABILITY = 0.75
EASYOCR_SPECIAL_TEXT_MIN_PROBABILITY = 0.5
EASYOCR_MAX_REPROJECTION_ERROR_PX = 18.0


@dataclass
class EasyOcrCandidate:
    text: str
    normalized_text: str
    probability: float
    bounding_box: list[int]
    center: dict[str, int]
    variant: str


_easyocr_reader = None

def get_easyocr_reader():
    """Loads and returns a singleton EasyOCR reader instance, initializing it if necessary."""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("Initializing EasyOCR model...")
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _easyocr_reader


def _easyocr_preprocess_variants(image: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    sharpened = cv2.filter2D(
        clahe,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )

    variants: list[tuple[str, np.ndarray, float]] = [("raw", image, 1.0)]
    for name, processed_gray in (
        ("clahe_up2", clahe),
        ("sharpened_up2", sharpened),
    ):
        upscaled = cv2.resize(
            processed_gray,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        variants.append((name, cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR), 2.0))
    return variants


def _normalize_easyocr_text(text: str) -> str:
    return "".join(character for character in text.upper() if character.isalnum())


def _scale_easyocr_bbox(bbox: list, scale: float) -> list[int]:
    xs = [float(point[0]) / scale for point in bbox]
    ys = [float(point[1]) / scale for point in bbox]
    return [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    ]


def _easyocr_candidates(
    reader,
    image: np.ndarray,
) -> list[EasyOcrCandidate]:
    candidates: list[EasyOcrCandidate] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()

    for variant_name, variant_image, scale in _easyocr_preprocess_variants(image):
        rgb_image = cv2.cvtColor(variant_image, cv2.COLOR_BGR2RGB)
        results = reader.readtext(
            rgb_image,
            allowlist=EASYOCR_ALLOWLIST,
            paragraph=False,
            min_size=8,
            text_threshold=0.4,
            low_text=0.2,
            link_threshold=0.2,
            add_margin=0.15,
            mag_ratio=1.2,
        )

        for bbox, text, probability in results:
            normalized_text = _normalize_easyocr_text(text)
            if not normalized_text:
                continue

            scaled_bbox = _scale_easyocr_bbox(bbox, scale)
            xmin, ymin, xmax, ymax = scaled_bbox
            dedupe_key = (
                normalized_text,
                (
                    round(xmin / 4),
                    round(ymin / 4),
                    round(xmax / 4),
                    round(ymax / 4),
                ),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            candidates.append(
                EasyOcrCandidate(
                    text=str(text),
                    normalized_text=normalized_text,
                    probability=float(probability),
                    bounding_box=scaled_bbox,
                    center={
                        "x": int(round((xmin + xmax) / 2)),
                        "y": int(round((ymin + ymax) / 2)),
                    },
                    variant=variant_name,
                )
            )

    return candidates


def _best_easyocr_anchor_by_letter(
    candidates: list[EasyOcrCandidate],
) -> dict[str, EasyOcrCandidate]:
    anchors: dict[str, EasyOcrCandidate] = {}
    for candidate in candidates:
        letter = candidate.normalized_text
        if (
            len(letter) != 1
            or letter not in EASYOCR_LAYOUT_BY_LETTER
            or candidate.probability < EASYOCR_ANCHOR_MIN_PROBABILITY
        ):
            continue

        current = anchors.get(letter)
        if current is None or candidate.probability > current.probability:
            anchors[letter] = candidate

    return anchors


def _best_easyocr_text_candidate(
    candidates: list[EasyOcrCandidate],
    text: str,
) -> EasyOcrCandidate | None:
    expected = text.upper()
    matches = [
        candidate
        for candidate in candidates
        if (
            candidate.normalized_text == expected
            and candidate.probability >= EASYOCR_SPECIAL_TEXT_MIN_PROBABILITY
        )
    ]
    if not matches:
        return None
    return max(matches, key=lambda candidate: candidate.probability)


def _candidate_center_array(candidate: EasyOcrCandidate) -> np.ndarray:
    return np.array([candidate.center["x"], candidate.center["y"]], dtype=np.float32)


def _fit_easyocr_keyboard_map(
    anchors_by_letter: dict[str, EasyOcrCandidate],
) -> dict[str, Any] | None:
    if len(anchors_by_letter) < 3:
        return None

    anchor_rows = {
        EASYOCR_LAYOUT_BY_LETTER[letter][1]
        for letter in anchors_by_letter
        if letter in EASYOCR_LAYOUT_BY_LETTER
    }
    if len(anchor_rows) < 2:
        return None

    source_points = []
    destination_points = []
    anchor_letters = []
    for letter, candidate in sorted(anchors_by_letter.items()):
        source_points.append(EASYOCR_LAYOUT_BY_LETTER[letter])
        destination_points.append(_candidate_center_array(candidate))
        anchor_letters.append(letter)

    src = np.asarray(source_points, dtype=np.float32)
    dst = np.asarray(destination_points, dtype=np.float32)

    transform_type = "affine"
    transform = None
    inlier_mask = None
    if len(anchors_by_letter) >= 4 and len({point[1] for point in source_points}) >= 2:
        transform, inlier_mask = cv2.findHomography(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=EASYOCR_MAX_REPROJECTION_ERROR_PX,
        )
        transform_type = "homography"

    if transform is None:
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=EASYOCR_MAX_REPROJECTION_ERROR_PX,
        )
        transform_type = "affine"

    if transform is None:
        return None

    projected = _project_easyocr_layout_points(src, transform, transform_type)
    errors = np.linalg.norm(projected - dst, axis=1)
    if inlier_mask is None:
        inliers = np.ones(len(anchor_letters), dtype=bool)
    else:
        inliers = np.asarray(inlier_mask, dtype=bool).reshape(-1)

    min_inliers = 4 if transform_type == "homography" else 3
    if int(np.count_nonzero(inliers)) < min_inliers:
        return None

    inlier_errors = errors[inliers]
    if (
        len(inlier_errors) == 0
        or float(np.median(inlier_errors)) > EASYOCR_MAX_REPROJECTION_ERROR_PX
    ):
        return None

    return {
        "type": transform_type,
        "transform": transform,
        "anchor_letters": anchor_letters,
        "inliers": inliers,
        "median_error_px": float(np.median(inlier_errors)),
        "max_error_px": float(np.max(inlier_errors)),
    }


def _project_easyocr_layout_points(
    points: np.ndarray,
    transform: np.ndarray,
    transform_type: str,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if transform_type == "homography":
        homogeneous = cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform)
        return homogeneous.reshape(-1, 2)

    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
    return (homogeneous @ transform.T).astype(np.float32)


def _predict_easyocr_key_center(
    keyboard_map: dict[str, Any],
    key: str,
) -> dict[str, int] | None:
    layout_point = EASYOCR_LAYOUT_BY_KEY.get(key)
    if layout_point is None:
        return None

    projected = _project_easyocr_layout_points(
        np.asarray([layout_point], dtype=np.float32),
        keyboard_map["transform"],
        keyboard_map["type"],
    )[0]
    return {"x": int(round(projected[0])), "y": int(round(projected[1]))}


def _easyocr_key_box_size(
    anchors_by_letter: dict[str, EasyOcrCandidate],
    keyboard_map: dict[str, Any],
) -> tuple[int, int]:
    widths = [
        candidate.bounding_box[2] - candidate.bounding_box[0]
        for candidate in anchors_by_letter.values()
    ]
    heights = [
        candidate.bounding_box[3] - candidate.bounding_box[1]
        for candidate in anchors_by_letter.values()
    ]
    if widths and heights:
        return (
            max(8, int(round(float(np.median(widths))))),
            max(8, int(round(float(np.median(heights))))),
        )

    q_center = _predict_easyocr_key_center(keyboard_map, "Q")
    w_center = _predict_easyocr_key_center(keyboard_map, "W")
    q_layout = EASYOCR_LAYOUT_BY_LETTER["Q"]
    a_layout = EASYOCR_LAYOUT_BY_LETTER["A"]
    q_px, a_px = _project_easyocr_layout_points(
        np.asarray([q_layout, a_layout], dtype=np.float32),
        keyboard_map["transform"],
        keyboard_map["type"],
    )
    row_pitch = 24.0 if q_center is None or w_center is None else np.linalg.norm(
        np.array([w_center["x"] - q_center["x"], w_center["y"] - q_center["y"]])
    )
    col_pitch = np.linalg.norm(a_px - q_px)
    return (
        max(8, int(round(row_pitch * 0.65))),
        max(8, int(round(col_pitch * 0.65))),
    )


def _centered_bbox(
    center: dict[str, int],
    width: int,
    height: int,
    image_shape: tuple[int, ...],
) -> list[int]:
    image_height, image_width = image_shape[:2]
    half_width = max(1, width // 2)
    half_height = max(1, height // 2)
    xmin = max(0, center["x"] - half_width)
    ymin = max(0, center["y"] - half_height)
    xmax = min(image_width - 1, center["x"] + half_width)
    ymax = min(image_height - 1, center["y"] + half_height)
    return [xmin, ymin, xmax, ymax]


def _easyocr_predicted_bbox(
    key: str,
    center: dict[str, int],
    key_width: int,
    key_height: int,
    image_shape: tuple[int, ...],
) -> list[int]:
    if key == "SPACE":
        width = max(key_width, int(round(key_width * 5.0)))
        height = max(key_height, int(round(key_height * 1.2)))
    elif key == "ENTER":
        width = max(key_width, int(round(key_width * 1.6)))
        height = max(key_height, int(round(key_height * 1.8)))
    else:
        width = key_width
        height = key_height

    return _centered_bbox(center, width, height, image_shape)


def _save_easyocr_debug_overlay(
    image: np.ndarray,
    candidates: list[EasyOcrCandidate],
    localized_results: list[GeminiLocalizationResult],
    anchors_by_letter: dict[str, EasyOcrCandidate] | None = None,
) -> Path:
    annotated = image.copy()
    for candidate in candidates:
        xmin, ymin, xmax, ymax = candidate.bounding_box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (255, 180, 0), 1)
        cv2.putText(
            annotated,
            f"{candidate.normalized_text}:{candidate.probability:.2f}",
            (xmin, max(14, ymin - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 180, 0),
            1,
            cv2.LINE_AA,
        )

    for letter, candidate in (anchors_by_letter or {}).items():
        xmin, ymin, xmax, ymax = candidate.bounding_box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (220, 0, 220), 2)
        cv2.putText(
            annotated,
            f"anchor:{letter}",
            (xmin, min(image.shape[0] - 6, ymax + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 0, 220),
            1,
            cv2.LINE_AA,
        )

    for result in localized_results:
        if not result.found or result.bounding_box is None or result.center is None:
            continue

        xmin, ymin, xmax, ymax = result.bounding_box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 200, 0), 2)
        cv2.circle(
            annotated,
            (result.center["x"], result.center["y"]),
            4,
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            annotated,
            result.target_letter,
            (xmin, min(image.shape[0] - 6, ymax + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    output_path = (
        EASYOCR_DEBUG_DIR
        / f"easyocr_detections_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return output_path


def localize_multiple_with_easyocr(
    image: np.ndarray,
    target_letters: list[str],
) -> list[GeminiLocalizationResult]:
    """Cerca le lettere in locale usando EasyOCR."""
    reader = get_easyocr_reader()
    candidates = _easyocr_candidates(reader, image)
    anchors_by_letter = _best_easyocr_anchor_by_letter(candidates)
    keyboard_map = _fit_easyocr_keyboard_map(anchors_by_letter)

    print(f"EasyOCR ha trovato {len(candidates)} candidati testuali.")
    for candidate in candidates:
        print(
            "  "
            f"text='{candidate.text}' norm='{candidate.normalized_text}' "
            f"prob={candidate.probability:.2f} bbox={candidate.bounding_box} "
            f"variant={candidate.variant}"
        )

    print(
        "EasyOCR keyboard anchors: "
        f"{', '.join(sorted(anchors_by_letter)) if anchors_by_letter else 'none'}"
    )
    if keyboard_map is None:
        print(
            "EasyOCR keyboard map unavailable; "
            "falling back to high-confidence exact OCR anchors."
        )
    else:
        inlier_letters = [
            letter
            for letter, is_inlier in zip(
                keyboard_map["anchor_letters"],
                keyboard_map["inliers"],
            )
            if is_inlier
        ]
        print(
            "EasyOCR keyboard map fitted: "
            f"type={keyboard_map['type']}, "
            f"inliers={','.join(inlier_letters)}, "
            f"median_error={keyboard_map['median_error_px']:.1f}px, "
            f"max_error={keyboard_map['max_error_px']:.1f}px"
        )

    key_width, key_height = (0, 0)
    if keyboard_map is not None:
        key_width, key_height = _easyocr_key_box_size(anchors_by_letter, keyboard_map)

    found_results = []
    for target_letter in target_letters:
        target = target_letter.upper()
        letter_result = GeminiLocalizationResult(
            target_letter=target_letter,
            found=False,
            center=None,
            bounding_box=None,
            raw_response={"provider": "easyocr", "candidates": []},
        )

        enter_candidate = None
        if target == "ENTER":
            enter_candidate = _best_easyocr_text_candidate(candidates, "ENTER")

        if enter_candidate is not None:
            print(
                "EasyOCR text prediction for ENTER: "
                f"prob={enter_candidate.probability:.2f}, "
                f"bbox={enter_candidate.bounding_box}"
            )
            letter_result = GeminiLocalizationResult(
                target_letter=target_letter,
                found=True,
                center=enter_candidate.center,
                bounding_box=enter_candidate.bounding_box,
                raw_response={
                    "provider": "easyocr",
                    "source": "special_text",
                    "text": enter_candidate.text,
                    "normalized_text": enter_candidate.normalized_text,
                    "probability": enter_candidate.probability,
                    "variant": enter_candidate.variant,
                },
            )
        elif keyboard_map is not None and target in EASYOCR_LAYOUT_BY_KEY:
            center = _predict_easyocr_key_center(keyboard_map, target)
            if center is not None:
                bounding_box = _easyocr_predicted_bbox(
                    target,
                    center,
                    key_width,
                    key_height,
                    image.shape,
                )
                print(
                    f"EasyOCR map prediction for {target_letter}: "
                    f"center=({center['x']}, {center['y']}), bbox={bounding_box}"
                )
                letter_result = GeminiLocalizationResult(
                    target_letter=target_letter,
                    found=True,
                    center=center,
                    bounding_box=bounding_box,
                    raw_response={
                        "provider": "easyocr",
                        "source": "keyboard_map",
                        "map_type": keyboard_map["type"],
                        "median_error_px": keyboard_map["median_error_px"],
                        "max_error_px": keyboard_map["max_error_px"],
                    },
                )
        elif target in anchors_by_letter:
            candidate = anchors_by_letter[target]
            print(
                f"EasyOCR exact-anchor fallback for {target_letter}: "
                f"prob={candidate.probability:.2f}, bbox={candidate.bounding_box}"
            )
            letter_result = GeminiLocalizationResult(
                target_letter=target_letter,
                found=True,
                center=candidate.center,
                bounding_box=candidate.bounding_box,
                raw_response={
                    "provider": "easyocr",
                    "source": "exact_anchor_fallback",
                    "text": candidate.text,
                    "normalized_text": candidate.normalized_text,
                    "probability": candidate.probability,
                    "variant": candidate.variant,
                },
            )

        found_results.append(letter_result)

    output_path = _save_easyocr_debug_overlay(
        image,
        candidates,
        found_results,
        anchors_by_letter=anchors_by_letter,
    )
    print(f"Saved EasyOCR debug overlay: {output_path}")
    return found_results
