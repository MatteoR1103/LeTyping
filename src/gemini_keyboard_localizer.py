from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
import easyocr
import numpy as np

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing computer-vision dependencies. Install them with `pip install -r requirements.txt`."
    ) from exc

try:
    from openai import APIError, OpenAI
except ImportError as exc:
    raise SystemExit(
        "Missing OpenAI SDK. Install it with `pip install -r requirements.txt`."
    ) from exc

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError:
    genai = None
    genai_errors = None
    types = None


API_IMAGE_MAX_DIM = 1920
API_IMAGE_JPEG_QUALITY = 100
GEMINI_BACKENDS = {"standard", "priority", "provisioned"}
LOCALIZATION_PROVIDERS = {"openai", "gemini"}
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite"
THINKING_BUDGET = 0
TEXT_ONLY_IMAGE_OUTPUT_MODELS = {
    "gemini-2.5-flash-image",
}
EASYOCR_DEBUG_DIR = Path("camera")
EASYOCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
EASYOCR_KEYBOARD_ROWS = ("QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM")
EASYOCR_LAYOUT_BY_LETTER = {
    letter: (float(column) + 0.5 * row_index, float(row_index))
    for row_index, row_letters in enumerate(EASYOCR_KEYBOARD_ROWS)
    for column, letter in enumerate(row_letters)
}
EASYOCR_LAYOUT_BY_KEY = {
    **EASYOCR_LAYOUT_BY_LETTER,
    "SPACE": (4.5, 3.25),
    "ENTER": (10.0, 1.0),
}
EASYOCR_ANCHOR_MIN_PROBABILITY = 0.75
EASYOCR_SPECIAL_TEXT_MIN_PROBABILITY = 0.5
EASYOCR_MAX_REPROJECTION_ERROR_PX = 18.0


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
    provider_used: str
    elapsed_seconds: float
    request_elapsed_seconds: float
    preprocess_elapsed_seconds: float
    api_image_width: int
    api_image_height: int
    api_image_bytes: int
    traffic_type: str | None = None


@dataclass
class EasyOcrCandidate:
    text: str
    normalized_text: str
    probability: float
    bounding_box: list[int]
    center: dict[str, int]
    variant: str


def build_skipped_validation_result(result: GeminiLocalizationResult) -> ValidationResult:
    return ValidationResult(
        passed=True,
        score=0,
        total_checks=0,
        checks={},
        metrics={"skipped": True, "found": result.found},
        reasons=["Classical validation was skipped."],
    )


def preprocess_for_gemini(
    image: np.ndarray,
    *,
    use_grayscale: bool = False,
    use_clahe: bool = False,
) -> np.ndarray:
    processed = image.copy()

    if use_grayscale:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if use_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        processed = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    return processed


def prepare_api_image_part(
    image: np.ndarray,
    *,
    api_max_dim: int = API_IMAGE_MAX_DIM,
    api_jpeg_quality: int = API_IMAGE_JPEG_QUALITY,
) -> tuple[str, int, int, int]:
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
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:image/jpeg;base64,{image_b64}"
    return image_url, api_image_width, api_image_height, len(image_bytes)


def parse_target_letters(letter_arg: str) -> list[str]:
    target_letters = [letter.strip().upper() for letter in letter_arg.split(",") if letter.strip()]
    if not target_letters:
        raise ValueError("At least one target letter is required.")

    invalid_letters = [
        letter
        for letter in target_letters
        if not ((len(letter) == 1 and letter.isalpha()) or letter in {"SPACE", "ENTER"})
    ]
    if invalid_letters:
        raise ValueError(
            "Each target must be a single alphabetic character, SPACE, or ENTER."
        )

    deduplicated_letters: list[str] = []
    seen_letters: set[str] = set()
    for letter in target_letters:
        if letter in seen_letters:
            continue
        seen_letters.add(letter)
        deduplicated_letters.append(letter)

    return deduplicated_letters

def parse_single_letter(letter_arg: str) -> str:
    target_letters = parse_target_letters(letter_arg)
    if len(target_letters) != 1:
        raise ValueError("track_to_wld expects exactly one target letter, for example --letter X.")
    return target_letters[0]


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
def build_response_schema() -> dict[str, Any]:
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

Target keys: {target_letters_text}
Image size: {image_width}x{image_height}

Return strict JSON only.
- Assume a standard QWERTY keyboard viewed from above.
- Top-level object: {{"results": [...]}}
- Exactly {len(target_letters)} results, in this exact order: {target_letters_text}
- For each result return only: center, bounding_box
- Coordinates must be integers in [0,1000] over the full image extent, never pixels
- bbox format must be [xmin, ymin, xmax, ymax]
- If a target is SPACE, localize the center of the keyboard spacebar key
- Disambiguation examples:
    - E is on the top row between W and R, and is above/above-left of D.
    - R is on the top row between E and T, and is above-left of F.
    - T is on the top row to the right of R, and is above/above-right of F.
    - W is on the top row between Q and E, and is above-left of S.
    - Q is the leftmost top-row letter key.
- If a key is not visible: center=null, bounding_box=null
""".strip()

def build_task1_prompt(image_width: int, image_height: int) -> str:
    return f"""
Localize the keyboard keys needed for Task 1 in one image.

Task 1 success is pressing SPACE, ENTER, R, and L sequentially and in that order.
Target keys: SPACE, ENTER, R, L
Image size: {image_width}x{image_height}

Return strict JSON only.
- Assume a standard QWERTY keyboard layout viewed from above.
    Keys are arranged in rows:
    Top letter row: Q W E R T Y U I O P
    Home row: A S D F G H J K L
    Bottom row: Z X C V B N M
- Top-level object: {{"results": [...]}}
- Exactly 4 results, in this exact order: SPACE, ENTER, R, L
- For each result return only: center, bounding_box
- Coordinates must be integers in [0,1000] over the full image extent, never pixels
- bbox format must be [xmin, ymin, xmax, ymax]
- SPACE means the keyboard spacebar key and you MUST LOCATE ITS MIDDLE POINT, NOT ONE OF THE TWO EDGES
- Locate the center of the word Enter on the ENTER key. It is on the right side of the keyboard, below Backspace, and taller than wide.
- For R: first use the surrounding keyboard layout internally to disambiguate it. R is on the top letter row, immediately to the right of E and 
  immediately to the left of T. Relative to F, R is above-left of F. Relative to D, R is above-right of D.
  Do not return F. Return only the center and bounding_box of R.
- Return the center of the physical key surface, not the printed glyph/ink
- If a key is not visible: center=null, bounding_box=null
""".strip()


@lru_cache(maxsize=16)
def _get_openai_client() -> OpenAI:
    return OpenAI()


def _resolve_model_for_provider(provider: str, model: str) -> str:
    if provider == "gemini" and model == DEFAULT_OPENAI_MODEL:
        return DEFAULT_GEMINI_MODEL
    return model


def _extract_api_image_base64(image_url: str) -> str:
    if "," not in image_url:
        raise ValueError("Expected a data URL with base64 image data.")
    return image_url.split(",", 1)[1]


def _extract_api_image_bytes(image_url: str) -> bytes:
    return base64.b64decode(_extract_api_image_base64(image_url))


def _call_openai_localizer(
    *,
    image_url: str,
    prompt: str,
    response_schema: dict[str, Any],
    model: str,
    start_time: float,
    preprocess_elapsed_seconds: float,
    api_image_width: int,
    api_image_height: int,
    api_image_bytes: int,
) -> GeminiCallResult:
    client = _get_openai_client()
    print(f"Calling OpenAI with model {model}...")
    request_start_time = time.perf_counter()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "keyboard_localization",
                "schema": response_schema,
                "strict": True,
            }
        },
        reasoning={"effort": "low"},
        max_output_tokens=1024,
    )
    response_text = response.output_text
    if not response_text:
        raise RuntimeError(f"OpenAI model {model} returned an empty response.")
    return GeminiCallResult(
        response_text=response_text,
        model_used=model,
        provider_used="openai",
        elapsed_seconds=time.perf_counter() - start_time,
        request_elapsed_seconds=time.perf_counter() - request_start_time,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
        api_image_width=api_image_width,
        api_image_height=api_image_height,
        api_image_bytes=api_image_bytes,
    )


def _vertex_request_headers(gemini_backend: str) -> dict[str, str]:
    mode = gemini_backend.strip().lower()
    if mode not in GEMINI_BACKENDS:
        raise ValueError(
            f"Unknown Gemini backend `{gemini_backend}`. "
            f"Expected one of: {', '.join(sorted(GEMINI_BACKENDS))}."
        )

    if mode == "standard":
        return {"X-Vertex-AI-LLM-Request-Type": "shared"}
    if mode == "priority":
        return {
            "X-Vertex-AI-LLM-Request-Type": "shared",
            "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
        }
    return {"X-Vertex-AI-LLM-Request-Type": "dedicated"}


@lru_cache(maxsize=16)
def _get_vertex_client(project: str, location: str, gemini_backend: str):
    if genai is None or types is None:
        raise RuntimeError("Missing Gemini SDK. Install it with `pip install google-genai`.")

    mode = gemini_backend.strip().lower()
    if mode == "priority" and location.lower() != "global":
        raise ValueError("Priority PayGo is supported only on the `global` Vertex AI endpoint.")

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(
            api_version="v1",
            headers=_vertex_request_headers(mode),
        ),
    )


def _response_traffic_type(response: Any) -> str | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    traffic_type = getattr(usage_metadata, "traffic_type", None)
    if traffic_type is None:
        return None
    return getattr(traffic_type, "name", str(traffic_type))


def _model_id(model_name: str) -> str:
    return model_name.strip().split("/")[-1].lower()


def _build_generate_content_config(
    model_name: str,
    response_schema: dict[str, Any],
):
    if types is None:
        raise RuntimeError("Missing Gemini SDK. Install it with `pip install google-genai`.")

    if _model_id(model_name) in TEXT_ONLY_IMAGE_OUTPUT_MODELS:
        return types.GenerateContentConfig(
            response_modalities=[types.Modality.TEXT],
            temperature=0,
        )

    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=response_schema,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    )


def _call_google_gemini_localizer(
    *,
    image_bytes: bytes,
    prompt: str,
    response_schema: dict[str, Any],
    model: str,
    project: str | None,
    location: str,
    gemini_backend: str,
    start_time: float,
    preprocess_elapsed_seconds: float,
    api_image_width: int,
    api_image_height: int,
    api_image_bytes: int,
) -> GeminiCallResult:
    if not project:
        raise ValueError("Missing Google Cloud project. Set GOOGLE_CLOUD_PROJECT or pass --project.")
    if types is None or genai_errors is None:
        raise RuntimeError("Missing Gemini SDK. Install it with `pip install google-genai`.")

    client = _get_vertex_client(project, location, gemini_backend)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    print(f"Calling Gemini with model {model}...")
    request_start_time = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[image_part, prompt],
            config=_build_generate_content_config(model, response_schema),
        )
    except genai_errors.APIError as exc:
        raise RuntimeError(_format_gemini_api_error(model, exc)) from exc

    response_text = response.text
    if not response_text:
        raise RuntimeError(f"Gemini model {model} returned an empty response.")
    traffic_type = _response_traffic_type(response)
    if traffic_type is not None:
        print(f"Gemini traffic type: {traffic_type}")

    return GeminiCallResult(
        response_text=response_text,
        model_used=model,
        provider_used="gemini",
        elapsed_seconds=time.perf_counter() - start_time,
        request_elapsed_seconds=time.perf_counter() - request_start_time,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
        api_image_width=api_image_width,
        api_image_height=api_image_height,
        api_image_bytes=api_image_bytes,
        traffic_type=traffic_type,
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
    gemini_backend: str = "standard",
    provider: str = "openai",
) -> GeminiCallResult:
    if provider not in LOCALIZATION_PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(LOCALIZATION_PROVIDERS)}.")

    start_time = time.perf_counter()
    preprocess_start_time = time.perf_counter()
    image_url, api_image_width, api_image_height, api_image_bytes = prepare_api_image_part(
        image,
        api_max_dim=api_max_dim,
        api_jpeg_quality=api_jpeg_quality,
    )
    if target_letters == ["SPACE", "ENTER", "R", "L"]:
        prompt = build_task1_prompt(image_width=image_width, image_height=image_height)
    else:
        prompt = build_gemini_prompt(
            target_letters=target_letters,
            image_width=image_width,
            image_height=image_height,
        )
    preprocess_elapsed_seconds = time.perf_counter() - preprocess_start_time

    model_candidates = [_resolve_model_for_provider(provider, model), *fallback_models]
    seen_models: set[str] = set()
    last_error: Exception | None = None
    response_schema = build_response_schema()

    for index, candidate_model in enumerate(model_candidates, start=1):
        candidate_model = candidate_model.strip()
        if not candidate_model or candidate_model in seen_models:
            continue
        seen_models.add(candidate_model)

        try:
            if provider == "openai":
                return _call_openai_localizer(
                    image_url=image_url,
                    prompt=prompt,
                    response_schema=response_schema,
                    model=candidate_model,
                    start_time=start_time,
                    preprocess_elapsed_seconds=preprocess_elapsed_seconds,
                    api_image_width=api_image_width,
                    api_image_height=api_image_height,
                    api_image_bytes=api_image_bytes,
                )
            return _call_google_gemini_localizer(
                image_bytes=_extract_api_image_bytes(image_url),
                prompt=prompt,
                response_schema=response_schema,
                model=candidate_model,
                project=project,
                location=location,
                gemini_backend=gemini_backend,
                start_time=start_time,
                preprocess_elapsed_seconds=preprocess_elapsed_seconds,
                api_image_width=api_image_width,
                api_image_height=api_image_height,
                api_image_bytes=api_image_bytes,
            )
        except (APIError, RuntimeError) as exc:
            last_error = exc
            if _is_retryable_unavailable_error(exc) and index < len(model_candidates):
                print(
                    f"Model {candidate_model} is temporarily overloaded (503). "
                    f"Retrying with the next configured model..."
                )
                time.sleep(1.0)
                continue
            if provider == "openai" and isinstance(exc, APIError):
                status_code = getattr(exc, "status_code", "unknown")
                details = getattr(exc, "message", None) or str(exc)
                raise RuntimeError(
                    f"OpenAI request failed for model `{candidate_model}` "
                    f"(status: {status_code}). Details: {details}"
                ) from exc
            raise

    raise RuntimeError(f"{provider} call failed without a usable response.") from last_error



_easyocr_reader = None

def get_easyocr_reader():
    """Carica il modello EasyOCR in memoria solo quando viene effettivamente richiesto."""
    global _easyocr_reader
    if _easyocr_reader is None:
        print("Inizializzazione del modello EasyOCR in locale in corso (richiede qualche secondo)...")
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
    raw_text = _strip_json_fence(raw_text)
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


def _strip_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


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


def localize_with_gemini(
    frame: np.ndarray,
    *,
    letter: str,
    model: str,
    fallback_models: list[str],
    project: str | None,
    location: str,
    gemini_backend: str = "standard",
    provider: str = "openai",
) -> GeminiLocalizationResult:
    """
    Main block of the VLM keypoint localization. Calls gemini API, then validates the result by running sanity checks
    on the answer. 
    """
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
        gemini_backend=gemini_backend,
        provider=provider,
    )
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
        f"Initial localization: center=({result.center['x']}, {result.center['y']}), "
        f"cv_check={'PASS' if validation.passed else 'FAIL'}"
    )
    return result


def point_from_result(result: GeminiLocalizationResult) -> np.ndarray:
    """
    Returns a single pixel from the predicted key bounding box.
    """
    if result.bounding_box is None:
        raise ValueError("Cannot initialize tracking without a Gemini bounding box.")
    xmin, ymin, xmax, ymax = result.bounding_box
    if result.raw_response.get("provider") == "easyocr":
        return np.array([(xmax + xmin) / 2, (ymax + ymin) / 2], dtype=np.float32)

    if result.target_letter == "SPACE": 
        result_arr = np.array([(xmax+xmin)/2, (ymax+ymin)/1.975], dtype=np.float32)
    elif result.target_letter == "ENTER": 
        result_arr = np.array([(xmax+xmin)/2, ymin + 5], dtype=np.float32)
    else: 
        result_arr = np.array([(xmax+xmin)/2, (ymax+ymin)/2], dtype=np.float32)

    return result_arr
