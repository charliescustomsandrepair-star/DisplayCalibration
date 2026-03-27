"""
Full calibration pipeline aligned with structured-light phase analysis:
- Per-frequency 5-step phase shifting (15 captures = 3 frequencies × 5 steps)
- Gamma estimation via modulation maximization
- Gamma linearization, wrapped phase + modulation quality per frequency
- Quality-guided phase unwrapping per band, multi-frequency bridging
- Weighted quadratic surface fit (perspective / distortion compensation)
- FFT layout hint (stripe vs pentile-like energy)
- Orientation from phase gradient dominance
- Robust fallbacks so a token is always emitted
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from calibration_utils import (
    compute_modulation_n,
    detect_layout_fft,
    fit_weighted_quadratic,
    gamma_linearize,
    gray_world_gains,
    quality_guided_unwrap,
)

log = logging.getLogger("calibration-pipeline")

LAYOUT_MAP = {
    "RGB Stripe": "RGB_STRIPE",
    "Pentile": "PENTILE_RG",
}

PERCEPTUAL = {"r": 1.1, "g": 0.95, "b": 1.05}


def _read_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def _resize_for_fit(img: np.ndarray, max_side: int = 480) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    scale = max_side / float(m)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def estimate_gamma_from_groups(
    groups: List[List[np.ndarray]], gamma_candidates: np.ndarray
) -> float:
    """Pick gamma that maximizes mean modulation (sinusoid visibility) after linearization."""
    best_g, best_score = 2.2, -1.0
    for g in gamma_candidates:
        scores = []
        for imgs in groups:
            if len(imgs) != 5:
                continue
            lin = [gamma_linearize(im, float(g)) for im in imgs]
            _, mod = compute_modulation_n(lin)
            scores.append(float(np.mean(mod)))
        if not scores:
            continue
        s = float(np.mean(scores))
        if s > best_score:
            best_score, best_g = s, float(g)
    return best_g


def multifrequency_unwrap(
    wrapped_by_freq: Dict[int, np.ndarray],
    quality_by_freq: Dict[int, np.ndarray],
    frequencies: List[int],
) -> Dict[int, np.ndarray]:
    """
    Unwrap lowest frequency first, then unwrap higher bands by scaling
    unwrapped low phase to predict 2π wraps on higher bands.
    """
    freqs = sorted(frequencies)
    out: Dict[int, np.ndarray] = {}
    u_low: Optional[np.ndarray] = None
    f_low: Optional[int] = None

    for f in freqs:
        w = wrapped_by_freq[f]
        q = quality_by_freq[f]
        u = quality_guided_unwrap(w, q)
        if u_low is None or f_low is None:
            out[f] = u
            u_low, f_low = u, f
            continue
        pred = u_low * (float(f) / float(f_low))
        k = np.round((pred - u) / (2.0 * np.pi))
        out[f] = u + k * (2.0 * np.pi)
        u_low, f_low = out[f], f
    return out


def derive_geometry_from_fit(
    coeffs: np.ndarray,
    crop_width_px: int,
    pattern_freq_cycles: int,
    layout_name: str,
    horizontal_orientation: bool,
) -> Dict[str, Any]:
    """
    coeffs: (6,) for phase = a x^2 + b y^2 + c x y + d x + e y + f (pixel coords of fit image).
    """
    c = coeffs.ravel()
    d = float(c[3])
    e = float(c[4])
    grad_x_rad_per_px = d
    grad_y_rad_per_px = e

    expected = (2.0 * np.pi * float(pattern_freq_cycles)) / max(float(crop_width_px), 1.0)
    ratio_x = grad_x_rad_per_px / expected if abs(expected) > 1e-9 else 1.0
    ratio_y = grad_y_rad_per_px / max(abs(expected) * 0.25, 1e-9)

    base_pitch = 1.0 / 3.0
    pitch_x = float(np.clip(base_pitch / max(abs(ratio_x), 0.25), 0.2, 0.45))
    pitch_y = float(np.clip(1.0 / max(abs(ratio_y), 0.5), 0.85, 1.15))

    layout_key = LAYOUT_MAP.get(layout_name, "RGB_STRIPE")
    if layout_key == "PENTILE_RG":
        offset_x = [0.0, pitch_x, 2.0 * pitch_x]
        offset_y = [0.0, 0.0, 0.0]
    else:
        offset_x = [0.0, pitch_x, min(1.0 - 1e-3, 2.0 * pitch_x)]
        offset_y = [0.0, 0.0, 0.0]

    return {
        "layout": layout_key,
        "horizontal_orientation": horizontal_orientation,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "phase_gradient_x_rad_per_px": grad_x_rad_per_px,
        "phase_gradient_y_rad_per_px": grad_y_rad_per_px,
        "expected_phase_gradient": float(expected),
    }


def run_calibration_session(
    session_dir: Path,
    pattern_sequence: List[Dict[str, Any]],
    frequencies_order: List[int],
) -> Dict[str, Any]:
    """
    Load processed_*.png in order, run pipeline, return token dict + diagnostics.
    """
    warnings: List[str] = []
    n = len(pattern_sequence)
    images: List[Optional[np.ndarray]] = []
    for i in range(n):
        p = session_dir / f"processed_{i:02d}.png"
        if not p.exists():
            p = session_dir / f"raw_{i:02d}.png"
        if not p.exists():
            images.append(None)
            warnings.append(f"missing_image_{i:02d}")
        else:
            images.append(_read_bgr(p))

    if all(im is None for im in images):
        return _fallback_token(session_dir, "no_images", warnings)

    groups: List[List[np.ndarray]] = []
    group_freq: List[int] = []
    buckets: Dict[int, List[np.ndarray]] = {f: [] for f in frequencies_order}
    for i, pat in enumerate(pattern_sequence):
        f = int(pat["frequency"])
        if f not in buckets:
            continue
        im = images[i] if i < len(images) else None
        if im is not None:
            buckets[f].append(im)

    for f in frequencies_order:
        chunk = buckets.get(f, [])
        if len(chunk) >= 5:
            groups.append(chunk[:5])
            group_freq.append(f)
        elif len(chunk) > 0:
            warnings.append(f"frequency_{f}_only_{len(chunk)}_frames")

    if len(groups) != len(frequencies_order):
        warnings.append("incomplete_frequency_groups_using_available")

    gamma_grid = np.concatenate(
        [np.linspace(1.7, 2.8, 24), np.array([2.0, 2.1, 2.2, 2.4])]
    )
    gamma_est = estimate_gamma_from_groups(groups, gamma_grid) if groups else 2.2

    linear_groups = [[gamma_linearize(im, gamma_est) for im in g] for g in groups]

    wrapped_by_freq: Dict[int, np.ndarray] = {}
    quality_by_freq: Dict[int, np.ndarray] = {}
    mean_modulations: Dict[int, float] = {}

    ref_shape = None
    for f, lg in zip(group_freq, linear_groups):
        wq, mod = compute_modulation_n(lg)
        if ref_shape is None:
            ref_shape = wq.shape
        if wq.shape != ref_shape:
            wq = cv2.resize(wq, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
            mod = cv2.resize(mod, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
        wrapped_by_freq[f] = wq
        quality_by_freq[f] = mod
        mean_modulations[f] = float(np.mean(mod))

    if not wrapped_by_freq:
        return _fallback_token(session_dir, "no_valid_groups", warnings)

    unwrapped_by_freq = multifrequency_unwrap(
        wrapped_by_freq, quality_by_freq, list(wrapped_by_freq.keys())
    )

    mid_f = group_freq[len(group_freq) // 2]
    u_full = unwrapped_by_freq[mid_f]
    w_full = quality_by_freq[mid_f]

    u_fit, scale = _resize_for_fit(u_full, max_side=512)
    w_fit, _ = _resize_for_fit(w_full, max_side=512)
    coeffs, rms_fit = fit_weighted_quadratic(u_fit, w_fit)

    layout_name = detect_layout_fft(groups[0][0])
    crop_h, crop_w = u_full.shape[:2]

    horiz = float(abs(coeffs[3])) >= float(abs(coeffs[4])) * 0.85

    geom = derive_geometry_from_fit(coeffs, crop_w, mid_f, layout_name, horiz)

    gains = gray_world_gains(groups[0][0])

    rms_phase = float(rms_fit)
    token = {
        "version": 1,
        "schema": "display_calibration_token",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_dir.name,
        "gamma_estimate": gamma_est,
        "illuminant_rgb_gains": gains,
        "pattern_frequencies": frequencies_order,
        "phase_carrier_frequency_used": mid_f,
        "mean_modulation_by_frequency": mean_modulations,
        "layout": geom["layout"],
        "horizontal_orientation": geom["horizontal_orientation"],
        "pitch_x": geom["pitch_x"],
        "pitch_y": geom["pitch_y"],
        "offset_x": geom["offset_x"],
        "offset_y": geom["offset_y"],
        "perceptual_weights": PERCEPTUAL.copy(),
        "phase_fit_coefficients": [float(x) for x in coeffs.ravel().tolist()],
        "phase_fit_rms": rms_phase,
        "quality_metrics": {
            "rms_phase_fit": rms_phase,
            "mean_modulation_mid_freq": mean_modulations.get(mid_f, 0.0),
        },
        "warnings": warnings,
        "pipeline": {
            "gamma_linearization": True,
            "quality_guided_unwrap": True,
            "weighted_quadratic_fit": True,
            "multifrequency_unwrap": True,
            "fft_layout_hint": True,
        },
    }
    return token


def _fallback_token(session_dir: Path, reason: str, warnings: List[str]) -> Dict[str, Any]:
    w = list(warnings)
    w.append(f"fallback:{reason}")
    return {
        "version": 1,
        "schema": "display_calibration_token",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_dir.name,
        "gamma_estimate": 2.2,
        "illuminant_rgb_gains": [1.0, 1.0, 1.0],
        "layout": "RGB_STRIPE",
        "horizontal_orientation": True,
        "pitch_x": 1.0 / 3.0,
        "pitch_y": 1.0,
        "offset_x": [0.0, 1.0 / 3.0, 2.0 / 3.0],
        "offset_y": [0.0, 0.0, 0.0],
        "perceptual_weights": PERCEPTUAL.copy(),
        "warnings": w,
        "quality_metrics": {"rms_phase_fit": None, "mean_modulation_mid_freq": None},
        "pipeline": {"fallback": True, "reason": reason},
    }


def save_token(token: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(token, f, indent=2)
