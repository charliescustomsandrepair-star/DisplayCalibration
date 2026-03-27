"""
Calibration math used by the capture server (Python + OpenCV).
"""
import cv2
import heapq
import numpy as np


def gamma_linearize(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    if gamma <= 0:
        gamma = 2.2
    x = img.astype(np.float32) / 255.0
    return np.power(np.clip(x, 1e-6, 1.0), 1.0 / gamma)


def compute_modulation_n(images: list) -> tuple:
    """N-step phase shifting; phase_i = 2π i / N (N must match capture count per frequency)."""
    n = len(images)
    if n < 3:
        raise ValueError("Need at least 3 phase-shifted images")
    acc_sin = np.zeros_like(images[0], dtype=np.float32)
    acc_cos = np.zeros_like(images[0], dtype=np.float32)
    for i, im in enumerate(images):
        ph = 2.0 * np.pi * float(i) / float(n)
        f = im.astype(np.float32)
        if f.max() > 1.5:
            f *= 1.0 / 255.0
        acc_sin += f * np.sin(ph)
        acc_cos += f * np.cos(ph)
    wrapped = np.arctan2(acc_sin, acc_cos)
    modulation = np.hypot(acc_sin, acc_cos) * (2.0 / float(n))
    return wrapped, modulation


def quality_guided_unwrap(wrapped: np.ndarray, quality: np.ndarray) -> np.ndarray:
    h, w = wrapped.shape
    unwrapped = np.zeros_like(wrapped, dtype=np.float32)
    processed = np.zeros((h, w), dtype=np.uint8)

    max_idx = np.unravel_index(int(np.argmax(quality)), quality.shape)
    y0, x0 = max_idx
    unwrapped[y0, x0] = wrapped[y0, x0]
    processed[y0, x0] = 1
    pq: list = [(-float(quality[y0, x0]), y0, x0)]

    neigh = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while pq:
        _, y, x = heapq.heappop(pq)
        if processed[y, x]:
            continue
        ref = 0.0
        cnt = 0
        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and processed[ny, nx]:
                ref += float(unwrapped[ny, nx])
                cnt += 1
        if cnt == 0:
            continue
        ref /= cnt
        wv = float(wrapped[y, x])
        diff = wv - ref
        nwrap = int(np.round(diff / (2.0 * np.pi)))
        unwrapped[y, x] = wv - nwrap * 2.0 * np.pi
        processed[y, x] = 1
        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not processed[ny, nx]:
                heapq.heappush(pq, (-float(quality[ny, nx]), ny, nx))

    # Fill any unreachable pixels (pathological) with wrapped value
    mask = processed == 0
    if np.any(mask):
        unwrapped[mask] = wrapped[mask]
    return unwrapped


def fit_weighted_quadratic(
    unwrapped: np.ndarray, weight: np.ndarray
) -> tuple:
    """
    Weighted LS fit: phase ≈ a x^2 + b y^2 + c x y + d x + e y + f.
    Returns (coeffs shape (6,1), rms residual).
    """
    h, w = unwrapped.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    flat_w = np.clip(weight.astype(np.float64).ravel(), 1e-9, None)
    X = np.column_stack(
        [
            xs.ravel() ** 2,
            ys.ravel() ** 2,
            (xs * ys).ravel(),
            xs.ravel(),
            ys.ravel(),
            np.ones(h * w, dtype=np.float64),
        ]
    )
    b = unwrapped.astype(np.float64).ravel()
    sqrt_w = np.sqrt(flat_w)
    Xw = X * sqrt_w[:, None]
    bw = b * sqrt_w
    coeffs, _, _, _ = np.linalg.lstsq(Xw, bw, rcond=None)
    pred = (X @ coeffs).reshape(h, w)
    rms = float(np.sqrt(np.mean((pred - unwrapped.astype(np.float64)) ** 2)))
    return coeffs.astype(np.float32).reshape(6, 1), rms


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    return np.outer(wy, wx)


def detect_layout_fft(img: np.ndarray) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = gray * _hann2d(gray.shape[0], gray.shape[1])
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
    stripe_energy = float(np.sum(mag[cy - 3 : cy + 4, max(cx // 3 - 3, 0) : cx // 3 + 4]))
    pentile_energy = float(np.sum(mag[cy - 3 : cy + 4, max(cx // 2 - 3, 0) : cx // 2 + 4]))
    if pentile_energy > stripe_energy * 1.5:
        return "Pentile"
    return "RGB Stripe"


def gray_world_gains(bgr: np.ndarray) -> list:
    """RGB multipliers (~1.0) to neutralize average color cast (sRGB-ish)."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = float(b.mean()), float(g.mean()), float(r.mean())
    gray = (mb + mg + mr) / 3.0 + 1e-6
    return [
        gray / max(mr, 1e-3),
        gray / max(mg, 1e-3),
        gray / max(mb, 1e-3),
    ]


# Back-compat
def compute_modulation(images: list) -> tuple:
    return compute_modulation_n(images)
