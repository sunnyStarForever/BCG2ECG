from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import Delaunay


RAW_SENSOR_COUNT = 1056
PRESSURE_ROWS = 77
PRESSURE_COLS = 32
PRESSURE_SHAPE = (PRESSURE_ROWS, PRESSURE_COLS)

# The legacy hardware layout contains a 6x4 unmeasured region.  The reference
# implementation intended to interpolate this region after mapping.
MISSING_REGION_ROWS = slice(38, 44)
MISSING_REGION_COLS = slice(14, 18)

LAYOUT_MAP_PATH = Path(__file__).with_name("pressure_layout_map.npz")


@lru_cache(maxsize=1)
def load_layout_matrix() -> sparse.csr_matrix:
    """Load the hardware wiring map from 1056 channels to a 77x32 pressure map."""
    matrix = sparse.load_npz(LAYOUT_MAP_PATH).tocsr().astype(np.float32)
    expected_shape = (PRESSURE_ROWS * PRESSURE_COLS, RAW_SENSOR_COUNT)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Pressure layout matrix has shape {matrix.shape}, expected {expected_shape}."
        )
    if matrix.nnz != 2306:
        raise ValueError(f"Pressure layout matrix has {matrix.nnz} entries, expected 2306.")
    coefficients = np.unique(matrix.data)
    if not np.array_equal(coefficients, np.asarray([0.5, 1.0], dtype=np.float32)):
        raise ValueError(f"Unexpected pressure layout coefficients: {coefficients}")
    return matrix


def map_pressure_frames(raw_frames: np.ndarray) -> np.ndarray:
    """
    Apply the calibrated hardware wiring map and mattress-orientation flip.

    The reference mapping is linear.  Most output pixels copy one sensor, while
    27 pixels average two adjacent source channels.  The sparse matrix is an
    exact refactor of the original assignment-based transformer.
    """
    raw = np.asarray(raw_frames, dtype=np.float32)
    if raw.shape[-1] != RAW_SENSOR_COUNT:
        raise ValueError(
            f"Expected last pressure dimension {RAW_SENSOR_COUNT}, got {raw.shape}."
        )
    leading_shape = raw.shape[:-1]
    flat = raw.reshape(-1, RAW_SENSOR_COUNT)
    mapped = np.asarray(flat @ load_layout_matrix().T, dtype=np.float32)
    mapped = mapped.reshape(*leading_shape, PRESSURE_ROWS, PRESSURE_COLS)
    return np.flip(mapped, axis=-1).copy()


@lru_cache(maxsize=1)
def _missing_interpolation_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute Delaunay vertices and barycentric weights for the fixed gap.

    This is equivalent to linear ``griddata`` for fixed sensor coordinates but
    avoids recomputing the triangulation for every frame.
    """
    rows, cols = PRESSURE_SHAPE
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    missing = np.zeros(PRESSURE_SHAPE, dtype=bool)
    missing[MISSING_REGION_ROWS, MISSING_REGION_COLS] = True

    valid_flat = np.flatnonzero(~missing.ravel())
    missing_flat = np.flatnonzero(missing.ravel())
    valid_points = np.column_stack(
        [x_grid.ravel()[valid_flat], y_grid.ravel()[valid_flat]]
    ).astype(float)
    missing_points = np.column_stack(
        [x_grid.ravel()[missing_flat], y_grid.ravel()[missing_flat]]
    ).astype(float)

    triangulation = Delaunay(valid_points)
    simplex = triangulation.find_simplex(missing_points)
    if np.any(simplex < 0):
        raise RuntimeError("Pressure gap contains points outside interpolation hull.")
    transform = triangulation.transform[simplex]
    delta = missing_points - transform[:, 2]
    first_weights = np.einsum("nij,nj->ni", transform[:, :2], delta)
    weights = np.column_stack(
        [first_weights, 1.0 - first_weights.sum(axis=1)]
    ).astype(np.float32)
    vertex_flat = valid_flat[triangulation.simplices[simplex]]
    return missing_flat, vertex_flat, weights


def fill_missing_region(mapped_frames: np.ndarray) -> np.ndarray:
    """Linearly interpolate the fixed 6x4 unmeasured pressure region."""
    values = np.asarray(mapped_frames, dtype=np.float32)
    if values.shape[-2:] != PRESSURE_SHAPE:
        raise ValueError(
            f"Expected pressure maps ending in {PRESSURE_SHAPE}, got {values.shape}."
        )
    output = values.copy()
    leading_shape = output.shape[:-2]
    flat = output.reshape(-1, PRESSURE_ROWS * PRESSURE_COLS)
    missing_flat, vertex_flat, weights = _missing_interpolation_weights()
    interpolated = np.sum(flat[:, vertex_flat] * weights[None, :, :], axis=-1)
    flat[:, missing_flat] = interpolated
    return flat.reshape(*leading_shape, PRESSURE_ROWS, PRESSURE_COLS)


def otsu_threshold(values: np.ndarray, levels: int = 256) -> float:
    """
    Compute an OTSU threshold in the original pressure-value domain.

    The legacy code calculated a threshold on a 0-255 rescaling and then
    compared that threshold directly with raw pressure values.  This corrected
    implementation maps the histogram threshold back to the original units.
    """
    image = np.asarray(values, dtype=np.float64)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if maximum <= minimum:
        return maximum

    histogram, edges = np.histogram(
        finite,
        bins=levels,
        range=(minimum, maximum),
    )
    probability = histogram.astype(np.float64)
    probability /= max(probability.sum(), 1.0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative_probability = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * centers)
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between = np.zeros_like(denominator)
    valid = denominator > 1e-12
    between[valid] = (
        global_mean * cumulative_probability[valid]
        - cumulative_mean[valid]
    ) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between))])


def remove_small_components(binary_map: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Retain 4-connected foreground components whose area is greater than min_area."""
    binary = np.asarray(binary_map, dtype=bool)
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, count = ndimage.label(binary, structure=structure)
    if count == 0:
        return np.zeros(binary.shape, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = areas > min_area
    keep[0] = False
    return keep[labels]


def zscore_nonzero(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score non-zero foreground values while preserving a zero background."""
    image = np.asarray(values, dtype=np.float32)
    output = np.zeros_like(image)
    foreground = image != 0
    if not np.any(foreground):
        return output
    foreground_values = image[foreground]
    mean = float(np.mean(foreground_values))
    std = float(np.std(foreground_values))
    if std < eps:
        output[foreground] = foreground_values - mean
    else:
        output[foreground] = (foreground_values - mean) / std
    return output


def preprocess_pressure_sequence(
    raw_sequence: np.ndarray,
    *,
    min_object_size: int = 10,
    use_zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert one raw pressure sequence into mapped, masked, and clean maps.

    Returns
    -------
    mapped:
        Hardware-mapped, horizontally flipped, gap-filled pressure maps.
    masked:
        Mapped pressure after OTSU segmentation and small-component removal.
    clean:
        Masked pressure after optional foreground-only z-score normalization.
    thresholds:
        Per-frame OTSU thresholds in original pressure units.
    """
    raw = np.asarray(raw_sequence, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != RAW_SENSOR_COUNT:
        raise ValueError(
            f"Expected pressure sequence (frames, {RAW_SENSOR_COUNT}), got {raw.shape}."
        )
    mapped = fill_missing_region(map_pressure_frames(raw))
    masked = np.zeros_like(mapped)
    clean = np.zeros_like(mapped)
    thresholds = np.empty(mapped.shape[0], dtype=np.float32)
    for frame_index, frame in enumerate(mapped):
        threshold = otsu_threshold(frame)
        foreground = remove_small_components(
            frame > threshold,
            min_area=min_object_size,
        )
        masked_frame = np.where(foreground, frame, 0.0).astype(np.float32)
        thresholds[frame_index] = threshold
        masked[frame_index] = masked_frame
        clean[frame_index] = (
            zscore_nonzero(masked_frame) if use_zscore else masked_frame
        )
    return mapped, masked, clean, thresholds
