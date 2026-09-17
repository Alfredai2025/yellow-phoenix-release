#!/usr/bin/env python3
"""
Compare ITQ rotation matrix against spectral eigenvectors.
If drift > 2.0, the encoder has lost alignment with data geometry.
"""
import numpy as np
import ctypes
import sys

ITQ_PATH = "itq_model_512.npz"


def load_itq_rotation(path=ITQ_PATH):
    data = np.load(path)
    rot = data["rotation"]
    return rot.astype(np.float32)


def load_spectral_eigenvectors(lib_path="./target/release/libpams.dylib"):
    lib = ctypes.CDLL(lib_path)
    lib.yp_spectral_eigenvectors.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.yp_spectral_eigenvectors.restype = ctypes.c_size_t

    buf = (ctypes.c_float * 512)()
    written = lib.yp_spectral_eigenvectors(buf, 512)
    if written == 0:
        print("[drift] No spectral eigenvectors available.")
        return None
    return np.array(buf[:written], dtype=np.float32)


def check_drift(itq_rot, spectral_eigen, top_k=64):
    """
    NOTE: yp_spectral_eigenvectors currently returns DUMMY eigenvectors.
    This check is MEANINGLESS until real eigenvectors are exposed from tensor_spectral.rs.
    To fix: store eigenvectors in HybridMesh at build time and expose via FFI.
    """
    if spectral_eigen is None:
        print("[drift] Skipping — no spectral data.")
        return None

    itq_sub = itq_rot[:top_k, :top_k]
    spec_sub = spectral_eigen[:top_k] if len(spectral_eigen) >= top_k else spectral_eigen

    min_dim = min(itq_sub.shape[0], len(spec_sub))
    itq_sub = itq_sub[:min_dim, :min_dim]
    spec_sub = spec_sub[:min_dim]

    distance = np.linalg.norm(itq_sub - np.eye(min_dim) * spec_sub[:, None])
    print(f"[drift] ITQ vs Spectral (top {min_dim}): {distance:.4f}")

    if distance > 2.0:
        print("[drift] ⚠️  DRIFT DETECTED. Retrain encoder.")
    elif distance > 1.0:
        print("[drift] ⚠️  Mild drift. Monitor closely.")
    else:
        print("[drift] ✅ Aligned.")

    return distance


if __name__ == "__main__":
    try:
        itq = load_itq_rotation()
    except FileNotFoundError:
        print(f"[drift] ITQ model not found at {ITQ_PATH}")
        sys.exit(1)

    spectral = load_spectral_eigenvectors()
    check_drift(itq, spectral)
