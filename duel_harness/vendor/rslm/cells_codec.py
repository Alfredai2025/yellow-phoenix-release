# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from: https://github.com/google-research/google-research/tree/master/rslm
# (code cells of rslm.ipynb, imports trimmed to codec needs only; codec code unmodified)
# Paper: Lenhardt et al., "RSLM: Training-Free Vector Quantization for ANN Search",
#        arXiv:2608.30384 (CC-BY 4.0)

import bisect
import math
import os
import struct
import numpy as np

# ==============================================================================
# 1. Rotation Tables & Constants (2-Pass Cascaded Fast Walsh-Hadamard Transform)
# ==============================================================================
FLIPS = [1.0 if c == '+' else -1.0 for c in (
    "+-----+--+----+++++--+++-++++-++-++++-++--+--+----++----+--+---++++-++++--+--++++---++++"
    "-+--++-++++-++++--+----+----++++-++-++--++++-++++-++-++-++-++--++--+--+-++++--++-++++--++"
    "++---++--+-+++-+++-+++++++---+-+--++++-++++++++-+--++--+--++++--+++++----++-+----"
)]

PERM = [
    106, 71, 37, 89, 32, 11, 101, 120, 19, 18, 24, 114, 103, 63, 58, 92,
    44, 38, 76, 23, 20, 1, 95, 17, 45, 82, 74, 14, 86, 5, 13, 123,
    117, 34, 53, 109, 40, 107, 115, 48, 49, 41, 3, 73, 52, 100, 22, 64,
    80, 55, 6, 12, 26, 94, 113, 50, 87, 105, 127, 36, 90, 59, 46, 111,
    102, 118, 35, 125, 65, 78, 42, 4, 110, 79, 126, 9, 0, 99, 81, 29,
    108, 75, 2, 43, 116, 28, 31, 15, 57, 66, 56, 47, 83, 85, 51, 39,
    91, 25, 88, 119, 96, 69, 27, 54, 77, 67, 33, 21, 70, 60, 84, 124,
    16, 98, 68, 97, 8, 104, 62, 93, 122, 10, 121, 72, 112, 30, 7, 61
]
INV_PERM = [PERM.index(i) for i in range(128)]

C1D = {
    4: ([-2.73263, -2.06904, -1.61797, -1.25623, -0.94236, -0.65676, -0.38810, -0.12840,
         0.12840, 0.38810, 0.65676, 0.94236, 1.25623, 1.61797, 2.06904, 2.73263],
        [-2.400835, -1.843505, -1.437100, -1.099295, -0.799560, -0.522430, -0.258250, 0.0,
         0.258250, 0.522430, 0.799560, 1.099295, 1.437100, 1.843505, 2.400835]),
    3: ([-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152],
        [-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748]),
}

C2D = list(zip(
    [-0.495701, -0.867608, -0.972934, 0.137670, -0.003935, 1.302806, -1.734326, 0.936089,
     0.426494, 0.336818, 1.480261, 0.398970, -1.156616, -0.556493, 1.994463, -1.767952],
    [-0.955058, -0.126081, -1.862329, 1.779942, -0.103267, 1.235003, -0.757077, -0.007075,
     -0.856475, 0.714455, -1.110360, -1.914858, 1.603435, 0.716960, 0.116584, 0.495094]
))

C4D = list(zip(
    [-0.667223, -0.000491, 1.188668, -0.467725, -0.229568, 0.511224, 1.366460, -0.532846,
     0.698791, 0.173459, -1.366599, -0.683748, -1.431984, 0.013678, 0.228653, 1.178144],
    [-1.156572, -0.000645, -0.673327, 0.209985, -1.455663, -1.060941, -0.348822, 0.303232,
     1.232136, -0.451141, -0.395756, 1.503576, 0.286363, 0.749156, 0.377476, 0.832109],
    [0.867890, -0.001313, 0.886783, -0.069217, -0.377640, -0.839085, -0.473865, -0.567821,
     0.409275, 1.204719, -0.898790, -0.367668, 0.771768, 1.416614, -1.638933, -0.293186],
    [0.619376, -0.002269, 0.485405, 1.624921, -0.769526, 0.904128, -0.835606, -1.482903,
     -0.842221, -1.121391, 0.266300, 0.225374, -0.418472, 0.582220, -0.070251, 0.860906]
))

# ==============================================================================
# 2. Ue7m9 2-Byte Scalar Conversion (Bit-Exact to C++ unsigned_float.h)
# ==============================================================================
def ue7m9_enc(val: float) -> bytes:
  """Converts float to 2-byte Ue7m9 format (7 exponent bits, 9 mantissa bits)."""
  if val <= 0.0 or math.isnan(val):
    return b'\x00\x00'
  u = struct.unpack('<' + 'I', struct.pack('<' + 'f', float(val)))[0]
  if u < 0x20800000:
    return b'\x00\x00'
  if u >= 0x5F800000:
    return b'\xFF\xFF'
  u -= 0x20000000
  u += 8191 + ((u >> 14) & 1)
  return struct.pack('<' + 'H', (u >> 14) & 0xFFFF)

def ue7m9_dec(raw_bytes: bytes) -> float:
  """Converts 2-byte Ue7m9 format back to float."""
  raw_bits = struct.unpack('<' + 'H', raw_bytes)[0]
  if not raw_bits:
    return 0.0
  u = ((raw_bits & 0xFFFF) << 14) + 0x20000000
  return struct.unpack('<' + 'f', struct.pack('<' + 'I', u))[0]

# ==============================================================================
# 3. Math & Rotation Helpers
# ==============================================================================
def fwht(x, off, bsize):
  step = 1
  while step < bsize:
    for i in range(0, bsize, 2 * step):
      for j in range(step):
        u, v = x[off + i + j], x[off + i + step + j]
        x[off + i + j], x[off + i + step + j] = u + v, u - v
    step <<= 1
  sc = 1.0 / math.sqrt(bsize)
  for i in range(bsize): x[off + i] *= sc

def apply_rot(x, dim, bsize=128, inv=False):
  if dim < bsize:
    bsize = 1
    while bsize * 2 <= dim: bsize *= 2
  if not inv:
    i = 0
    while i + bsize <= dim: fwht(x, i, bsize); i += bsize
    if i < dim: fwht(x, dim - bsize, bsize)
  else:
    fwd_i = 0
    while fwd_i + bsize <= dim: fwd_i += bsize
    if fwd_i < dim: fwht(x, dim - bsize, bsize)
    for j in range(fwd_i - bsize, -1, -bsize): fwht(x, j, bsize)

def bit_rev(j, bsize):
  res, b = 0, 1
  while b < bsize: res = (res << 1) | (j & 1); j >>= 1; b <<= 1
  return res

def cascaded_rot(x, dim, inv=False, single_pass_small=False):
  if single_pass_small and dim <= 256:
    if inv:
      apply_rot(x, dim, inv=True)
      for i in range(dim): x[i] *= FLIPS[i % 256]
    else:
      for i in range(dim): x[i] *= FLIPS[i % 256]
      apply_rot(x, dim)
    return

  bsize = 128
  if dim < 128:
    bsize = 1
    while bsize * 2 <= dim: bsize *= 2

  temp = [0.0] * dim
  if not inv:
    for j in range(dim): x[j] *= FLIPS[j % 256]
    apply_rot(x, dim, bsize)
    if dim > bsize:
      nblocks = dim // bsize
      for b in range(nblocks):
        for k in range(bsize):
          pk = PERM[k] if bsize == 128 else bit_rev(k, bsize)
          temp[k * nblocks + b] = x[b * bsize + pk]
      temp[nblocks * bsize:] = x[nblocks * bsize:]
      for j in range(dim): temp[j] *= FLIPS[(j + 127) % 256]
      apply_rot(temp, dim, bsize)
    else:
      for j in range(bsize):
        pk = PERM[j] if bsize == 128 else bit_rev(j, bsize)
        temp[j] = x[pk] * FLIPS[(j + 127) % 256]
      fwht(temp, 0, bsize)
  else:
    if dim > bsize:
      nblocks = dim // bsize
      apply_rot(x, dim, bsize, inv=True)
      for j in range(dim): x[j] *= FLIPS[(j + 127) % 256]
      for b in range(nblocks):
        for k in range(bsize):
          pk = PERM[k] if bsize == 128 else bit_rev(k, bsize)
          temp[b * bsize + pk] = x[k * nblocks + b]
      temp[nblocks * bsize:] = x[nblocks * bsize:]
      apply_rot(temp, dim, bsize, inv=True)
      for j in range(dim): temp[j] *= FLIPS[j % 256]
    else:
      fwht(temp, 0, bsize)
      for j in range(bsize): x[j] *= FLIPS[(j + 127) % 256]
      for j in range(bsize):
        ipk = INV_PERM[j] if bsize == 128 else bit_rev(j, bsize)
        temp[j] = x[ipk]
      fwht(temp, 0, bsize)
      for j in range(bsize): temp[j] *= FLIPS[j % 256]
  x[:dim] = temp

def find_nd(v, centroids):
  return min(range(len(centroids)), key=lambda c: sum((v[i] - centroids[c][i])**2 for i in range(len(v))))

def pack_3b(idx):
  return bytes([
      ((idx[0] & 7) | ((idx[1] & 7) << 3) | ((idx[2] & 7) << 6)) & 255,
      (((idx[2] & 7) >> 2) | ((idx[3] & 7) << 1) | ((idx[4] & 7) << 4) | ((idx[5] & 7) << 7)) & 255,
      (((idx[5] & 7) >> 1) | ((idx[6] & 7) << 2) | ((idx[7] & 7) << 5)) & 255
  ])

def unpack_3b(b):
  return [
      b[0] & 7, (b[0] >> 3) & 7, ((b[0] >> 6) & 3) | ((b[1] & 1) << 2),
      (b[1] >> 1) & 7, (b[1] >> 4) & 7, ((b[1] >> 7) & 1) | ((b[2] & 3) << 1),
      (b[2] >> 2) & 7, (b[2] >> 5) & 7
  ]

# ==============================================================================
# 4. Base Codec Class (Memory-Safe Standard & Residual Modes)
# ==============================================================================
class BaseRslmCodec:
  def __init__(self, dim, is_residual=False):
    if dim <= 0: raise ValueError(f"Invalid dim: {dim}")
    self.dim, self.is_residual = dim, is_residual
    n = 2.0 * dim; log_n = math.log(n); l = 2.0 * log_n; sqrt_l = math.sqrt(l)
    exp_max = sqrt_l - (math.log(log_n) + math.log(4.0 * math.pi) - 2 * 0.5772156649) / (2 * sqrt_l)
    self.inv_expected_max = 1.0 / exp_max

  @property
  def compressed_size(self):
    return self._raw_size() + (2 if self.is_residual else 0)

  def compress(self, input_vectors, original_vectors=None, chunk_size=50000):
    """Compresses vectors in memory-safe chunks."""
    if len(input_vectors) == 0:
      return b""
    is_np = isinstance(input_vectors, np.ndarray)
    nvec = len(input_vectors)
    out = bytearray()
    
    for start in range(0, nvec, chunk_size):
      end = min(start + chunk_size, nvec)
      chunk_in = input_vectors[start:end]
      if is_np:
        chunk_in = chunk_in.tolist()
      elif isinstance(chunk_in[0], (float, int)):
        chunk_in = [chunk_in]

      if self.is_residual:
        chunk_orig = None
        if original_vectors is not None:
          chunk_orig = original_vectors[start:end]
          if isinstance(chunk_orig, np.ndarray):
            chunk_orig = chunk_orig.tolist()
        for i, res_vec in enumerate(chunk_in):
          pk = self._compress_raw([res_vec])
          ratio = 1.0
          if chunk_orig is not None:
            orig = chunk_orig[i]
            r_rec = self._decompress_raw(pk, 1)[0]
            o_sq = sum(x * x for x in orig)
            v_sq = sum((orig[j] - res_vec[j] + r_rec[j])**2 for j in range(self.dim))
            ratio = math.sqrt(o_sq / v_sq) if v_sq > 1e-24 else (math.sqrt(o_sq) / 1e-12 if o_sq > 1e-24 else 1.0)
          out.extend(pk + ue7m9_enc(ratio))
      else:
        out.extend(self._compress_raw(chunk_in))
    return bytes(out)

  def decompress(self, packed_data, base_vectors=None, as_numpy=True):
    """Decompresses packed binary data directly into float32 array or list.
    
    If is_residual is True and base_vectors is provided, returns the full
    reconstructed vector: S_global * (base_vectors + r_raw).
    If base_vectors is None, returns the scaled residual: S_global * r_raw.
    """
    if not packed_data:
      return np.empty((0, self.dim), dtype=np.float32) if as_numpy else []
    rec_sz = self.compressed_size
    raw_sz = self._raw_size()
    nvec = len(packed_data) // rec_sz

    if not self.is_residual:
      raw_res = self._decompress_raw(packed_data, nvec)
      return np.array(raw_res, dtype=np.float32) if as_numpy else raw_res

    has_base = base_vectors is not None
    if as_numpy:
      out_arr = np.empty((nvec, self.dim), dtype=np.float32)
      for i in range(nvec):
        off = i * rec_sz
        r_vec = self._decompress_raw(packed_data[off:off+raw_sz], 1)[0]
        s = ue7m9_dec(packed_data[off+raw_sz:off+rec_sz])
        if has_base:
          b_vec = base_vectors[i]
          out_arr[i] = [(b_vec[j] + r_vec[j]) * s for j in range(self.dim)]
        else:
          out_arr[i] = [x * s for x in r_vec]
      return out_arr
    else:
      res = []
      for i in range(nvec):
        off = i * rec_sz
        r_vec = self._decompress_raw(packed_data[off:off+raw_sz], 1)[0]
        s = ue7m9_dec(packed_data[off+raw_sz:off+rec_sz])
        if has_base:
          b_vec = base_vectors[i]
          res.append([(b_vec[j] + r_vec[j]) * s for j in range(self.dim)])
        else:
          res.append([x * s for x in r_vec])
      return res

# ==============================================================================
# 5. Codec Implementations (Rslm4, Rslm4Lite, Rslm3, Rslm2, Rslm1)
# ==============================================================================
class Rslm4Codec(BaseRslmCodec):
  """Rslm4 (Rotated Scaled Lloyd-Max 4-bits) quantization codec."""
  def _raw_size(self): return 2 + (self.dim + 1) // 2

  def _compress_raw(self, vecs):
    idx_b, pdim = (self.dim + 1) // 2, self.dim // 2
    cents, mids = C1D[4]
    out = bytearray()
    for doc in vecs:
      r = list(doc); cascaded_rot(r, self.dim, single_pass_small=True)
      amax, l2_orig = max(abs(x) for x in r), sum(x * x for x in r)
      if amax == 0: out.extend(bytes(idx_b) + ue7m9_enc(0.0)); continue
      inv_s = 1.0 / (amax * self.inv_expected_max)
      codes = bytearray(idx_b); l2_q = 0.0
      for j in range(pdim):
        i1 = bisect.bisect_right(mids, r[2 * j] * inv_s)
        i2 = bisect.bisect_right(mids, r[2 * j + 1] * inv_s)
        codes[j] = (i1 << 4) | i2
        l2_q += cents[i1]**2 + cents[i2]**2
      if self.dim % 2:
        i1 = bisect.bisect_right(mids, r[self.dim - 1] * inv_s)
        codes[pdim] = i1 << 4; l2_q += cents[i1]**2
      cs = math.sqrt(l2_orig / l2_q) if l2_q > 1e-8 else 0.0
      out.extend(codes + ue7m9_enc(cs))
    return bytes(out)

  def _decompress_raw(self, pk, nvec):
    rsz, idx_b, pdim = self._raw_size(), (self.dim + 1) // 2, self.dim // 2
    cents = C1D[4][0]
    out = []
    for i in range(nvec):
      off = i * rsz
      sc = ue7m9_dec(pk[off+idx_b : off+rsz])
      doc = [0.0] * self.dim
      for j in range(pdim):
        b = pk[off + j]
        doc[2 * j], doc[2 * j + 1] = cents[b >> 4] * sc, cents[b & 15] * sc
      if self.dim % 2: doc[self.dim - 1] = cents[pk[off + pdim] >> 4] * sc
      cascaded_rot(doc, self.dim, inv=True, single_pass_small=True)
      out.append(doc)
    return out

class Rslm4LiteCodec(Rslm4Codec):
  """Rslm4Lite (Rotated Scaled Lloyd-Max 4-bits Lite) quantization codec."""
  def __init__(self, dim, is_residual=False):
    if is_residual: raise ValueError("Rslm4Lite does not support is_residual=True")
    super().__init__(dim, is_residual=False)

  def _raw_size(self): return (self.dim + 1) // 2 if self.dim >= 64 else (self.dim + 1) // 2 + 2

  def _compress_raw(self, vecs):
    if self.dim < 64: return super()._compress_raw(vecs)
    rsz, pdim = self._raw_size(), self.dim // 2
    c3, m3 = C1D[3]; c4, m4 = C1D[4]
    out = bytearray()
    for doc in vecs:
      r = list(doc); cascaded_rot(r, self.dim, single_pass_small=True)
      amax, l2_orig = max(abs(x) for x in r), sum(x * x for x in r)
      if amax == 0: out.extend(bytes(rsz)); continue
      inv_s = 1.0 / (amax * self.inv_expected_max)
      l2_q = 0.0; doc_b = bytearray(rsz)
      e16, o16 = [0]*16, [0]*16
      for j in range(16):
        i1 = bisect.bisect_right(m3, r[2*j] * inv_s)
        i2 = bisect.bisect_right(m4, r[2*j+1] * inv_s)
        e16[j], o16[j] = i1, i2
        l2_q += c3[i1]**2 + c4[i2]**2
      for j in range(16, pdim):
        i1 = bisect.bisect_right(m4, r[2*j] * inv_s)
        i2 = bisect.bisect_right(m4, r[2*j+1] * inv_s)
        doc_b[j] = (i1 << 4) | i2
        l2_q += c4[i1]**2 + c4[i2]**2
      if self.dim % 2:
        i1 = bisect.bisect_right(m4, r[self.dim-1] * inv_s)
        doc_b[pdim] = i1 << 4; l2_q += c4[i1]**2
      cs = math.sqrt(l2_orig / l2_q) if l2_q > 1e-8 else 0.0
      scale_bits = struct.unpack('<' + 'H', ue7m9_enc(cs))[0]
      for j in range(16):
        doc_b[j] = (((scale_bits >> j) & 1) << 7) | ((e16[j] & 7) << 4) | (o16[j] & 15)
      out.extend(doc_b)
    return bytes(out)

  def _decompress_raw(self, pk, nvec):
    if self.dim < 64: return super()._decompress_raw(pk, nvec)
    rsz, pdim = self._raw_size(), self.dim // 2
    c3, c4 = C1D[3][0], C1D[4][0]
    out = []
    for i in range(nvec):
      pdoc = pk[i * rsz : (i + 1) * rsz]
      scale_bits = sum(((pdoc[j] >> 7) & 1) << j for j in range(16))
      sc = ue7m9_dec(struct.pack('<' + 'H', scale_bits))
      doc = [0.0] * self.dim
      for j in range(16):
        b = pdoc[j]
        doc[2*j], doc[2*j+1] = c3[(b >> 4) & 7] * sc, c4[b & 15] * sc
      for j in range(16, pdim):
        b = pdoc[j]
        doc[2*j], doc[2*j+1] = c4[b >> 4] * sc, c4[b & 15] * sc
      if self.dim % 2: doc[self.dim-1] = c4[pdoc[pdim] >> 4] * sc
      cascaded_rot(doc, self.dim, inv=True, single_pass_small=True)
      out.append(doc)
    return out

class Rslm3Codec(BaseRslmCodec):
  """Rslm3 (Rotated Scaled Lloyd-Max 3-bits) quantization codec."""
  def _raw_size(self): return ((self.dim + 7) // 8) * 3 + 2

  def _compress_raw(self, vecs):
    idx_b, pdim = ((self.dim + 7) // 8) * 3, (self.dim + 7) & ~7
    cents, mids = C1D[3]
    out = bytearray()
    for doc in vecs:
      r = list(doc) + [0.0] * (pdim - len(doc))
      cascaded_rot(r, self.dim, single_pass_small=True)
      amax, l2_orig = max(abs(x) for x in r[:self.dim]), sum(x * x for x in r[:self.dim])
      inv_s = (1.0 / (amax * self.inv_expected_max)) if amax > 1e-8 else 0.0
      tidx = [bisect.bisect_right(mids, r[j] * inv_s) for j in range(self.dim)] + [0] * (pdim - self.dim)
      l2_q = sum(cents[tidx[j]]**2 for j in range(self.dim))
      cs = math.sqrt(l2_orig / l2_q) if l2_q > 1e-8 else 0.0
      pk_idx = bytearray()
      for j in range(pdim // 8): pk_idx.extend(pack_3b(tidx[j*8 : j*8+8]))
      out.extend(pk_idx + ue7m9_enc(cs))
    return bytes(out)

  def _decompress_raw(self, pk, nvec):
    rsz, idx_b, pdim = self._raw_size(), ((self.dim + 7) // 8) * 3, (self.dim + 7) & ~7
    cents = C1D[3][0]
    out = []
    for i in range(nvec):
      off = i * rsz
      sc = ue7m9_dec(pk[off+idx_b : off+rsz])
      pdoc = pk[off : off + idx_b]
      indices = []
      for j in range(pdim // 8): indices.extend(unpack_3b(pdoc[j*3 : j*3+3]))
      doc = [cents[indices[j]] * sc for j in range(self.dim)]
      cascaded_rot(doc, self.dim, inv=True, single_pass_small=True)
      out.append(doc)
    return out

class Rslm2Codec(BaseRslmCodec):
  """Rslm2 (Rotated Scaled Lloyd-Max 2D) quantization codec."""
  def _raw_size(self):
    npairs = ((self.dim + 1) & ~1) // 2
    return (npairs + 1) // 2 + 2

  def _compress_raw(self, vecs):
    pdim = (self.dim + 1) & ~1
    npairs = pdim // 2
    f2d_sz = (npairs + 1) // 2
    out = bytearray()
    for doc in vecs:
      r = list(doc) + [0.0] * (pdim - len(doc))
      cascaded_rot(r, self.dim, single_pass_small=True)
      amax, l2_orig = max(abs(x) for x in r[:self.dim]), sum(x * x for x in r[:self.dim])
      inv_s = (1.0 / (amax * self.inv_expected_max)) if amax > 1e-8 else 0.0
      codes = bytearray(f2d_sz); l2_q = 0.0
      for s in range(npairs // 2):
        i0 = find_nd([r[4*s] * inv_s, r[4*s+1] * inv_s], C2D)
        i1 = find_nd([r[4*s+2] * inv_s, r[4*s+3] * inv_s], C2D)
        codes[s] = (i0 << 4) | i1
        l2_q += C2D[i0][0]**2 + C2D[i0][1]**2 + C2D[i1][0]**2
        if 4*s + 3 < self.dim: l2_q += C2D[i1][1]**2
      if npairs % 2:
        s_odd = npairs - 1
        i0 = find_nd([r[2*s_odd] * inv_s, r[2*s_odd+1] * inv_s], C2D)
        codes[npairs // 2] = i0 << 4
        l2_q += C2D[i0][0]**2
        if self.dim % 2 == 0: l2_q += C2D[i0][1]**2
      cs = math.sqrt(l2_orig / l2_q) if l2_q > 1e-8 else 0.0
      out.extend(codes + ue7m9_enc(cs))
    return bytes(out)

  def _decompress_raw(self, pk, nvec):
    pdim = (self.dim + 1) & ~1
    npairs = pdim // 2
    f2d_sz, rsz = (npairs + 1) // 2, self._raw_size()
    out = []
    for i in range(nvec):
      off = i * rsz
      codes = pk[off : off + f2d_sz]
      sc = ue7m9_dec(pk[off+f2d_sz : off+rsz])
      temp = [0.0] * pdim
      for s in range(npairs // 2):
        b = codes[s]; i0, i1 = b >> 4, b & 15
        temp[4*s] = C2D[i0][0] * sc; temp[4*s+1] = C2D[i0][1] * sc
        temp[4*s+2] = C2D[i1][0] * sc; temp[4*s+3] = C2D[i1][1] * sc
      if npairs % 2:
        b = codes[npairs // 2]; i0 = b >> 4
        temp[2*(npairs-1)] = C2D[i0][0] * sc; temp[2*(npairs-1)+1] = C2D[i0][1] * sc
      cascaded_rot(temp, self.dim, inv=True, single_pass_small=True)
      out.append(temp[:self.dim])
    return out

class Rslm1Codec(BaseRslmCodec):
  """Rslm1 (Rotated Scaled Lloyd-Max 1D/4D) quantization codec."""
  def _raw_size(self):
    nquads = ((self.dim + 3) & ~3) // 4
    return (nquads + 1) // 2 + 2

  def _compress_raw(self, vecs):
    pdim = (self.dim + 3) & ~3
    nquads = pdim // 4
    f4d_sz = (nquads + 1) // 2
    out = bytearray()
    for doc in vecs:
      r = list(doc) + [0.0] * (pdim - len(doc))
      cascaded_rot(r, self.dim, single_pass_small=True)
      amax, l2_orig = max(abs(x) for x in r[:self.dim]), sum(x * x for x in r[:self.dim])
      inv_s = (1.0 / (amax * self.inv_expected_max)) if amax > 1e-8 else 0.0
      codes = bytearray(f4d_sz); l2_q = 0.0
      for s in range(nquads // 2):
        i0 = find_nd([r[8*s + k] * inv_s for k in range(4)], C4D)
        i1 = find_nd([r[8*s + 4 + k] * inv_s for k in range(4)], C4D)
        codes[s] = (i0 << 4) | i1
        l2_q += sum(c**2 for c in C4D[i0]) + C4D[i1][0]**2
        for k in range(1, 4):
          if 8*s + 4 + k < self.dim: l2_q += C4D[i1][k]**2
      if nquads % 2:
        s_odd = nquads - 1
        i0 = find_nd([r[4*s_odd + k] * inv_s for k in range(4)], C4D)
        codes[nquads // 2] = i0 << 4
        l2_q += C4D[i0][0]**2
        for k in range(1, 4):
          if 4*s_odd + k < self.dim: l2_q += C4D[i0][k]**2
      cs = math.sqrt(l2_orig / l2_q) if l2_q > 1e-8 else 0.0
      out.extend(codes + ue7m9_enc(cs))
    return bytes(out)

  def _decompress_raw(self, pk, nvec):
    pdim = (self.dim + 3) & ~3
    nquads = pdim // 4
    f4d_sz, rsz = (nquads + 1) // 2, self._raw_size()
    out = []
    for i in range(nvec):
      off = i * rsz
      codes = pk[off : off + f4d_sz]
      sc = ue7m9_dec(pk[off+f4d_sz : off+rsz])
      temp = [0.0] * pdim
      for s in range(nquads // 2):
        b = codes[s]; i0, i1 = b >> 4, b & 15
        for k in range(4): temp[8*s + k] = C4D[i0][k] * sc
        for k in range(4): temp[8*s + 4 + k] = C4D[i1][k] * sc
      if nquads % 2:
        b = codes[nquads // 2]; i0 = b >> 4
        for k in range(4): temp[4*(nquads-1) + k] = C4D[i0][k] * sc
      cascaded_rot(temp, self.dim, inv=True, single_pass_small=True)
      out.append(temp[:self.dim])
    return out

CODEC_MAP = {"rslm4": Rslm4Codec, "rslm4lite": Rslm4LiteCodec, "rslm3": Rslm3Codec, "rslm2": Rslm2Codec, "rslm1": Rslm1Codec}

def create_rslm_codec(codec_type: str, dim: int, is_residual: bool = False) -> BaseRslmCodec:
  """Factory function to instantiate an RSLM codec."""
  k = codec_type.lower().strip()
  if k not in CODEC_MAP:
    raise ValueError(f"Unknown codec '{codec_type}'. Choose from: {list(CODEC_MAP.keys())}")
  return CODEC_MAP[k](dim, is_residual=is_residual)

print("RSLM library initialized successfully!")