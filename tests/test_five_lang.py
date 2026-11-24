"""Numerical regression checks for the optional five-language packaging tools."""
import unittest

import numpy as np

try:
    import gguf
except ImportError:
    gguf = None


@unittest.skipUnless(gguf and hasattr(gguf.GGMLQuantizationType, "NVFP4"), "requires converter gguf>=0.19")
class ExpertPackingTests(unittest.TestCase):
    def test_signed_codes_block_scales_and_row_order(self):
        from tools.prepare_five_lang import compact_int3, repack_int3, Q
        rng = np.random.default_rng(415)
        signs = rng.integers(0, 2, size=(3, 32, 16), dtype=np.uint8)
        magnitude = rng.integers(0, 4, size=signs.shape, dtype=np.uint8)
        planes = np.concatenate([np.packbits(bits, axis=-1, bitorder="little")
                                 for bits in (signs, magnitude >> 1, magnitude & 1)], axis=-1).reshape(3, -1)
        # E4M3 bytes for 1, 1.125, ... 1.875. All fit on a single Q3_K scale grid.
        scales = rng.integers(56, 64, size=(3, 32), dtype=np.uint8)
        expected = ((1 - 2 * signs.astype(np.float32)) * magnitude * 2 *
                    (1 + (scales & 7).astype(np.float32)[:, :, None] / 8)).reshape(3, -1)
        reference = repack_int3(planes, scales).reshape(-1, 36)
        codes = reference[:, 4:].reshape(-1, 4, 8)
        table = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        reconstructed = np.concatenate([table[codes & 15], table[codes >> 4]], axis=-1)
        reconstructed *= 1 + (reference[:, :4] & 7)[:, :, None] / 8
        np.testing.assert_array_equal(reconstructed.reshape(3, -1), expected)
        compact = compact_int3(planes, scales, 1 / 1024)
        np.testing.assert_array_equal(gguf.dequantize(compact, Q.Q3_K), expected / 1024)
