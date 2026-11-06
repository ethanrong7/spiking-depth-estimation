"""Tiny synthetic CPU checks of prospective input controls; no experiment."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from input_codes import transform_input, transform_stereo


class InputCodesTest(unittest.TestCase):
    def test_every_count_zero_through_fifty_and_exact_integer_sequence(self):
        counts = torch.arange(51).reshape(1, 1, 1, 51)
        raw = (torch.arange(50).reshape(1, 50, 1, 1) < counts).to(torch.float32)
        encoded = transform_input(raw, "count_only")
        torch.testing.assert_close(encoded.sum(1, keepdim=True), counts.float())
        for count in range(51):
            expected = torch.tensor(
                [((t + 1) * count) // 50 - (t * count) // 50 for t in range(50)],
                dtype=raw.dtype,
            )
            torch.testing.assert_close(encoded[0, :, 0, count], expected)
        self.assertEqual(encoded[..., 0].count_nonzero().item(), 0)
        self.assertTrue(torch.all(encoded[..., 50] == 1))

    def test_random_stereo_preserves_counts_shape_range_dtype_device_and_source(self):
        generator = torch.Generator().manual_seed(731)
        left = torch.randint(0, 2, (2, 50, 13, 17), generator=generator)
        right = torch.randint(0, 2, left.shape, generator=generator)
        for dtype in (torch.bool, torch.uint8, torch.int16, torch.int32, torch.int64,
                      torch.float16, torch.bfloat16, torch.float32, torch.float64):
            a, b = left.to(dtype), right.to(dtype)
            originals = (a.clone(), b.clone())
            outputs = transform_stereo(a, b, "count_only")
            for source, original, output in zip((a, b), originals, outputs):
                self.assertEqual(output.shape, source.shape)
                self.assertEqual(output.dtype, source.dtype)
                self.assertEqual(output.device, source.device)
                self.assertTrue(torch.all((output == 0) | (output == 1)))
                torch.testing.assert_close(output.sum(1, dtype=torch.int64),
                                           source.sum(1, dtype=torch.int64))
                torch.testing.assert_close(source, original)

    def test_raw_identity_preserves_object_and_gradient_path(self):
        raw = torch.zeros(1, 50, 2, 3, requires_grad=True)
        other = torch.ones_like(raw, requires_grad=True)
        self.assertIs(transform_input(raw, "raw"), raw)
        left, right = transform_stereo(raw, other, "raw")
        self.assertIs(left, raw)
        self.assertIs(right, other)
        (left.sum() + right.sum()).backward()
        torch.testing.assert_close(raw.grad, torch.ones_like(raw))
        torch.testing.assert_close(other.grad, torch.ones_like(other))

    def test_count_only_discards_timing_and_handles_noncontiguous_input(self):
        generator = torch.Generator().manual_seed(211)
        raw = torch.randint(0, 2, (1, 50, 3, 5), generator=generator).float()
        self.assertFalse(torch.equal(raw, raw.flip(1)))
        torch.testing.assert_close(transform_input(raw, "count_only"),
                                   transform_input(raw.flip(1), "count_only"))
        transposed = raw.transpose(2, 3)
        self.assertFalse(transposed.is_contiguous())
        torch.testing.assert_close(transform_input(transposed, "count_only"),
                                   transform_input(raw, "count_only").transpose(2, 3))

    def test_rejects_truncation_empty_shapes_and_nonbinary_values_in_both_modes(self):
        invalid_shapes = [(50, 2, 3), (1, 49, 2, 3), (1, 51, 2, 3),
                          (0, 50, 2, 3), (1, 50, 0, 3), (1, 50, 2, 0)]
        for mode in ("raw", "count_only"):
            for shape in invalid_shapes:
                with self.subTest(mode=mode, shape=shape), self.assertRaises(ValueError):
                    transform_input(torch.zeros(shape), mode)
            for bad_value in (-1., .5, 2., float("nan"), float("inf")):
                raw = torch.zeros(1, 50, 2, 3)
                raw[0, 1, 0, 0] = bad_value
                with self.subTest(mode=mode, value=bad_value), self.assertRaises(ValueError):
                    transform_input(raw, mode)

    def test_normalization_boundary_and_stereo_contract(self):
        raw = torch.zeros(1, 50, 2, 3)
        raw[:, :17] = 1
        encoded = transform_input(raw, "count_only")
        self.assertTrue(torch.all(((2 * encoded - 1) == -1) | ((2 * encoded - 1) == 1)))
        for mode in ("raw", "count_only"):
            with self.assertRaisesRegex(ValueError, "before 2\\*S-1"):
                transform_input(2 * raw - 1, mode)
            with self.assertRaisesRegex(ValueError, "shapes must match"):
                transform_stereo(raw, torch.zeros(1, 50, 2, 4), mode)
            with self.assertRaisesRegex(ValueError, "dtype and device"):
                transform_stereo(raw, raw.double(), mode)
        with self.assertRaises(ValueError):
            transform_input(raw, "unknown")
        with self.assertRaises(TypeError):
            transform_input([0, 1], "raw")
        with self.assertRaises(TypeError):
            transform_input(raw.to(torch.complex64), "count_only")


if __name__ == "__main__":
    unittest.main()
