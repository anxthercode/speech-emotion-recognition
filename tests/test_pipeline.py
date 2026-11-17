"""Numerical checks and protections against train/test leakage."""
import unittest

import numpy as np

from qt_gui.audio import log_mel_preview
from ser.features import (AudioConfig, delta, evaluation_starts, extract_features,
                          log_mel_frames, log_mel_window, mfcc_features, mono_resample)
from ser.noise import measured_snr, mix_at_snr, noise_segment, stable_seed
from ser.splits import LABELS, make_splits, validate_split
from ser.statistics import fit_training_statistics


def toy_records():
    return [{"file_id": f"{actor}_{emotion}_{i}", "speaker_id": str(actor),
             "label_id": str(label), "emotion": emotion, "sha256": f"hash_{actor}_{emotion}_{i}"}
            for actor in range(12) for emotion, label in LABELS.items() for i in range(3 + label)]


class SplitTests(unittest.TestCase):
    def test_complete_disjoint_and_reproducible_splits_with_matched_sizes(self):
        records = toy_records()
        splits = make_splits(records, validation_speakers=2, test_speakers=2)
        self.assertEqual(splits, make_splits(records[::-1], validation_speakers=2, test_speakers=2))
        for subset in ("train", "validation", "test"):
            speaker = [r for r in splits["speaker"] if r["subset"] == subset]
            random = [r for r in splits["random"] if r["subset"] == subset]
            self.assertEqual(len(speaker), len(random))
            for label in LABELS.values():
                expected = len(random) * sum(int(r["label_id"]) == label for r in records) / len(records)
                self.assertLessEqual(abs(sum(int(r["label_id"]) == label for r in random) - expected), 2)
        self.assertNotEqual(splits["speaker"], make_splits(records, seed=7, validation_speakers=2, test_speakers=2)["speaker"])

    def test_rejects_missing_recording_duplicate_hash_and_actor_leak(self):
        records = toy_records()
        splits = make_splits(records, validation_speakers=2, test_speakers=2)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_split(records, splits["speaker"][:-1], grouped=True)
        bad = [dict(row) for row in splits["speaker"]]
        bad[0]["subset"] = "test" if bad[0]["subset"] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "speaker_id leakage"):
            validate_split(records, bad, grouped=True)
        copied = [dict(row) for row in records]
        left = next(row for row in splits["speaker"] if row["subset"] == "train")["file_id"]
        right = next(row for row in splits["speaker"] if row["subset"] == "test")["file_id"]
        for rows in (copied, splits["speaker"]):
            for row in rows:
                if row["file_id"] in (left, right):
                    row["sha256"] = "same_audio"
        with self.assertRaisesRegex(ValueError, "sha256 leakage"):
            validate_split(copied, splits["speaker"], grouped=True)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.audio = (.15 * np.sin(2 * np.pi * 700 * np.arange(32000) / 16000)).astype(np.float32)

    def test_full_recording_preview_uses_same_spectrum_without_cnn_padding(self):
        preview = log_mel_preview(self.audio)
        self.assertEqual(preview.shape, (64, 197))
        np.testing.assert_allclose(preview, log_mel_window(self.audio)[:, :preview.shape[1]], atol=1e-5)
        self.assertEqual(log_mel_window(self.audio).shape, (64, 397))

    def test_mfcc_has_no_artificial_four_second_padding(self):
        result = extract_features(self.audio)
        self.assertEqual(result["mfcc"].shape, (197, 39))
        self.assertEqual(result["svm"].shape, (78,))
        # C0 of an orthonormal DCT-II is sum(log-Mel)/sqrt(number of bands).
        expected_c0 = log_mel_frames(self.audio).sum(axis=0) / np.sqrt(64)
        np.testing.assert_allclose(result["mfcc"][:, 0], expected_c0, atol=1e-4)
        np.testing.assert_allclose(result["svm"][:39], result["mfcc"].mean(axis=0))

    def test_tail_window_preserves_end_and_delta_responds_to_slope(self):
        starts = evaluation_starts(80080)
        np.testing.assert_array_equal(starts, [0, 16080])
        self.assertEqual(starts[-1] + 64000, 80080)
        np.testing.assert_array_equal(evaluation_starts(120), [0])
        ramp = np.arange(20, dtype=np.float32)[:, None]
        np.testing.assert_allclose(delta(ramp)[2:-2], 1)
        np.testing.assert_allclose(delta(np.ones((10, 13))), 0)

    def test_audio_input_validation_and_resampling(self):
        stereo = np.repeat(self.audio[:, None], 2, axis=1)
        np.testing.assert_array_equal(mono_resample(stereo, 16000), self.audio)
        self.assertEqual(len(mono_resample(self.audio, 32000)), 16000)
        self.assertTrue(np.isfinite(mfcc_features(np.zeros(100))).all())
        with self.assertRaises(ValueError):
            mono_resample(np.array([np.nan]), 16000)
        with self.assertRaises(ValueError):
            AudioConfig(fmax=9000)


class NoiseTests(unittest.TestCase):
    def test_snr_reproducibility_and_same_noise_direction(self):
        clean = np.random.default_rng(0).normal(0, .1, 16000).astype(np.float32)
        seed = stable_seed("clip", "white", 2026)
        noise, _ = noise_segment(len(clean), seed)
        np.testing.assert_array_equal(noise, noise_segment(len(clean), seed)[0])
        residuals = []
        for snr in (20, 10, 0):
            mixed, alpha, actual = mix_at_snr(clean, noise, snr)
            self.assertAlmostEqual(actual, snr, places=5)
            residuals.append((mixed.astype(np.float64) - clean) / alpha)
        np.testing.assert_allclose(residuals[0], residuals[-1], atol=2e-6)
        self.assertNotEqual(seed, stable_seed("other_clip", "white", 2026))

    def test_no_clipping_and_background_offset(self):
        mixed, _, _ = mix_at_snr(np.ones(200), np.ones(200), 0)
        self.assertGreater(mixed.max(), 1)
        self.assertAlmostEqual(measured_snr(np.ones(200), mixed), 0)
        background = np.arange(1000, dtype=np.float32)
        noise, offset = noise_segment(200, 17, background)
        np.testing.assert_array_equal(noise, background[offset:offset + 200])
        with self.assertRaises(ValueError):
            noise_segment(2000, 17, background)
        for clean, noise in ((np.zeros(10), np.ones(10)), (np.ones(10), np.zeros(10))):
            with self.assertRaises(ValueError):
                mix_at_snr(clean, noise, 10)


class StatisticsTests(unittest.TestCase):
    def test_only_training_is_read_and_variance_matches_direct_calculation(self):
        rows = [{"file_id": "a", "subset": "train"}, {"file_id": "test", "subset": "test"},
                {"file_id": "b", "subset": "train"}, {"file_id": "val", "subset": "validation"}]
        seen = []
        def load(file_id):
            seen.append(file_id)
            if file_id not in ("a", "b"):
                raise AssertionError("Read validation or test while fitting")
            number = 2 if file_id == "a" else 4
            return {"log_mel": np.full((2, 64, 397), number), "mfcc": np.full((10, 39), number),
                    "svm": np.full(78, number)}
        stats = fit_training_statistics(rows, load)
        self.assertEqual(seen, ["a", "b"])
        for name in ("log_mel", "mfcc", "svm"):
            np.testing.assert_allclose(stats[name]["mean"], 3)
            np.testing.assert_allclose(stats[name]["std"], 1)
        self.assertEqual(stats["log_mel"]["count"], 2 * 397)


if __name__ == "__main__":
    unittest.main()
