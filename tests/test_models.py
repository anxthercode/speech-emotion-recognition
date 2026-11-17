"""Independent numerical and checkpoint checks for the three training methods."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from ser.cnn import EmotionCNN
from ser.features import DEFAULT_AUDIO
from ser.gmm import DiagonalGMM, GMMClassifier, logsumexp
from ser.train_cnn import seed_everything, train_cnn
from ser.training_common import metrics
from ser.training_common import write_json as actual_write_json


class GMMTests(unittest.TestCase):
    def test_log_density_against_gaussian_formula_and_logsumexp(self):
        model = DiagonalGMM(n_components=1)
        model.weights_ = np.array([1.])
        model.means_ = np.array([[1., -2.]])
        model.variances_ = np.array([[4., .5]])
        x = np.array([[0., 0.], [1., -2.], [100., -100.]])
        expected = -.5 * (2 * np.log(2 * np.pi) + np.log(2) + ((x - model.means_[0]) ** 2 / model.variances_[0]).sum(axis=1))
        np.testing.assert_allclose(model.score_samples(x), expected, atol=1e-12)
        self.assertAlmostEqual(float(logsumexp(np.array([-10000., -10000.]))), -10000 + np.log(2))
        np.testing.assert_allclose(model.responsibilities(x).sum(axis=1), 1)

    def test_em_recovers_synthetic_mixture_and_matches_independent_reference(self):
        rng = np.random.default_rng(6)
        x = np.concatenate([rng.normal([-4, 1], [.5, .8], (600, 2)), rng.normal([4, -1], [.6, .4], (400, 2))])
        with threadpool_limits(2):
            model = DiagonalGMM(2, variance_floor=1e-6, max_iter=100, tol=1e-8, seed=9).fit(x)
            reference = GaussianMixture(2, covariance_type="diag", reg_covar=1e-6, n_init=2, random_state=9).fit(x)
        np.testing.assert_allclose(model.means_[np.argsort(model.means_[:, 0])], [[-4, 1], [4, -1]], atol=.12)
        np.testing.assert_allclose(model.score_samples(x), reference.score_samples(x), atol=1e-4)
        self.assertAlmostEqual(float(model.weights_.sum()), 1)
        self.assertTrue(np.all(model.variances_ > 0))
        for initialization in model.initializations_:
            self.assertTrue(np.all(np.diff([row["mean_log_likelihood"] for row in initialization["history"]]) >= -1e-7))

    def test_variance_floor_handles_constant_features_and_roundtrip(self):
        sequences = [np.full((20, 3), label, dtype=float) for label in range(6)]
        classifier = GMMClassifier(n_components=1, max_iter=10, n_init=1, seed=5).fit(sequences, list(range(6)))
        scores = classifier.decision_function(sequences)
        np.testing.assert_array_equal(scores.argmax(axis=1), np.arange(6))
        buffer = io.BytesIO()
        classifier.save(buffer)
        buffer.seek(0)
        restored = GMMClassifier.load(buffer)
        np.testing.assert_allclose(scores, restored.decision_function(sequences), rtol=0, atol=0)


class CNNTests(unittest.TestCase):
    def test_logits_gradients_and_state_roundtrip(self):
        torch.set_num_threads(2)
        seed_everything(17)
        model = EmotionCNN(.2)
        x = torch.randn(3, 1, 64, 397)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (3, 6))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2, 5]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.inference_mode():
            expected = model(x)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        other = EmotionCNN(.2).eval()
        other.load_state_dict(torch.load(stream, weights_only=True))
        with torch.inference_mode():
            torch.testing.assert_close(expected, other(x), rtol=0, atol=0)

    def test_small_training_set_can_be_memorized(self):
        torch.set_num_threads(2)
        seed_everything(8)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EmotionCNN(0).to(device)
        # Four artificial examples, unrelated to validation or test recordings.
        x = torch.randn(4, 1, 32, 32, device=device)
        y = torch.arange(4, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x).argmax(1), y))


class MetricTests(unittest.TestCase):
    def test_missing_predicted_class_still_counts_in_macro_average(self):
        result = metrics(np.arange(6), np.zeros(6, dtype=int))
        self.assertAlmostEqual(result["uar"], 1 / 6)
        self.assertAlmostEqual(result["macro_f1"], (2 / 7) / 6)


class ResumeTests(unittest.TestCase):
    def test_resumed_training_matches_uninterrupted_weights(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "splits").mkdir()
            (root / "splits" / "speaker.csv").write_text("synthetic test split", encoding="utf-8")
            generator = torch.Generator().manual_seed(91)
            data = SimpleNamespace(
                x=torch.randn(12, 1, 16, 24, generator=generator),
                labels=np.tile(np.arange(6), 2), targets=torch.arange(12) % 6,
                owners=np.arange(12),
                records=[{"file_id": str(i), "speaker_id": "unit", "label_id": str(i % 6)} for i in range(12)],
                dataset=SimpleNamespace(root=root, split="speaker", report={"fingerprint": "unit"},
                                        config=DEFAULT_AUDIO, normalization={}),
                random_crops=lambda seed, epoch: None,
            )
            parameters = {"dropout": .2, "learning_rate": .001, "weight_decay": .0001,
                          "amp": False, "max_epochs": 3, "patience": 3, "batch_size": 6}
            train_cnn(data, data, parameters, 17, root / "uninterrupted")
            def interrupt_after_checkpoint(path, value):
                actual_write_json(path, value)
                if Path(path).name == "history.json":
                    raise KeyboardInterrupt()
            with patch("ser.train_cnn.write_json", side_effect=interrupt_after_checkpoint):
                with self.assertRaises(KeyboardInterrupt):
                    train_cnn(data, data, parameters, 17, root / "resumed")
            train_cnn(data, data, parameters, 17, root / "resumed")
            uninterrupted = torch.load(root / "uninterrupted" / "last.pt", weights_only=True)
            resumed = torch.load(root / "resumed" / "last.pt", weights_only=True)
            self.assertEqual(len(resumed["history"]), 3)
            for name in uninterrupted["state_dict"]:
                torch.testing.assert_close(uninterrupted["state_dict"][name], resumed["state_dict"][name], rtol=0, atol=0)
            self.assertEqual(uninterrupted["best_metrics"], resumed["best_metrics"])


if __name__ == "__main__":
    unittest.main()
