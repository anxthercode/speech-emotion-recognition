"""Run with: python -m unittest discover -s tests -v"""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import soundfile as sf
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QAudioBufferOutput, QMediaPlayer

from qt_gui.audio import load_preview, log_mel_preview, mel_filterbank
from qt_gui.main import MainWindow, configure_application


class AudioTests(unittest.TestCase):
    def test_silence_is_finite_with_expected_shape(self):
        result = log_mel_preview(np.zeros(100))
        self.assertEqual(result.shape, (64, 1))
        np.testing.assert_allclose(result, -100)

    def test_filterbank_and_tone_frequency(self):
        bank = mel_filterbank()
        self.assertEqual(bank.shape, (64, 257))
        self.assertTrue(np.all(bank >= 0))
        self.assertTrue(np.all(bank.sum(axis=1) > 0))
        tone = .5*np.sin(2*np.pi*1000*np.arange(64000)/16000)
        result = log_mel_preview(tone)
        peak = int(result.mean(axis=1).argmax())
        centers = 700*(10**(np.linspace(2595*np.log10(1+50/700), 2595*np.log10(1+7600/700), 66)[1:-1]/2595)-1)
        self.assertLess(abs(centers[peak]-1000), 100)

    def test_stereo_resampling_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            t = np.arange(44100)/44100
            source = np.column_stack([.2*np.sin(2*np.pi*440*t), .2*np.sin(2*np.pi*440*t)])
            path = Path(folder)/"stereo.wav"
            sf.write(path, source, 44100)
            result = load_preview(str(path))
            self.assertEqual((result.sample_rate, result.channels), (44100, 2))
            self.assertAlmostEqual(result.duration_s, 1)
            self.assertEqual(result.log_mel.shape, (64, 97))
            self.assertTrue(np.isfinite(result.log_mel).all())
            self.assertIsNone(json.loads(json.dumps(result.metadata()))["prediction"])

    def test_invalid_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "WAV"):
                load_preview(str(Path(folder)/"audio.txt"))
            path = Path(folder)/"empty.wav"
            path.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:example\nsize 12345\n")
            with self.assertRaisesRegex(ValueError, "Git LFS"):
                load_preview(str(path))
            sf.write(path, np.zeros(0), 16000)
            with self.assertRaises(ValueError):
                load_preview(str(path))
            sf.write(path, np.array([np.nan, 0], dtype=np.float32), 16000, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "NaN"):
                load_preview(str(path))


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_application(cls.app)

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.wait_for_worker()
        self.window.close()
        QTest.qWait(100)
        self.folder.cleanup()

    def wait_for_worker(self):
        for _ in range(500):
            if self.window.thread is None:
                return
            QTest.qWait(10)
        self.fail("Audio worker did not finish")

    def test_empty_state_and_real_async_preview(self):
        self.assertFalse(self.window.recognize_button.isEnabled())
        self.assertFalse(self.window.play_button.isEnabled())
        source = Path(self.folder.name)/"tone.wav"
        sf.write(source, .2*np.sin(2*np.pi*440*np.arange(16000)/16000), 16000)
        self.window.load_file(str(source))
        self.wait_for_worker()
        for _ in range(100):
            if self.window.play_button.isEnabled():
                break
            QTest.qWait(20)
        self.assertIsNotNone(self.window.preview)
        self.assertTrue(self.window.play_button.isEnabled())
        self.assertTrue(self.window.export_action.isEnabled())
        self.assertEqual(self.window.recognize_button.isEnabled(), bool(self.window.runs))
        self.assertEqual(self.window.results_table.item(0, 1).text(), "—")
        self.window.update_position(500)
        self.assertEqual(self.window.wave_plot.position, .5)
        self.assertFalse(self.window.grab().isNull())
        self.window.load_file(str(Path(self.folder.name)/"missing.wav"))
        self.wait_for_worker()
        self.assertTrue(self.window.error_label.isVisible())
        self.assertEqual(self.window.preview.path, str(source.resolve()))

    def test_mp3_drop_and_waveform_seeking(self):
        source = Path(self.folder.name) / "voice.mp3"
        sf.write(source, .2 * np.sin(2 * np.pi * 440 * np.arange(32000) / 16000),
                 16000, format="MP3", subtype="MPEG_LAYER_III")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(source))])
        args = (Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        enter = QDragEnterEvent(QPoint(100, 100), *args)
        QApplication.sendEvent(self.window, enter)
        self.assertTrue(enter.isAccepted())
        drop = QDropEvent(QPointF(100, 100), *args)
        QApplication.sendEvent(self.window, drop)
        self.assertTrue(drop.isAccepted())
        self.wait_for_worker()
        self.assertEqual(self.window.preview.path, str(source.resolve()))
        self.assertEqual(self.window.format_label.text(), "MP3")
        for _ in range(100):
            if self.window.player.isSeekable():
                break
            QTest.qWait(20)
        wave = self.window.wave_plot
        self.assertTrue(wave.seekable)
        rect = wave.plot_rect()
        QTest.mouseClick(wave, Qt.MouseButton.LeftButton, pos=rect.center().toPoint())
        self.assertAlmostEqual(self.window.player.position() / 1000, wave.duration / 2, delta=.04)
        self.assertAlmostEqual(wave.position, self.window.player.position() / 1000, delta=.001)

        # Let the first file finish before dropping another recording into the same player.
        buffers = []
        output = QAudioBufferOutput(self.window)
        output.audioBufferReceived.connect(lambda buffer: buffers.append(buffer.sampleCount()))
        self.window.player.setAudioOutput(None)
        self.window.player.setAudioBufferOutput(output)
        self.window.play_button.click()
        for _ in range(200):
            if self.window.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
                break
            QTest.qWait(20)
        self.assertEqual(self.window.player.mediaStatus(), QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertTrue(any(buffers))
        buffers.clear()
        second = Path(self.folder.name) / "second.wav"
        sf.write(second, .1 * np.sin(2 * np.pi * 650 * np.arange(24000) / 16000), 16000)
        mime.setUrls([QUrl.fromLocalFile(str(second))])
        QApplication.sendEvent(self.window, QDragEnterEvent(QPoint(100, 100), *args))
        drop = QDropEvent(QPointF(100, 100), *args)
        QApplication.sendEvent(self.window, drop)
        self.assertTrue(drop.isAccepted())
        self.wait_for_worker()
        for _ in range(100):
            if self.window.play_button.isEnabled():
                break
            QTest.qWait(20)
        self.assertTrue(Path(self.window.player.source().toLocalFile()).samefile(second))
        self.assertEqual(self.window.player.position(), 0)
        self.window.play_button.click()
        for _ in range(100):
            if any(buffers):
                break
            QTest.qWait(20)
        self.assertTrue(any(buffers))

    def test_opening_details_keeps_plots_in_place(self):
        self.window.resize(1040, 760)
        QTest.qWait(50)
        before = [plot.geometry() for plot in (self.window.wave_plot, self.window.mel_plot)]
        self.window.details_button.click()
        self.window.apply_prediction({"clean": {"emotion": "ANG", "model": "cnn",
                                     "scores": dict.fromkeys(("ANG", "DIS", "FEA", "HAP", "NEU", "SAD"), .125),
                                     "score_type": "softmax_probability"}})
        QTest.qWait(50)
        self.assertEqual(before, [plot.geometry() for plot in (self.window.wave_plot, self.window.mel_plot)])
        self.assertEqual(self.window.predicted_label.text(), "Гнев")
        self.assertEqual(self.window.results_table.item(0, 1).text(), "12.5%")

    def test_csv_validation_and_split_selection(self):
        path = Path(self.folder.name)/"metrics.csv"
        header = "split,condition,snr_db,macro_f1,uar,accuracy\n"
        path.write_text(header+"speaker,clean,,0.4,0.5,0.6\nrandom,white,20,0.7,0.8,0.9\n", encoding="utf-8")
        self.window.read_metrics(path)
        self.assertEqual(self.window.metrics_table.item(0, 2).text(), "0.400")
        self.window.split_combo.setCurrentIndex(1)
        self.assertEqual(self.window.metrics_table.item(1, 2).text(), "0.700")
        old = self.window.loaded_metrics.copy()
        for invalid in ["speaker,clean,0,0.4,0.5,0.6\n", "speaker,clean,,nan,0.5,0.6\n", "speaker,clean,,40,0.5,0.6\n"]:
            path.write_text(header+invalid, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.window.read_metrics(path)
            self.assertEqual(self.window.loaded_metrics, old)


if __name__ == "__main__":
    unittest.main()
