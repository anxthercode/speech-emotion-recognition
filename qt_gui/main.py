from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot, Qt
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QFontDatabase, QKeySequence, QPalette
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QFileDialog, QFormLayout,
    QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QLayout, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .audio import AUDIO_EXTENSIONS, AUDIO_FILTER, AUDIO_FORMATS, AudioPreview, load_preview
from .plots import SignalPlot
from ser.inference import MODELS, PROJECT, Predictor, demo_run, final_runs

CLASS_NAMES = [("ANG", "Гнев"), ("DIS", "Отвращение"), ("FEA", "Страх"),
               ("HAP", "Радость"), ("NEU", "Нейтрально"), ("SAD", "Грусть")]
CONDITIONS = [("clean", "Чистая запись", None)] + [
    (kind, name, snr) for kind, name in [("white", "Белый шум"), ("background", "Шум кондиционера")]
    for snr in [20, 10, 0]
]
CSV_FIELDS = ["model", "split", "condition", "snr_db", "n_test", "n_seeds", "macro_f1", "uar", "accuracy"]
EMOTION_COLORS = {"ANG": "#ff9090", "DIS": "#b6cf85", "FEA": "#c7b0fc",
                  "HAP": "#f7cb71", "NEU": "#b4bdcd", "SAD": "#85bff4"}
SCORE_HELP = {
    "softmax_probability": "Оценки CNN по классам в процентах.",
    "svm_ovr_decision_score": "Внутренние оценки SVM. Класс выбирается голосованием, возможны отрицательные числа.",
    "mean_log_posterior_score": "Логарифмические оценки GMM. Выше — лучше, например −53 выше −55.",
}

from .theme import STYLE



def configure_application(app: QApplication):
    app.setApplicationName("Распознавание эмоций по голосу")
    app.setStyle("Fusion")
    # The Windows offscreen plugin may not enumerate installed system fonts.
    # Load the standard UI font explicitly for reproducible screenshot checks.
    if app.platformName() == "offscreen" and os.name == "nt":
        font_directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for font_file in ["segoeui.ttf", "segoeuib.ttf"]:
            font_path = font_directory / font_file
            if font_path.exists():
                QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Segoe UI", 9))
    palette = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window, "#0c0c0e"),
        (QPalette.ColorRole.WindowText, "#eaeaf2"),
        (QPalette.ColorRole.Base, "#111114"),
        (QPalette.ColorRole.Text, "#eaeaf2"),
        (QPalette.ColorRole.Button, "#0c0c0e"),
        (QPalette.ColorRole.ButtonText, "#eaeaf2"),
        (QPalette.ColorRole.AlternateBase, "#17171b"),
        (QPalette.ColorRole.ToolTipBase, "#1e1e24"),
        (QPalette.ColorRole.ToolTipText, "#eaeaf2"),
        (QPalette.ColorRole.PlaceholderText, "#85859a"),
        (QPalette.ColorRole.Highlight, "#27395e"),
        (QPalette.ColorRole.HighlightedText, "#eaeaf2"),
    ]:
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)


def label(text: str, muted=False, wrap=False) -> QLabel:
    widget = QLabel(text)
    widget.setWordWrap(wrap)
    if muted:
        widget.setProperty("role", "muted")
    return widget


def table(headers: list[str], rows=0) -> QTableWidget:
    widget = QTableWidget(rows, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.verticalHeader().setVisible(False)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setAlternatingRowColors(True)
    widget.setShowGrid(False)
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return widget


class PreviewWorker(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    @Slot()
    def run(self):
        try:
            self.result.emit(load_preview(self.path))
        except Exception as error:
            self.error.emit(str(error))
        finally:
            self.finished.emit()


class PredictionWorker(PreviewWorker):
    def __init__(self, preview, run, condition, snr):
        super().__init__(preview.path)
        self.preview, self.model_run, self.condition, self.snr = preview, run, condition, snr

    @Slot()
    def run(self):
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=2):
                predictor = Predictor(self.model_run)
                if predictor.name == "cnn":
                    predictor.torch.set_num_threads(2)
                self.result.emit(predictor.compare_waveform(self.preview.waveform, self.preview.sample_rate,
                                                            self.condition, self.snr))
        except Exception as error:
            self.error.emit(str(error))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание эмоций по голосу")
        self.setAcceptDrops(True)
        self.resize(1280, 880)
        self.setMinimumSize(1040, 760)
        self.preview: AudioPreview | None = None
        self.thread = None
        self.worker = None
        self.close_requested = False
        self.loaded_metrics = {}
        self.prediction = None
        self.loading_audio = False
        try:
            self.runs = final_runs()
            model_error = None
        except (OSError, ValueError, KeyError) as error:
            self.runs = []
            model_error = str(error)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(.6)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)
        self.player.seekableChanged.connect(self.update_seekability)
        self.player.playbackStateChanged.connect(self.update_playback)
        self.player.mediaStatusChanged.connect(self.update_media_status)
        self.player.errorOccurred.connect(self.playback_error)
        self._build_actions()
        self._build_ui()
        self.statusBar().showMessage("Откройте аудиофайл или перетащите его в окно.")
        if model_error:
            self.model_state.setToolTip(model_error)
        results_path = PROJECT / "reports/stage3/test_summary.csv"
        if results_path.is_file():
            try:
                self.read_metrics(results_path)
            except (OSError, ValueError, csv.Error) as error:
                self.metrics_note.setText(f"Ошибка чтения результатов: {error}")

    def _build_actions(self):
        self.open_action = QAction("Открыть аудио…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.choose_file)
        self.export_action = QAction("Экспорт сведений…", self)
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export_metadata)
        close_action = QAction("Выход", self)
        close_action.setShortcut(QKeySequence("Alt+F4"))
        close_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(close_action)
        view_menu = self.menuBar().addMenu("Вид")
        view_menu.addAction("Результаты экспериментов…", self.show_experiments)
        help_menu = self.menuBar().addMenu("Справка")
        about_action = help_menu.addAction("О программе")
        about_action.triggered.connect(lambda: QMessageBox.information(
            self, "О программе", "Распознавание эмоций по голосу с помощью CNN, GMM и SVM.\n\n"
            f"Поддерживаемые форматы: {AUDIO_FORMATS}."))

    def _panel(self, title, hint=None):
        frame = QFrame()
        frame.setObjectName("panel")
        box = QVBoxLayout(frame)
        box.setContentsMargins(17, 16, 17, 15)
        box.setSpacing(12)
        box.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        header = QHBoxLayout()
        title_label = label(title)
        title_label.setProperty("role", "section")
        header.addWidget(title_label)
        header.addStretch()
        if hint:
            note = label(hint, muted=True)
            note.setProperty("role", "quiet")
            header.addWidget(note)
        box.addLayout(header)
        return frame, box

    def _rule(self):
        line = QFrame()
        line.setProperty("role", "rule")
        return line

    def _scroll_page(self, page):
        page.layout().setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 19, 24, 15)
        layout.setSpacing(14)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        title = label("Распознавание эмоций")
        title.setProperty("role", "title")
        titles.addWidget(title)
        header.addLayout(titles, 1)
        experiments_button = QPushButton("Эксперименты")
        experiments_button.clicked.connect(self.show_experiments)
        header.addWidget(experiments_button)
        self.export_button = QPushButton("Экспорт сведений")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_metadata)
        self.export_action.changed.connect(lambda: self.export_button.setEnabled(self.export_action.isEnabled()))
        header.addWidget(self.export_button)
        self.open_button = QPushButton("Открыть запись")
        self.open_button.setProperty("variant", "primary")
        self.open_button.setToolTip("Выбрать аудиофайл · Ctrl+O")
        self.open_button.clicked.connect(self.choose_file)
        self.open_action.changed.connect(lambda: self.open_button.setEnabled(self.open_action.isEnabled()))
        header.addWidget(self.open_button)
        layout.addLayout(header)
        layout.addWidget(self._scroll_page(self._analysis_tab()), 1)
        self.error_label = label("", wrap=True)
        self.error_label.setProperty("role", "error")
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.setCentralWidget(central)
        self.experiments_dialog = QDialog(self)
        self.experiments_dialog.setWindowTitle("Результаты экспериментов")
        self.experiments_dialog.resize(980, 690)
        dialog_layout = QVBoxLayout(self.experiments_dialog)
        dialog_layout.addWidget(self._scroll_page(self._experiments_tab()))

    def show_experiments(self):
        self.experiments_dialog.show()
        self.experiments_dialog.raise_()
        self.experiments_dialog.activateWindow()

    def _analysis_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.file_panel = QFrame()
        self.file_panel.setObjectName("filePanel")
        file_layout = QVBoxLayout(self.file_panel)
        file_layout.setContentsMargins(16, 11, 16, 12)
        file_layout.setSpacing(3)
        name_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("fileName")
        self.path_edit.setReadOnly(True)
        self.path_edit.setAcceptDrops(False)
        self.path_edit.setPlaceholderText("Перетащите аудиофайл сюда")
        self.path_edit.setAccessibleName("Имя аудиозаписи; полный путь во всплывающей подсказке")
        name_row.addWidget(self.path_edit, 1)
        self.format_label = label("", muted=True)
        self.format_label.setProperty("role", "quiet")
        name_row.addWidget(self.format_label)
        file_layout.addLayout(name_row)
        self.file_info = label(AUDIO_FORMATS, muted=True)
        file_layout.addWidget(self.file_info)
        layout.addWidget(self.file_panel)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(16)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)
        wave_panel, wave_box = self._panel("Звуковой сигнал", "Перемотка по волне")
        self.wave_plot = SignalPlot("wave")
        self.wave_plot.seek_requested.connect(self.seek_audio)
        wave_box.addWidget(self.wave_plot, 1)
        playback = QHBoxLayout()
        playback.setSpacing(10)
        self.play_button = QPushButton("Воспроизвести")
        self.play_button.setMinimumWidth(125)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_playback)
        self.stop_button = QPushButton("Стоп")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.player.stop)
        self.position_label = label("00:00 / 00:00", muted=True)
        self.position_label.setMinimumWidth(89)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        playback.addWidget(self.play_button)
        playback.addWidget(self.stop_button)
        playback.addStretch(1)
        playback.addWidget(self.position_label)
        wave_box.addLayout(playback)
        left_layout.addWidget(wave_panel, 2)
        mel_panel, mel_box = self._panel("Log-Mel спектрограмма", "Вся запись")
        self.mel_plot = SignalPlot("mel")
        mel_box.addWidget(self.mel_plot, 1)
        self.scale_info = label("", muted=True)
        self.scale_info.setProperty("role", "quiet")
        mel_box.addWidget(self.scale_info)
        left_layout.addWidget(mel_panel, 3)
        splitter.addWidget(left)
        inspector = QWidget()
        inspector.setMinimumWidth(285)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(16)
        results_panel, results_box = self._panel("Распознавание")
        results_box.setSpacing(10)
        self.model_state = label("Обученные модели недоступны")
        self.model_state.setProperty("role", "state")
        self.model_state.setVisible(not self.runs)
        results_box.addWidget(self.model_state)
        settings = QFormLayout()
        self.model_combo, self.noise_combo = QComboBox(), QComboBox()
        for name in MODELS:
            self.model_combo.addItem(name.upper(), name)
        settings.addRow("Модель", self.model_combo)
        self.model_combo.currentIndexChanged.connect(self.clear_prediction)
        results_box.addLayout(settings)
        self.recognize_button = QPushButton("Распознать эмоцию")
        self.recognize_button.setProperty("variant", "primary")
        self.recognize_button.setEnabled(False)
        self.recognize_button.clicked.connect(self.recognize)
        results_box.addWidget(self.recognize_button)
        results_box.addWidget(self._rule())
        result_card = QFrame()
        result_card.setObjectName("predictionCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(12, 9, 12, 9)
        result_layout.setSpacing(3)
        result_layout.addWidget(label("Предсказанная эмоция", muted=True))
        self.predicted_label = label("—")
        self.predicted_label.setProperty("role", "emotion")
        result_layout.addWidget(self.predicted_label)
        self.noisy_label = label("")
        self.noisy_label.setFixedHeight(20)
        result_layout.addWidget(self.noisy_label)
        results_box.addWidget(result_card)
        self.details_button = QPushButton("Параметры и оценки")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self.update_details)
        results_box.addWidget(self.details_button)
        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 4, 0)
        details_layout.setSpacing(8)
        self.noise_combo.setToolTip("Шум добавляется к копии сигнала. При 0 дБ мощности шума и сигнала равны.")
        for kind, name, snr in CONDITIONS:
            self.noise_combo.addItem("Без добавления шума" if snr is None else f"{name} · {snr} дБ", (kind, snr))
        self.noise_combo.currentIndexChanged.connect(self.clear_prediction)
        details_layout.addWidget(label("Сравнение с шумом", muted=True))
        details_layout.addWidget(self.noise_combo)
        self.score_note = label("", muted=True, wrap=True)
        details_layout.addWidget(self.score_note)
        self.results_table = table(["Эмоция", "Оценка", "С шумом"], 6)
        self.results_table.setColumnHidden(2, True)
        self.results_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(1, 65)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(2, 66)
        self.results_table.verticalHeader().setDefaultSectionSize(32)
        self.results_table.horizontalHeader().setFixedHeight(32)
        self.results_table.setFixedHeight(225)
        for row_number, (code, name) in enumerate(CLASS_NAMES):
            item = QTableWidgetItem(name)
            item.setForeground(QColor(EMOTION_COLORS[code]))
            self.results_table.setItem(row_number, 0, item)
            for column in (1, 2):
                value = QTableWidgetItem("—")
                value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.results_table.setItem(row_number, column, value)
        details_layout.addWidget(self.results_table)
        details_layout.addStretch()
        # The details can scroll independently, so showing scores never widens the main page.
        self.details_scroll = self._scroll_page(details)
        self.details_scroll.setMinimumHeight(100)
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        policy.setRetainSizeWhenHidden(True)
        self.details_scroll.setSizePolicy(policy)
        results_box.addWidget(self.details_scroll, 1)
        self.update_details()
        results_box.addStretch()
        inspector_layout.addWidget(results_panel, 1)
        splitter.addWidget(inspector)
        splitter.setSizes([870, 310])
        layout.addWidget(splitter, 1)
        return page

    def _experiments_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 18, 0, 0)
        layout.setSpacing(16)
        panel, box = self._panel("Качество модели", "Чистая речь и фоновый шум")
        controls = QHBoxLayout()
        controls.setSpacing(12)
        controls.addWidget(label("Модель", muted=True))
        self.metrics_model = QComboBox()
        for name in ("cnn", "gmm", "svm"):
            self.metrics_model.addItem(name.upper(), name)
        self.metrics_model.currentIndexChanged.connect(self.refresh_metrics)
        controls.addWidget(self.metrics_model)
        controls.addWidget(label("Разбиение", muted=True))
        self.split_combo = QComboBox()
        self.split_combo.setMinimumWidth(190)
        self.split_combo.addItem("По дикторам", "speaker")
        self.split_combo.addItem("Случайное", "random")
        self.split_combo.currentIndexChanged.connect(self.refresh_metrics)
        controls.addWidget(self.split_combo)
        controls.addStretch()
        template_button = QPushButton("Шаблон CSV")
        template_button.clicked.connect(self.export_template)
        controls.addWidget(template_button)
        import_button = QPushButton("Импорт результатов")
        import_button.setProperty("variant", "primary")
        import_button.clicked.connect(self.import_metrics)
        controls.addWidget(import_button)
        box.addLayout(controls)
        self.metrics_table = table(["Тестовые данные", "SNR, дБ", "Macro-F1", "UAR", "Accuracy"], 7)
        self.metrics_table.verticalHeader().setDefaultSectionSize(46)
        self.metrics_table.setFixedHeight(367)
        box.addWidget(self.metrics_table)
        self.metrics_note = label("Результаты ещё не загружены.", muted=True, wrap=True)
        box.addWidget(self.metrics_note)
        layout.addWidget(panel)
        notes = QHBoxLayout()
        protocol_panel, protocol_box = self._panel("Протокол оценки")
        self.protocol_note = label("", muted=True, wrap=True)
        protocol_box.addWidget(self.protocol_note)
        metric_panel, metric_box = self._panel("Как читать результаты")
        metric_box.addWidget(label("Macro-F1 учитывает все эмоции с равным весом. UAR — средняя полнота по классам. Все значения находятся в диапазоне 0–1.", muted=True, wrap=True))
        notes.addWidget(protocol_panel, 1)
        notes.addWidget(metric_panel, 1)
        layout.addLayout(notes)
        layout.addStretch(1)
        self.refresh_metrics()
        return page

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть аудиозапись", "", AUDIO_FILTER)
        if path:
            self.load_file(path)

    def _dropped_audio(self, event):
        urls = event.mimeData().urls()
        if self.thread is not None or len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        return str(path) if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file() else None

    def _highlight_drop(self, active):
        self.file_panel.setProperty("dropActive", active)
        self.file_panel.style().unpolish(self.file_panel)
        self.file_panel.style().polish(self.file_panel)

    def dragEnterEvent(self, event):
        if self._dropped_audio(event):
            self._highlight_drop(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._highlight_drop(False)
        event.accept()

    def dropEvent(self, event):
        self._highlight_drop(False)
        path = self._dropped_audio(event)
        if path:
            event.acceptProposedAction()
            self.load_file(path)
        else:
            event.ignore()

    def load_file(self, path: str):
        if self.thread is not None:
            return
        self.loading_audio = True
        self.player.stop()
        self.update_media_status(self.player.mediaStatus())
        self.statusBar().showMessage("Чтение аудио и расчёт Log-Mel…")
        self.start_worker(PreviewWorker(path), self.apply_preview)

    def start_worker(self, worker, handler):
        self.error_label.hide()
        self.open_action.setEnabled(False)
        for widget in (self.model_combo, self.noise_combo, self.recognize_button, self.details_button):
            widget.setEnabled(False)
        self.thread = QThread(self)
        self.worker = worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(handler)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.finish_loading)
        self.thread.start()

    @Slot(object)
    def apply_preview(self, preview: AudioPreview):
        self.preview = preview
        self.clear_prediction()
        self.path_edit.setText(Path(preview.path).name)
        self.format_label.setText(Path(preview.path).suffix.lstrip(".").upper())
        self.path_edit.setToolTip(preview.path)
        self.path_edit.setCursorPosition(0)
        self.file_info.setText(f"{preview.duration_s:.2f} с   ·   {preview.sample_rate / 1000:g} кГц   ·   {preview.channels} кан.   ·   Локальный файл")
        self.wave_plot.set_data(preview.waveform, preview.duration_s)
        self.mel_plot.set_data(preview.log_mel, preview.duration_s)
        self.scale_info.setText(f"Мощность: {self.mel_plot.vmin:.1f} … {self.mel_plot.vmax:.1f} дБ. Частоты по шкале Mel.")
        self.player.setSource(QUrl())
        device = QMediaDevices.defaultAudioOutput()
        if not device.isNull() and device != self.audio_output.device():
            self.audio_output.setDevice(device)
        self.player.setSource(QUrl.fromLocalFile(preview.path))
        self.update_media_status(self.player.mediaStatus())
        self.export_action.setEnabled(True)
        self.update_position(0)
        self.statusBar().showMessage("Запись открыта.")

    @Slot()
    def recognize(self):
        if self.thread is not None or self.preview is None or not self.runs:
            return
        self.clear_prediction()
        run = demo_run(self.runs, self.model_combo.currentData(), "speaker")
        kind, snr = self.noise_combo.currentData()
        self.statusBar().showMessage("Распознавание…")
        self.start_worker(PredictionWorker(self.preview, run, kind, snr), self.apply_prediction)

    @Slot()
    def clear_prediction(self):
        self.prediction = None
        self.predicted_label.setText("—")
        self.predicted_label.setStyleSheet("")
        self.noisy_label.clear()
        self.noisy_label.setStyleSheet("")
        self.score_note.clear()
        self.update_details()
        self.results_table.setColumnHidden(2, True)
        for row in range(6):
            for column in (1, 2):
                self.results_table.item(row, column).setText("—")
                self.results_table.item(row, column).setBackground(QBrush())

    @Slot(object)
    def apply_prediction(self, result):
        self.prediction = result
        names = dict(CLASS_NAMES)
        clean = result["clean"]
        noisy = result.get("noisy")
        clean_code = clean["emotion"]
        self.predicted_label.setText(names[clean_code])
        self.predicted_label.setStyleSheet(f"color: {EMOTION_COLORS[clean_code]};")
        if noisy:
            code = noisy["emotion"]
            self.noisy_label.setText(f"С шумом: {names[code]}")
            self.noisy_label.setStyleSheet(f"color: {EMOTION_COLORS[code]};")
        self.score_note.setText(SCORE_HELP[clean["score_type"]])
        self.update_details()
        self.results_table.setColumnHidden(2, noisy is None)
        for column, prediction in enumerate((clean, noisy), 1):
            if prediction is None:
                continue
            for row, (code, _) in enumerate(CLASS_NAMES):
                value = prediction["scores"][code]
                item = self.results_table.item(row, column)
                item.setText(f"{value:.1%}" if prediction["model"] == "cnn" else f"{value:.3f}")
                if code == prediction["emotion"]:
                    tint = QColor(EMOTION_COLORS[code])
                    tint.setAlpha(55)
                    item.setBackground(QBrush(tint))
        self.statusBar().showMessage("Распознавание завершено.")

    @Slot()
    def update_details(self):
        expanded = self.details_button.isChecked()
        self.details_scroll.setVisible(expanded)
        self.score_note.setVisible(expanded and self.prediction is not None)
        self.results_table.setVisible(expanded and self.prediction is not None)

    @Slot()
    def finish_loading(self):
        self.thread = None
        self.worker = None
        self.loading_audio = False
        self.open_action.setEnabled(True)
        self.update_media_status(self.player.mediaStatus())
        for widget in (self.model_combo, self.noise_combo, self.details_button):
            widget.setEnabled(True)
        self.recognize_button.setEnabled(bool(self.runs and self.preview is not None))
        if self.close_requested:
            self.close()

    @Slot(str)
    def show_error(self, message: str):
        self.error_label.setText(message)
        self.error_label.show()
        self.statusBar().showMessage("Операция не выполнена.")

    def toggle_playback(self):
        if not self.play_button.isEnabled():
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if (self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia
                    or self.player.position() >= self.player.duration() > 0):
                self.player.setPosition(0)
            self.player.play()

    def update_media_status(self, status):
        ready = not self.loading_audio and self.preview is not None and status in {
            QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferingMedia,
            QMediaPlayer.MediaStatus.BufferedMedia, QMediaPlayer.MediaStatus.StalledMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        }
        self.play_button.setEnabled(ready)
        self.stop_button.setEnabled(ready)
        self.update_seekability(self.player.isSeekable())

    def playback_error(self, _error, message):
        if self.player.source().isValid():
            self.show_error(f"Воспроизведение: {message}")

    def update_playback(self, state):
        self.play_button.setText("Пауза" if state == QMediaPlayer.PlaybackState.PlayingState else "Воспроизвести")

    def update_duration(self, duration):
        self.update_position(self.player.position())

    def update_seekability(self, seekable):
        self.wave_plot.set_seekable(bool(seekable and self.preview and not self.loading_audio))

    def seek_audio(self, seconds):
        if self.player.isSeekable():
            position = min(round(seconds * 1000), self.player.duration())
            self.player.setPosition(position)
            self.update_position(self.player.position())

    def update_position(self, position):
        self.wave_plot.set_position(position / 1000)
        duration = self.player.duration() / 1000 or (self.preview.duration_s if self.preview else 0)
        def time_text(seconds):
            value = int(seconds)
            return f"{value//60:02d}:{value%60:02d}"
        self.position_label.setText(f"{time_text(position/1000)} / {time_text(duration)}")

    def export_metadata(self):
        if self.preview is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить сведения", "audio-info.json", "JSON (*.json)")
        if path:
            info = self.preview.metadata()
            info.update(model_connected=bool(self.runs), prediction=self.prediction)
            self.save_text(path, json.dumps(info, ensure_ascii=False, indent=2))

    def save_text(self, path, content):
        try:
            Path(path).write_text(content, encoding="utf-8")
            self.statusBar().showMessage("Сохранено: " + path)
        except OSError as error:
            self.show_error(str(error))

    def export_template(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить шаблон", "metrics-template.csv", "CSV (*.csv)")
        if not path:
            return
        stream = io.StringIO()
        writer = csv.DictWriter(stream, CSV_FIELDS)
        writer.writeheader()
        for model in MODELS:
            for split in ["speaker", "random"]:
                for kind, _name, snr in CONDITIONS:
                    writer.writerow({"model": model, "split": split, "condition": kind, "snr_db": snr})
        self.save_text(path, "\ufeff" + stream.getvalue())

    def import_metrics(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить результаты", "", "CSV (*.csv)")
        if not path:
            return
        try:
            self.read_metrics(path)
            self.error_label.hide()
            self.statusBar().showMessage("Загружены результаты: " + path)
        except (OSError, ValueError, csv.Error) as error:
            self.show_error("CSV: " + str(error))

    def read_metrics(self, path):
        loaded = {}
        with open(path, encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"split", "condition", "snr_db", "macro_f1", "uar", "accuracy"}.issubset(reader.fieldnames or []):
                raise ValueError("Используйте столбцы из шаблона CSV.")
            for line, row in enumerate(reader, 2):
                split, kind = row["split"], row["condition"]
                if split not in {"speaker", "random"} or kind not in {"clean", "white", "background"}:
                    raise ValueError(f"Неизвестное условие или разбиение в строке {line}.")
                raw_snr = (row["snr_db"] or "").strip()
                snr = None if raw_snr == "" else float(raw_snr)
                if (kind == "clean" and snr is not None) or (kind != "clean" and snr not in {0, 10, 20}):
                    raise ValueError(f"Некорректный SNR в строке {line}.")
                values = []
                for field in ["macro_f1", "uar", "accuracy"]:
                    raw = (row[field] or "").strip()
                    value = None if raw == "" else float(raw)
                    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError(f"{field}: требуется число от 0 до 1, строка {line}.")
                    values.append(value)
                model = (row.get("model") or "cnn").strip().lower()
                if model not in MODELS:
                    raise ValueError(f"Неизвестная модель в строке {line}.")
                key = (model, split, kind, snr)
                if key in loaded:
                    raise ValueError(f"Повтор условия в строке {line}. Сначала агрегируйте seed обучения.")
                loaded[key] = values
        self.loaded_metrics = loaded
        self.refresh_metrics()

    def refresh_metrics(self):
        split = self.split_combo.currentData()
        if hasattr(self, "protocol_note"):
            self.protocol_note.setText("Дикторы тестовой выборки не встречаются при обучении и валидации. Проверяем перенос на новые голоса." if split == "speaker" else "Записи разделены случайно с учётом эмоций. Один диктор может встречаться в разных выборках.")
        for row_number, (kind, name, snr) in enumerate(CONDITIONS):
            values = self.loaded_metrics.get((self.metrics_model.currentData(), split, kind, snr), [None]*3)
            cells = [name, "—" if snr is None else str(snr)] + ["—" if value is None else f"{value:.3f}" for value in values]
            for column, value in enumerate(cells):
                self.metrics_table.setItem(row_number, column, QTableWidgetItem(value))
        if hasattr(self, "metrics_note"):
            self.metrics_note.setText("Показаны значения из загруженного CSV." if self.loaded_metrics else "Результаты ещё не загружены. Импортируйте CSV после оценки модели.")

    def closeEvent(self, event):
        if self.thread is not None:
            self.close_requested = True
            self.statusBar().showMessage("Завершение обработки перед закрытием…")
            event.ignore()
            return
        self.player.stop()
        self.player.setSource(QUrl())
        super().closeEvent(event)
