"""Small native Qt plot widgets; no browser or plotting runtime required."""
import numpy as np
from PySide6.QtCore import Qt, QRectF, QLineF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .theme import SURFACE, BORDER, MUTED, QUIET, ACCENT


def color_lut():
    anchors = np.array([[12, 12, 14], [24, 33, 57], [41, 71, 126], [79, 126, 248], [220, 232, 255]])
    coordinates = np.linspace(0, 1, len(anchors))
    return np.column_stack([np.interp(np.linspace(0, 1, 256), coordinates, anchors[:, i]) for i in range(3)]).astype(np.uint8)


LUT = color_lut()


class SignalPlot(QWidget):
    seek_requested = Signal(float)

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.data = None
        self.duration = 4.0
        self.image = None
        self.position = 0.0
        self.seekable = False
        self._dragging = False
        self._wave_size = None
        self._wave_lines = []
        self.vmin, self.vmax = -100.0, 0.0
        self.setMinimumSize(300, 120 if kind == "wave" else 190)
        self.setAccessibleName("Осциллограмма" if kind == "wave" else "Log-Mel спектрограмма")
        if kind == "wave":
            self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            self.setToolTip("Щёлкните по волне или потяните курсор для перемотки.")

    def set_data(self, data: np.ndarray, duration: float = 4.0):
        self.data = data
        self.duration = duration
        self.position = 0.0
        self._wave_size = None
        if self.kind == "mel":
            self.vmax = float(data.max())
            self.vmin = self.vmax - 80
            indices = np.flipud(np.clip((data-self.vmin)/80*255, 0, 255).astype(np.uint8))
            pixels = np.ascontiguousarray(LUT[indices])
            self.image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def plot_rect(self):
        return QRectF(46, 10, max(1, self.width()-104), max(1, self.height()-44))

    def set_position(self, seconds):
        self.position = min(max(seconds, 0.0), self.duration)
        self.update()

    def set_seekable(self, enabled):
        self.seekable = enabled
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _seek_at(self, point):
        plot = self.plot_rect()
        fraction = min(max((point.x() - plot.left()) / plot.width(), 0.0), 1.0)
        self.seek_requested.emit(fraction * self.duration)

    def mousePressEvent(self, event):
        if self.seekable and self.data is not None and event.button() == Qt.MouseButton.LeftButton and self.plot_rect().contains(event.position()):
            self._dragging = True
            self._seek_at(event.position())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_at(event.position())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._seek_at(event.position())
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        steps = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}
        if self.seekable and self.data is not None and event.key() in steps:
            self.seek_requested.emit(min(max(self.position + steps[event.key()], 0.0), self.duration))
            event.accept()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(SURFACE))
        plot = self.plot_rect()
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRect(plot)
        if self.data is None:
            painter.setPen(QColor(QUIET))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Выберите аудиозапись")
        elif self.kind == "wave":
            self._draw_wave(painter, plot)
        else:
            painter.drawImage(plot, self.image)
            for index in range(256):
                color = QColor(*map(int, LUT[255-index]))
                painter.fillRect(QRectF(plot.right()+12, plot.top()+index*plot.height()/256, 6, plot.height()/256+1), color)
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(plot.right()+22, plot.top()-2, 36, 18), str(round(self.vmax)))
            painter.drawText(QRectF(plot.right()+22, plot.bottom()-16, 36, 18), str(round(self.vmin)))
            painter.drawText(QRectF(plot.right()+28, plot.bottom()+5, 30, 18), "дБ")
        painter.setPen(QColor(QUIET))
        for index in range(5):
            x = plot.left() + plot.width()*index/4
            painter.drawText(QRectF(x-21, plot.bottom()+5, 42, 18), Qt.AlignmentFlag.AlignCenter, f"{self.duration*index/4:.1f}")
        painter.drawText(QRectF(plot.right()-60, self.height()-15, 65, 15), Qt.AlignmentFlag.AlignRight, "Время, с")
        if self.kind == "wave":
            for text, fraction in [("1.0", 0), ("0", .5), ("−1.0", 1)]:
                painter.drawText(QRectF(1, plot.top()+plot.height()*fraction-8, 37, 16), Qt.AlignmentFlag.AlignRight, text)
        else:
            # Positions are linear in the HTK Mel scale, not in Hz.
            lo = 2595*np.log10(1+50/700)
            hi = 2595*np.log10(1+7600/700)
            for frequency in [7600, 2000, 500, 50]:
                fraction = (2595*np.log10(1+frequency/700)-lo)/(hi-lo)
                y = plot.bottom()-plot.height()*fraction
                painter.drawText(QRectF(0, y-8, 39, 16), Qt.AlignmentFlag.AlignRight, str(frequency))

    def _draw_wave(self, painter: QPainter, plot: QRectF):
        painter.setPen(QPen(QColor(BORDER), 1))
        for i in range(1, 4):
            y = plot.top() + plot.height()*i/4
            painter.drawLine(plot.left(), y, plot.right(), y)
        if self._wave_size != self.size():
            self._wave_size = self.size()
            self._wave_lines = []
            columns = max(1, int(plot.width()))
            edges = np.linspace(0, len(self.data), columns+1, dtype=int)
            for col in range(columns):
                values = self.data[edges[col]:max(edges[col]+1, edges[col+1])]
                if len(values):
                    x = plot.left() + col
                    scale = plot.height() / 2
                    self._wave_lines.append(QLineF(x, plot.center().y() - float(values.max()) * scale,
                                                  x, plot.center().y() - float(values.min()) * scale))
        x = plot.left() + plot.width() * self.position / self.duration
        painter.setClipRect(plot)
        painter.setPen(QPen(QColor("#416299"), 1))
        painter.drawLines(self._wave_lines)
        if self.position > 0:
            painter.setClipRect(QRectF(plot.left(), plot.top(), x - plot.left(), plot.height()))
            painter.setPen(QPen(QColor(ACCENT), 1))
            painter.drawLines(self._wave_lines)
        painter.setClipRect(plot)
        painter.setPen(QPen(QColor("#dce8ff"), 1))
        painter.drawLine(QLineF(min(x, plot.right() - 1), plot.top(), min(x, plot.right() - 1), plot.bottom()))
        painter.setClipping(False)
