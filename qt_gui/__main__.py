import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .main import MainWindow, configure_application


def main():
    parser = argparse.ArgumentParser(description="Распознавание эмоций по голосу")
    parser.add_argument("--audio", help="Аудиофайл для открытия при запуске")
    parser.add_argument("--screenshot", help="Сохранить снимок окна и завершить работу")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    configure_application(app)
    window = MainWindow()
    window.show()
    if args.audio:
        QTimer.singleShot(0, lambda: window.load_file(args.audio))
    if args.screenshot:
        elapsed = [0]
        timer = QTimer()
        def capture():
            elapsed[0] += 100
            if elapsed[0] < 500 or window.thread is not None:
                if elapsed[0] >= 30000:
                    print("Не удалось дождаться обработки аудио.", file=sys.stderr)
                    window.close_requested = True
                return
            timer.stop()
            success = window.grab().save(str(Path(args.screenshot).resolve()))
            exit_code = 0 if success and (not args.audio or window.preview is not None) else 1
            window.close()
            app.exit(exit_code)
        timer.timeout.connect(capture)
        timer.start(100)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
