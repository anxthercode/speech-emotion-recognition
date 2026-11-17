"""Dark workspace palette inspired by the user-provided Fathora/al-media themes.

See DESIGN.md for reference links. Layouts and widgets are implemented locally.
"""
from pathlib import Path

BG = "#0c0c0e"
SURFACE = "#111114"
SURFACE_2 = "#17171b"
SURFACE_3 = "#1e1e24"
BORDER = "#222228"
FIELD_BORDER = "#2c2c35"
TEXT = "#eaeaf2"
MUTED = "#a8a8be"
QUIET = "#85859a"
ACCENT = "#4f7ef8"

STYLE = """
QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #eaeaf2; }
QMainWindow, QDialog { background: #0c0c0e; }
QLabel { background: transparent; }
QLabel[role='muted'] { color: #a8a8be; }
QLabel[role='quiet'] { color: #85859a; font-size: 12px; }
QLabel[role='title'] { font-size: 21px; font-weight: 600; }
QLabel[role='section'] { font-size: 14px; font-weight: 600; }
QLabel[role='emotion'] { font-size: 21px; font-weight: 600; }
QLabel[role='filename'] { font-size: 14px; font-weight: 600; }
QLabel[role='error'] { color: #e58a8a; padding: 8px 12px; background: #25191d; border: 1px solid #553038; border-radius: 4px; }
QLabel[role='state'] { color: #a8a8be; background: #1e1e24; border: 1px solid #2c2c35; border-radius: 4px; padding: 5px 9px; }
QMenuBar { background: #111114; border-bottom: 1px solid #222228; }
QMenuBar::item { padding: 5px 13px; background: transparent; color: #a8a8be; }
QMenuBar::item:selected, QMenu::item:selected { background: #1e2943; color: #eaeaf2; }
QMenu { background: #17171b; border: 1px solid #2c2c35; padding: 5px; }
QMenu::item { padding: 7px 28px 7px 13px; }
QMenu::separator { height: 1px; background: #2c2c35; margin: 5px; }
QPushButton, QToolButton { background: #1e1e24; border: 1px solid #2c2c35; border-radius: 5px; min-height: 24px; padding: 6px 13px; }
QPushButton:hover, QToolButton:hover { background: #272731; border-color: #464659; }
QPushButton:pressed { background: #17171b; }
QPushButton:focus, QToolButton:focus { border-color: #4f7ef8; }
QPushButton[variant='primary'] { background: #3968df; border-color: #4f7ef8; color: #ffffff; }
QPushButton[variant='primary']:hover { background: #4677f0; }
QPushButton:disabled, QPushButton[variant='primary']:disabled { color: #77778a; background: #19191f; border-color: #292932; }
QLineEdit, QComboBox { background: #17171b; border: 1px solid #2c2c35; border-radius: 4px; min-height: 24px; padding: 5px 10px; selection-background-color: #293e6d; }
QLineEdit:focus, QComboBox:focus { border-color: #4f7ef8; }
QLineEdit:read-only { color: #eaeaf2; }
QLineEdit#fileName { background: transparent; border: none; font-size: 15px; font-weight: 600; padding: 0; }
QComboBox::drop-down { width: 28px; border: none; }
QComboBox::down-arrow { image: url(__ARROW__); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: #1e1e24; color: #eaeaf2; border: 1px solid #2c2c35; selection-background-color: #253457; }
QFrame#panel { background: #111114; border: 1px solid #222228; border-radius: 6px; }
QFrame#filePanel { background: #17171b; border: 1px solid #2c2c35; border-radius: 6px; }
QFrame#filePanel[dropActive='true'] { background: #1b2640; border-color: #4f7ef8; }
QFrame#predictionCard { background: #17171b; border: 1px solid #2c2c35; border-radius: 5px; }
QFrame[role='rule'] { background: #222228; border: none; max-height: 1px; min-height: 1px; }
QTabWidget::pane { border: none; background: #0c0c0e; }
QTabBar::tab { background: transparent; color: #a8a8be; border: none; border-bottom: 2px solid transparent; padding: 12px 3px 10px; margin-right: 26px; }
QTabBar::tab:selected { color: #eaeaf2; border-bottom-color: #4f7ef8; }
QTabBar::tab:hover:!selected { color: #eaeaf2; border-bottom-color: #2c2c35; }
QTableWidget { background: #111114; alternate-background-color: #151519; border: none; gridline-color: #222228; selection-background-color: #202e4c; selection-color: #eaeaf2; }
QTableWidget::item { padding-left: 8px; padding-right: 8px; border-bottom: 1px solid #222228; }
QTableWidget:focus { outline: none; }
QHeaderView { background: #111114; }
QHeaderView::section { background: #17171b; color: #a8a8be; border: none; border-bottom: 1px solid #2c2c35; padding: 10px 8px; font-size: 12px; font-weight: 400; }
QSlider::groove:horizontal { height: 4px; background: #2c2c35; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #4f7ef8; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -4px 0; border-radius: 6px; background: #b8caff; }
QSlider::handle:horizontal:disabled { background: #58586a; }
QSlider::sub-page:horizontal:disabled { background: #3b3b4c; }
QStatusBar { background: #111114; border-top: 1px solid #222228; padding: 3px 12px; color: #85859a; }
QStatusBar QLabel { color: #85859a; font-size: 11px; }
QStatusBar::item { border: none; }
QSplitter::handle { background: #0c0c0e; width: 12px; }
QScrollArea { border: none; background: #0c0c0e; }
QScrollBar:vertical { background: #111114; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #383844; min-height: 35px; border-radius: 4px; margin: 2px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QToolTip { background: #1e1e24; color: #eaeaf2; border: 1px solid #464659; padding: 5px; }
""".replace("__ARROW__", (Path(__file__).parent / "assets" / "chevron.svg").as_posix())
