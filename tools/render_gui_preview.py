"""Render the actual Qt views and rebuild the standalone HTML preview."""
import argparse
import base64
from html import escape
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from qt_gui.main import MainWindow, configure_application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=880)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "previews")
    parser.add_argument("--caption", default="Главное окно и результаты экспериментов.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    configure_application(app)
    window = MainWindow()
    window.resize(args.width, args.height)
    window.show()
    if args.audio:
        window.load_file(str(args.audio))
        deadline = time.monotonic() + 15
        while window.thread is not None and time.monotonic() < deadline:
            QTest.qWait(20)
        if window.thread is not None or window.preview is None:
            raise RuntimeError(window.error_label.text() or "Unable to load audio for the screenshot")
    captures = []
    for index, (name, widget) in enumerate([("Аудиозапись", window), ("Эксперименты", window.experiments_dialog)]):
        widget.show()
        QTest.qWait(100)
        filename = "qt_preview.png" if index == 0 else f"qt_view_{index}.png"
        pixmap = widget.grab()
        if not pixmap.save(str(out/filename)):
            raise RuntimeError("Unable to save Qt screenshot")
        captures.append((name, filename, pixmap.width(), pixmap.height()))
    window.close()
    QTest.qWait(100)
    buttons = "".join(f'<button id="tab-{i}" role="tab" aria-controls="screen-{i}" aria-selected="{str(i==0).lower()}" tabindex="{0 if i==0 else -1}" data-index="{i}">{name}</button>' for i, (name, _, _, _) in enumerate(captures))
    images = "".join(f'<section id="screen-{i}" role="tabpanel" aria-labelledby="tab-{i}"'+(' hidden' if i else '')+f'><img alt="Снимок окна Qt: {name}" width="{w}" height="{h}" src="data:image/png;base64,{base64.b64encode((out/file).read_bytes()).decode()}"></section>' for i,(name,file,w,h) in enumerate(captures))
    template = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Распознавание эмоций — Qt GUI</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c0c0e;color:#eaeaf2;font:13px/1.5 "Segoe UI",sans-serif;color-scheme:dark}main{max-width:1340px;margin:24px auto;padding:0 20px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:18px}h1{font-size:18px;font-weight:600;margin:0}p{margin:5px 0;color:#a8a8be;font-size:12px}nav{display:flex;gap:4px;margin-bottom:12px}button{font:inherit;color:#a8a8be;background:#111114;border:1px solid #222228;border-radius:4px;padding:8px 15px;cursor:pointer}button[aria-selected=true]{color:#eaeaf2;background:#1b2640;border-color:#4166c0}button:focus-visible{outline:2px solid #4f7ef8;outline-offset:2px}section{border:1px solid #2c2c35;background:#111114}img{width:100%;height:auto;display:block}footer{border-top:1px solid #222228;padding-top:12px;margin-top:18px}code{font:12px Consolas,monospace;color:#c1d0ff}a{color:#8aaaff}[hidden]{display:none!important}@media(max-width:700px){main{padding:0 10px;margin:14px auto}header{display:block}button{padding:7px 10px;font-size:11px}}
</style></head><body><main><header><div><h1>Распознавание эмоций по голосу</h1><p>Qt Widgets / PySide6 · графитовая тема</p></div></header><nav role="tablist" aria-label="Снимки окон Qt">__BUTTONS__</nav>__IMAGES__<footer><p>__CAPTION__</p><p>Кнопки внутри снимков не интерактивны. Для работы с аудио запустите <code>run_gui.bat</code> или <code>python -m qt_gui</code>.</p><p><a href="README.md">Запуск</a> · <a href="DATASET_NEXT_STEPS.md">Что делать с CREMA-D</a> · <a href="PLAN.md">План проекта</a></p></footer></main><script>
const buttons=[...document.querySelectorAll('[data-index]')];function activate(index){buttons.forEach((b,i)=>{b.setAttribute('aria-selected',String(i===index));b.tabIndex=i===index?0:-1;document.getElementById('screen-'+i).hidden=i!==index;});}buttons.forEach((b,index)=>{b.onclick=()=>activate(index);b.onkeydown=e=>{let n;if(e.key==='ArrowRight')n=(index+1)%buttons.length;else if(e.key==='ArrowLeft')n=(index+buttons.length-1)%buttons.length;else if(e.key==='Home')n=0;else if(e.key==='End')n=buttons.length-1;else return;e.preventDefault();activate(n);buttons[n].focus();};});
</script></body></html>'''
    for document in ("README.md", "DATASET_NEXT_STEPS.md", "PLAN.md"):
        template = template.replace(f'href="{document}"', f'href="{escape(Path(os.path.relpath(ROOT / document, out)).as_posix(), quote=True)}"')
    (out/"gui_mockup.html").write_text(template.replace("__BUTTONS__", buttons).replace("__IMAGES__", images).replace("__CAPTION__", escape(args.caption)), encoding="utf-8")
    print(f"Rendered {len(captures)} Qt views at {captures[0][2]}x{captures[0][3]}")


if __name__ == "__main__":
    main()
