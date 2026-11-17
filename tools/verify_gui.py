"""Offscreen integration probes and captures of the existing application."""
import os
import argparse
import tempfile
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from qt_gui.main import MainWindow, configure_application, CONDITIONS
from qt_gui.audio import load_preview
from ser.inference import Predictor, demo_run, final_runs
from threadpoolctl import threadpool_limits

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('reports/verification'))
parser.add_argument('--screenshots',action='store_true')
args=parser.parse_args()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
app=QApplication([])
configure_application(app)
window=MainWindow()
window.show()
def wait_worker():
    deadline=time.monotonic()+30
    while window.thread is not None and time.monotonic()<deadline:
        QTest.qWait(20)
    assert window.thread is None, 'Worker timed out'
    assert not window.error_label.isVisible(),window.error_label.text()
record=ROOT/'data/raw/CREMA-D/AudioWAV/1003_DFA_ANG_XX.wav'
window.load_file(str(record))
wait_worker()
report={'predictions':[],'geometry':[],'silence':{},'formats':[]}
window.details_button.setChecked(True)
with threadpool_limits(2):
    for model in ('svm','gmm','cnn'):
        predictor=Predictor(demo_run(final_runs(),model,'speaker'))
        if model=='cnn': predictor.torch.set_num_threads(2)
        silent=predictor.predict_waveform(np.zeros(64000,dtype=np.float32),16000)
        report['silence'][model]=silent
        for condition,snr in [('clean',None),('white',0),('background',10)]:
            window.model_combo.setCurrentIndex(window.model_combo.findData(model))
            window.noise_combo.setCurrentIndex(next(i for i,(kind,_,level) in enumerate(CONDITIONS) if (kind,level)==(condition,snr)))
            window.recognize()
            wait_worker()
            expected=predictor.compare_waveform(window.preview.waveform,window.preview.sample_rate,condition,snr)
            assert window.prediction==expected
            report['predictions'].append({'model':model,'condition':condition,'snr_db':snr,'emotion':expected['clean']['emotion'],'matches_cli_backend':True})
            if args.screenshots and condition=='background':
                window.grab().save(str(OUT/f'gui_{model}_noise.png'))
    for width,height in [(1040,760),(1280,880),(1920,1080)]:
        window.resize(width,height)
        window.details_button.setChecked(False)
        QTest.qWait(50)
        before=[p.geometry().getRect() for p in (window.wave_plot,window.mel_plot)]
        window.details_button.setChecked(True)
        QTest.qWait(50)
        after=[p.geometry().getRect() for p in (window.wave_plot,window.mel_plot)]
        assert before==after
        report['geometry'].append({'width':width,'height':height,'plots_do_not_move':True,
                                   'details_scroll_max':window.details_scroll.verticalScrollBar().maximum()})
        if args.screenshots:
            window.grab().save(str(OUT/f'gui_{width}x{height}.png'))
    for model in ('cnn','gmm','svm'):
        window.metrics_model.setCurrentIndex(window.metrics_model.findData(model))
        for split in ('speaker','random'):
            window.split_combo.setCurrentIndex(window.split_combo.findData(split))
            for i,(kind,_,snr) in enumerate(CONDITIONS):
                values=window.loaded_metrics[(model,split,kind,snr)]
                assert [window.metrics_table.item(i,c).text() for c in (2,3,4)]==[f'{v:.3f}' for v in values]
    report['metrics_display_rows_verified']=42
    window.experiments_dialog.show()
    QTest.qWait(50)
    if args.screenshots:
        window.experiments_dialog.grab().save(str(OUT/'gui_experiments.png'))
    window.experiments_dialog.hide()
    temporary=tempfile.TemporaryDirectory()
    scratch=Path(temporary.name)
    with patch('qt_gui.main.QFileDialog.getSaveFileName',return_value=(str(scratch/'export_example.json'),'JSON')):
        window.export_metadata()
    exported=json.loads((scratch/'export_example.json').read_text(encoding='utf-8'))
    assert exported['prediction']==window.prediction and exported['model_connected']
    report['json_export']='verified'
    tone=(.1*np.sin(2*np.pi*440*np.arange(16000)/16000)).astype(np.float32)
    for ext,fmt,subtype in [('wav','WAV','PCM_16'),('mp3','MP3','MPEG_LAYER_III'),('flac','FLAC','PCM_16'),('ogg','OGG','VORBIS'),('aiff','AIFF','PCM_16')]:
        path=scratch/f'audio.{ext}'
        sf.write(path,tone,16000,format=fmt,subtype=subtype)
        preview=load_preview(str(path))
        assert np.isfinite(preview.log_mel).all()
        report['formats'].append(ext)
    temporary.cleanup()
window.close()
QTest.qWait(100)
(OUT/'gui_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2))
