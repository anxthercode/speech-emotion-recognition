"""Independent audit of existing artifacts. Does not train or replace results."""
import csv
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
import torch
from threadpoolctl import threadpool_limits
from ser.dataset import PreparedDataset, read_csv
from ser.features import DEFAULT_AUDIO, extract_features, log_mel_frames, mel_filterbank
from ser.inference import Predictor, final_runs, demo_run
from ser.splits import read_metadata, validate_split, make_splits, summarize
from ser.noise import noise_segment, measured_snr
from tools.audit_crema import inspect_record, handle_duplicates

OUT = None
result = {}
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a, b, atol=1e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=0)
def save():
    (OUT / 'checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

def audit():
    started = time.perf_counter()
    torch.set_num_threads(2)
    data = PreparedDataset(split='speaker', subset='test')
    root = data.root
    records = read_metadata('data/metadata.csv')
    all_records = read_csv('data/metadata.csv')
    inspected = []
    for index, row in enumerate(all_records, 1):
        actual = inspect_record(ROOT / 'data/raw/CREMA-D/AudioWAV' / row['relative_path'])
        assert actual['sha256'] == row['sha256']
        inspected.append(actual)
        if index % 1000 == 0:
            print(f'Raw WAV decoded and hashed: {index}', flush=True)
    handle_duplicates(inspected)
    for a, b in zip(inspected, all_records):
        assert str(a['valid']) == b['valid']
        assert a['exclusion_reason'] == b['exclusion_reason']
    result['raw_audio'] = {'decoded': len(inspected), 'accepted': len(records),
                           'excluded': [r['file_id'] for r in inspected if not r['valid']]}
    splits = {name: read_csv(root / 'splits' / f'{name}.csv') for name in ('speaker', 'random')}
    regenerated = make_splits(records)
    result['splits'] = {}
    for name, rows in splits.items():
        validate_split(records, rows, name == 'speaker')
        assert {r['file_id']: r['subset'] for r in rows} == {r['file_id']: r['subset'] for r in regenerated[name]}
        result['splits'][name] = summarize(rows)
        norm = read(root / 'normalization' / f'{name}.json')
        assert set(norm['train_file_ids']) == {r['file_id'] for r in rows if r['subset'] == 'train'}
        assert norm['split_sha256'] == sha(root / 'splits' / f'{name}.csv')
    save()
    accumulators = {s: {k: [0, np.zeros(d), np.zeros(d)] for k, d in [('log_mel',64), ('mfcc',39), ('svm',78)]} for s in splits}
    random_test = {r['file_id'] for r in splits['random'] if r['subset'] == 'test'}
    result['cross_protocol_overlap'] = {subset: len(random_test & {r['file_id'] for r in splits['speaker'] if r['subset'] == subset}) for subset in ('train','validation','test')}
    assignments = {s: {r['file_id']: r['subset'] for r in rows} for s, rows in splits.items()}
    max_diff = dict.fromkeys(('log_mel', 'mfcc', 'svm', 'waveform'), 0.)
    wave_hashes = defaultdict(list)
    for i, row in enumerate(records, 1):
        with np.load(root / 'features' / (row['file_id'] + '.npz'), allow_pickle=False) as archive:
            cached = {k: archive[k] for k in archive.files}
        assert cached['source_sha256'].item() == row['sha256']
        waveform, rate = sf.read(ROOT / 'data/raw/CREMA-D/AudioWAV' / row['relative_path'], dtype='float32')
        assert rate == DEFAULT_AUDIO.sample_rate and waveform.ndim == 1
        expected = extract_features(waveform)
        wave_hashes[hashlib.sha256(waveform.tobytes()).hexdigest()].append(row['file_id'])
        for key in max_diff:
            assert np.isfinite(cached[key]).all()
            same(cached[key], expected[key], 1e-5)
            max_diff[key] = max(max_diff[key], float(np.max(np.abs(cached[key] - expected[key]))))
        np.testing.assert_array_equal(cached['window_starts'], expected['window_starts'])
        for split in splits:
            if assignments[split][row['file_id']] == 'train':
                for key, values in [('log_mel', cached['log_mel'][0].T), ('mfcc', cached['mfcc']), ('svm', cached['svm'][None])]:
                    count, sums, squares = accumulators[split][key]
                    values = values.astype(np.float64)
                    accumulators[split][key] = [count + len(values), sums + values.sum(0), squares + (values ** 2).sum(0)]
        if i % 1000 == 0:
            print(f'Features recomputed: {i}', flush=True)
    result['features'] = {'recomputed_files': len(records), 'maximum_absolute_error': max_diff,
                          'decoded_duplicates': [v for v in wave_hashes.values() if len(v)>1]}
    normalization_error = {}
    for split, groups in accumulators.items():
        norm = read(root / 'normalization' / f'{split}.json')
        normalization_error[split] = {}
        for key, (count, sums, squares) in groups.items():
            mean = sums / count
            std = np.maximum(np.sqrt(np.maximum(0, squares / count - mean ** 2)), 1e-6)
            assert count == norm[key]['count']
            same(mean, norm[key]['mean'], 1e-8)
            same(std, norm[key]['std'], 1e-8)
            normalization_error[split][key] = float(max(np.max(np.abs(mean - norm[key]['mean'])), np.max(np.abs(std - norm[key]['std']))))
    result['normalization_max_error'] = normalization_error
    save()
    noise_count = 0
    error = 0.
    for split in splits:
        dataset = PreparedDataset(split=split, subset='test')
        for row in dataset.records:
            clean = dataset.load(row['file_id'])['waveform']
            for kind in ('white', 'background'):
                for snr in (20,10,0):
                    mixed = dataset.noisy_audio(row['file_id'], kind, snr)
                    error = max(error, abs(measured_snr(clean, mixed) - snr))
                    noise_count += 1
        print(f'Noise recipes replayed: {split}', flush=True)
    result['noise'] = {'replayed': noise_count, 'maximum_snr_error_db': error}
    result['data_audit_seconds'] = time.perf_counter() - started
    save()

def audit_models():
    started = time.perf_counter()
    torch.set_num_threads(2)
    root = PreparedDataset(split='speaker', subset='test').root
    splits = {name: read_csv(root / 'splits' / f'{name}.csv') for name in ('speaker', 'random')}
    # Independent STFT implementation: torch, using double precision.
    waveform = np.random.default_rng(12).normal(0,.1,64000)
    spectrum = torch.stft(torch.tensor(waveform), n_fft=512, hop_length=160, win_length=400,
                          window=torch.hann_window(400, dtype=torch.float64), center=False, return_complex=True).abs().square().numpy()
    reference = 10*np.log10(np.maximum(mel_filterbank() @ spectrum, 1e-10))
    difference = float(np.max(np.abs(reference - log_mel_frames(waveform))))
    assert difference < 1e-5
    # Explicit orthonormal cosine matrix, without scipy.fft.dct.
    mel = log_mel_frames(waveform)
    basis = np.cos(np.pi * np.arange(13)[:,None] * (np.arange(64)[None,:]+.5)/64) * np.sqrt(2/64)
    basis[0] /= np.sqrt(2)
    mfcc_error = float(np.max(np.abs(extract_features(waveform)['mfcc'][:,:13].T - basis @ mel)))
    assert mfcc_error < 1e-4
    logits = torch.tensor([[1.,-2.,.5,3.,0.,-1.]], dtype=torch.float64, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([3]))
    loss.backward()
    gradient = logits.detach().softmax(1)
    gradient[0,3] -= 1
    same(logits.grad.numpy(), gradient.numpy())
    result['independent_math'] = {'torch_stft_max_db_error': difference, 'explicit_dct_max_error': mfcc_error, 'ce_gradient': 'matches softmax minus one-hot'}
    runs = final_runs()
    stage2 = read(read('runs/stage2_current.json')['directory'] + '/summary.json')
    manifest = stage2['manifest']
    result['training'] = {'device': manifest['device'], 'gpu': manifest['gpu'], 'versions': manifest['versions'],
                          'final_checkpoints': len(runs), 'models': {},
                          'source_hash_matches': {p: sha(p) == h for p,h in manifest['implementation'].items()}}
    for model in ('cnn','gmm','svm'):
        model_dir = Path(stage2['run_root']) / model
        trials = read(model_dir / 'selection.json')['trials']
        selected = read(model_dir / 'selected.json')
        assert selected['validation']['macro_f1'] == max(t['validation']['macro_f1'] for t in trials)
        all_jobs = list(model_dir.rglob('result.json'))
        result['training']['models'][model] = {'trials': len(trials), 'total_fits': len(all_jobs),
            'parameters': selected['parameters'], 'fit_seconds_all_jobs': sum(read(p)['fit_seconds'] for p in all_jobs),
            'final': [{k:r[k] for k in ('split','seed','validation','best_epoch','epochs','fit_seconds') if k in r} for r in runs if r['model']==model]}
    em = []
    for path in Path(stage2['run_root']).rglob('em_history.json'):
        for group in read(path):
            for init in group['initializations']:
                history = init['history']
                steps = [b['mean_log_likelihood']-a['mean_log_likelihood'] for a,b in zip(history, history[1:]) if not a['reinitialized']]
                assert min(steps, default=0) >= -1e-7
                em.append({'converged':init['converged'], 'iterations':len(history), 'reinitialized':sum(r['reinitialized'] for r in history)})
    result['em'] = {'initializations':len(em),'converged':sum(r['converged'] for r in em), 'maximum_iterations':max(r['iterations'] for r in em),'reinitializations':sum(r['reinitialized'] for r in em)}
    evaluation = read('reports/stage3/evaluation.json')
    result_lookup = {(r['model'],r['split'],str(r['seed']),r['condition'],str(r['snr_db']) if r['snr_db'] is not None else ''):r for r in evaluation['results']}
    groups = defaultdict(list)
    for row in read_csv('reports/stage3/test_predictions.csv'):
        groups[tuple(row[k] for k in ('model','split','seed','condition','snr_db'))].append(row)
    assert set(groups) == set(result_lookup)
    with open('reports/stage3/test_per_class.csv', encoding='utf-8-sig', newline='') as stream:
        class_rows = list(csv.DictReader(stream))
    class_lookup = {tuple(r[k] for k in ('model','split','seed','condition','snr_db','class')):r for r in class_rows}
    for key, rows in groups.items():
        test = {r['file_id']: r for r in splits[key[1]] if r['subset']=='test'}
        assert len(rows)==len(test) and {r['file_id'] for r in rows}==set(test)
        cm = np.zeros((6,6),dtype=int)
        for row in rows:
            assert row['y_true']==test[row['file_id']]['label_id'] and row['speaker_id']==test[row['file_id']]['speaker_id']
            cm[int(row['y_true']),int(row['y_pred'])] += 1
        saved = result_lookup[key]
        same(cm,saved['confusion_matrix'])
        tp = cm.diagonal()
        recall = tp/cm.sum(1)
        precision = np.divide(tp,cm.sum(0),out=np.zeros(6),where=cm.sum(0)>0)
        f1 = 2*tp/(cm.sum(1)+cm.sum(0))
        same([f1.mean(),recall.mean(),tp.sum()/cm.sum()], [saved[k] for k in ('macro_f1','uar','accuracy')])
        same(cm/cm.sum(1,keepdims=True),saved['confusion_normalized'])
        for i,code in enumerate(('ANG','DIS','FEA','HAP','NEU','SAD')):
            row = class_lookup[(*key,code)]
            same([precision[i],recall[i],f1[i],cm.sum(1)[i]], [float(row[k]) for k in ('precision','recall','f1','support')])
    for row in evaluation['summary']:
        matching = [r for r in evaluation['results'] if all(r[k]==row[k] for k in ('model','split','condition','snr_db'))]
        for metric in ('macro_f1','uar','accuracy'):
            same(row[metric],np.mean([r[metric] for r in matching]))
            if len(matching)>1:
                same(row[metric+'_std'],np.std([r[metric] for r in matching],ddof=1))
    with open('reports/stage3/test_summary.csv', encoding='utf-8-sig', newline='') as stream:
        summary_csv = list(csv.DictReader(stream))
    assert len(summary_csv) == len(evaluation['summary']) == 42
    for row in summary_csv:
        matching = next(r for r in evaluation['summary'] if (r['model'],r['split'],r['condition'],str(r['snr_db']) if r['snr_db'] is not None else '') == tuple(row[k] for k in ('model','split','condition','snr_db')))
        for key in ('macro_f1','uar','accuracy','f1_drop_pp','n_test','n_seeds'):
            same(float(row[key]),matching[key])
    result['evaluation'] = {'prediction_rows':sum(map(len,groups.values())),'seed_conditions':len(groups),'summary_conditions':len(evaluation['summary']),'metrics_recomputed':'all confusion matrices, F1/UAR/accuracy, class metrics, means and std match'}
    save()
    # Fresh predictions on balanced samples for every checkpoint and condition.
    fresh = 0
    max_error = defaultdict(float)
    for split in ('speaker','random'):
        dataset = PreparedDataset(split=split,subset='test')
        chosen = [next(r for r in dataset.records if r['label_id']==str(i)) for i in range(6)]
        predictors = [Predictor(r,'cuda' if r['model']=='cnn' else 'cpu') for r in runs if r['split']==split]
        for kind,snr in [('clean',None)]+[(kind,snr) for kind in ('white','background') for snr in (20,10,0)]:
            features = [dataset.features(r['file_id'],kind,snr) for r in chosen]
            for p in predictors:
                key = (p.name,split,str(p.run['seed']),kind,'' if snr is None else str(snr))
                saved = {r['file_id']:r for r in groups[key]}
                predicted,scores = p.predict_batch(features)
                for row,label,score in zip(chosen,predicted,scores):
                    expected = saved[row['file_id']]
                    assert label==int(expected['y_pred'])
                    target = [float(expected['score_'+code]) for code in ('ANG','DIS','FEA','HAP','NEU','SAD')]
                    err = float(np.max(np.abs(score-target)))
                    max_error[p.name] = max(max_error[p.name],err)
                    same(score,target,2e-3 if p.name=='cnn' else 1e-9)
                    fresh += 1
        print(f'Fresh inference verified for all models and noise conditions: {split}',flush=True)
    result['fresh_inference'] = {'predictions':fresh,'maximum_score_error':dict(max_error)}
    result['model_audit_seconds'] = time.perf_counter()-started
    result['complete'] = True
    save()
    print(json.dumps({k:result[k] for k in ('features','noise','independent_math','em','evaluation','fresh_inference','model_audit_seconds')},indent=2),flush=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/verification'))
    parser.add_argument('--resume-models', action='store_true', help='Reuse the data audit in the output directory')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(2):
        if args.resume_models:
            result = read(OUT / 'checks.json')
        else:
            audit()
        audit_models()
