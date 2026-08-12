#!/usr/bin/env python3
"""从冻结预测 CSV 重建 Loop151 决策链消融（B/C/D/E 四臂）。"""
import csv, json, math, sys, time
from collections import defaultdict
from pathlib import Path

PROJ = Path(r'e:\Project\python\Axon_v2.6Exp')
OUTDIR = PROJ / 'reports/roadmap_9997/loop151_field_ablation'
OUTDIR.mkdir(parents=True, exist_ok=True)

# 关键阈值
PRIMARY_THR = 0.31
CONSERVATIVE_THR = 0.415
CROSS_THR = 0.4
NOISE_THR = 0.39
SELECTOR_THR = 0.79

def load_csv(path: Path, key_col='source_sha256') -> dict:
    """载入 CSV 并以 key_col 为键返回 dict。"""
    rows = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = row[key_col].strip().casefold()
            rows[key] = row
    return rows

def load_loop151_csv(split: str) -> dict:
    """载入 loop151 预测 CSV（含所有中间分数）。"""
    paths = {
        'val': PROJ / 'reports/phase3_loop151/loop151_trusted_signer_guard_val_predictions.csv',
        'test10k': PROJ / 'reports/phase3_loop151/loop151_trusted_signer_guard_test10k_predictions.csv',
        'full': PROJ / 'reports/phase3_loop151/loop151_trusted_signer_guard_full_predictions.csv',
    }
    return load_csv(paths[split])

def load_loop127_primary(split: str) -> dict:
    """载入 primary (loop127 with_logreg) 预测。"""
    paths = {
        'val': PROJ / 'reports/phase3_loop127/oof_fixed_v2_all_valonly_with_logreg/stage2_oof_stacker_val_predictions.csv',
        'test10k': PROJ / 'reports/phase3_loop127/oof_fixed_v2_all_test10k_predictions.csv',
        'full': PROJ / 'reports/phase3_loop127/oof_fixed_v2_all_full_test_predictions.csv',
    }
    return load_csv(paths[split])

def load_loop130_r5(split: str) -> dict:
    """载入 loop130 R5 guard（含 guard_flip）。"""
    paths = {
        'val': PROJ / 'reports/phase3_loop128/loop130_content_string_guard_val_predictions.csv',
        'test10k': PROJ / 'reports/phase3_loop128/loop130_content_string_guard_r5_test10k_predictions.csv',
        'full': PROJ / 'reports/phase3_loop128/loop130_content_string_guard_r5_full_test_predictions.csv',
    }
    return load_csv(paths[split])

def load_loop136_selector(split: str) -> dict:
    """载入 loop136 selector（含 selector_score, selector_accept_candidate）。"""
    paths = {
        'test10k': PROJ / 'reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_test10k_predictions.csv',
        'full': PROJ / 'reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_full_test_predictions.csv',
    }
    return load_csv(paths[split])

def reconstruct_ablation(split: str) -> dict:
    """对指定 split 重建消融。"""
    print(f'\n=== {split} ===', flush=True)
    t0 = time.time()

    loop151 = load_loop151_csv(split)
    print(f'  loop151: {len(loop151)} rows', flush=True)

    try:
        primary = load_loop127_primary(split)
        print(f'  primary: {len(primary)} rows', flush=True)
    except FileNotFoundError:
        primary = None

    try:
        r5 = load_loop130_r5(split)
        print(f'  r5_guard: {len(r5)} rows', flush=True)
    except FileNotFoundError:
        r5 = None

    try:
        selector = load_loop136_selector(split)
        print(f'  selector: {len(selector)} rows', flush=True)
    except FileNotFoundError:
        selector = None

    records = []
    errors = {'missing_primary': 0, 'missing_r5': 0, 'missing_selector': 0}

    merged = 0
    for sha, row151 in loop151.items():
        label = int(row151.get('label', -1))
        if label not in (0, 1):
            continue

        # Arm B: primary prediction (stage2 with_logreg, threshold=0.31)
        primary_prob = float(row151.get('stage2_prob_malicious', -1))
        arm_b = int(primary_prob >= PRIMARY_THR)

        # Arm C: loop130 (after R5 content rules)
        # baseline_prob_malicious = loop130 probability
        # If guard_flip in loop130 CSV is True, arm_c = 0 else arm_b
        baseline_prob = float(row151.get('baseline_prob_malicious', primary_prob))
        guard_flip = False
        if r5 and sha in r5:
            guard_flip = r5[sha].get('guard_flip', '').strip().lower() == 'true'
            merged += 1
        arm_c = 0 if guard_flip else arm_b

        # Arm D: loop136 (after noise selector)
        candidate_prob = float(row151.get('candidate_prob_malicious', baseline_prob))
        selector_score = float(row151.get('selector_score', -1)) if row151.get('selector_score') else None
        selector_accept = row151.get('selector_accept_candidate', '').strip().lower()
        noise_pred = int(candidate_prob >= NOISE_THR)
        if selector_accept == 'true':
            arm_d = noise_pred
        else:
            arm_d = arm_c

        # Arm E: final (after signer guard)
        signer_downgrade = row151.get('trusted_signer_guard_downgrade', '').strip().lower() == 'true'
        final_pred = int(row151.get('trusted_signer_guard_prediction', arm_d))
        arm_e = 0 if signer_downgrade else arm_d

        records.append({
            'sha': sha,
            'label': label,
            'primary_prob': round(primary_prob, 6),
            'baseline_prob': round(baseline_prob, 6),
            'candidate_prob': round(candidate_prob, 6),
            'selector_score': round(selector_score, 6) if selector_score is not None else None,
            'guard_flip': guard_flip,
            'selector_accept': selector_accept == 'true',
            'signer_downgrade': signer_downgrade,
            'arm_b': arm_b,
            'arm_c': arm_c,
            'arm_d': arm_d,
            'arm_e': arm_e,
        })

    print(f'  merged r5: {merged}', flush=True)
    print(f'  records: {len(records)}', flush=True)

    # 汇总
    successful = records
    arms_metrics = {}
    for arm_name, arm_col in [('B','arm_b'),('C','arm_c'),('D','arm_d'),('E','arm_e')]:
        tp = sum(1 for r in successful if r['label']==1 and r[arm_col]==1)
        fn = sum(1 for r in successful if r['label']==1 and r[arm_col]==0)
        fp = sum(1 for r in successful if r['label']==0 and r[arm_col]==1)
        tn = sum(1 for r in successful if r['label']==0 and r[arm_col]==0)
        tpr = tp/(tp+fn) if tp+fn else 0.0
        fpr = fp/(fp+tn) if fp+tn else 0.0
        acc = (tp+tn)/(tp+fn+fp+tn) if (tp+fn+fp+tn) else 0.0
        arms_metrics[arm_name] = {'tp':tp,'fn':fn,'fp':fp,'tn':tn,'tpr':tpr,'fpr':fpr,'accuracy':acc,'n':len(successful)}
        print(f'  {arm_name}: TPR={tpr:.4f} FPR={fpr:.4f} Acc={acc:.4f} (TP={tp} FN={fn} FP={fp} TN={tn})', flush=True)

    # 相邻阶段 repairs/breaks
    transitions = {}
    for left, lcol, right, rcol in [('B','arm_b','C','arm_c'),('C','arm_c','D','arm_d'),('D','arm_d','E','arm_e')]:
        repairs = sum(1 for r in successful if r[lcol]!=r['label'] and r[rcol]==r['label'])
        breaks = sum(1 for r in successful if r[lcol]==r['label'] and r[rcol]!=r['label'])
        unchanged = len(successful) - repairs - breaks
        transitions[f'{left}->{right}'] = {'repairs':repairs,'breaks':breaks,'unchanged':unchanged}
        print(f'  {left}->{right}: repairs={repairs} breaks={breaks} unchanged={unchanged}', flush=True)

    elapsed = time.time() - t0
    print(f'  elapsed: {elapsed:.1f}s', flush=True)

    result = {
        'split': split,
        'records': records,
        'arms': arms_metrics,
        'transitions': transitions,
        'errors': errors,
        'elapsed_seconds': round(elapsed, 1),
    }
    return result


if __name__ == '__main__':
    for split in ('test10k', 'full'):
        result = reconstruct_ablation(split)
        path = OUTDIR / f'ablation_{split}.json'
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'  written: {path} ({path.stat().st_size/1024:.0f}KB)', flush=True)
