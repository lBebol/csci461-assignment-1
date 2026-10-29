#!/usr/bin/env python3
"""Check exported evidence, text preservation, leakage, and measured predictions.

Uses only the standard library so the same artifact checks can run on the host.
This does not replace executing the notebook inside Docker.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIGURES = [
    'correlation_heatmap.png', 'word_frequency.png',
    'group_comparison.png', 'confusion_matrices.png',
]


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def verify(output, notebook):
    raw = read_csv(ROOT / 'data/dataset.csv')
    cleaned = read_csv(output / 'cleaned_data.csv')
    predictions = read_csv(output / 'test_predictions.csv')
    summary = json.loads((output / 'run_summary.json').read_text())
    assert summary['executed_in_container'] is True
    assert summary['dataset_sha256'] == hashlib.sha256((ROOT / 'data/dataset.csv').read_bytes()).hexdigest()
    assert len(cleaned) == summary['cleaned_rows']
    assert len({row['COMMENT_ID'] for row in cleaned}) == len(cleaned)
    assert summary['raw_rows'] - summary['exact_duplicates'] - summary['invalid_rows'] - summary['empty_cleaned_rows'] == len(cleaned)
    groups = {}
    counts = {'train': 0, 'test': 0}
    for row in cleaned:
        original = raw[int(row['source_row'])]
        assert row['CONTENT'] == original['CONTENT'], 'Original text was changed.'
        assert row['COMMENT_ID'] == original['COMMENT_ID']
        assert row['label'] == original['CLASS']
        assert row['cleaned_text'].strip()
        assert int(row['char_count']) == len(original['CONTENT'])
        assert 0 < int(row['unique_word_count']) <= int(row['word_count'])
        assert int(row['word_count']) == len(row['cleaned_text'].split())
        assert math.isfinite(float(row['avg_word_length']))
        groups.setdefault(row['cleaned_text'], set()).add(row['split'])
        counts[row['split']] += 1
    assert all(len(splits) == 1 for splits in groups.values()), 'Repeated-text leakage across train/test.'
    assert counts == {'train': summary['train_rows'], 'test': summary['test_rows']}
    assert len(predictions) == counts['test']
    test_rows = {row['COMMENT_ID']: row for row in cleaned if row['split'] == 'test'}
    assert {row['COMMENT_ID'] for row in predictions} == set(test_rows)
    for row in predictions:
        assert row['label'] == test_rows[row['COMMENT_ID']]['label']
        assert row['source_row'] == test_rows[row['COMMENT_ID']]['source_row']

    # Independently derive reported scores from the exported held-out predictions.
    for model in summary['models']:
        name = model['model']
        assert {row[name] for row in predictions} <= {'0', '1'}
        tp = sum(row['label'] == row[name] == '1' for row in predictions)
        tn = sum(row['label'] == row[name] == '0' for row in predictions)
        fp = sum(row['label'] == '0' and row[name] == '1' for row in predictions)
        fn = sum(row['label'] == '1' and row[name] == '0' for row in predictions)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1_spam = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        f1_ham = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0
        derived = {'accuracy': (tp + tn) / len(predictions), 'precision_spam': precision,
                   'recall_spam': recall, 'f1_spam': f1_spam, 'f1_macro': (f1_spam + f1_ham) / 2}
        for metric, value in derived.items():
            assert math.isclose(value, model[metric], abs_tol=1e-12), f'{name}: {metric} mismatch'

    cluster_counts = {}
    for row in cleaned:
        cluster = int(row['cluster'])
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    for cluster in summary['clusters']:
        assert cluster_counts[cluster['cluster']] == cluster['records']
        assert cluster['top_terms'] and cluster['interpretation'] and cluster['representatives']
    assert cluster_counts.get(-1, 0) == summary['unclustered_rows']
    assert summary['clustered_rows'] + summary['unclustered_rows'] == len(cleaned)
    assert len(summary['clusters']) == summary['selected_k']
    assert -1 <= summary['silhouette'] <= 1

    for name in REQUIRED_FIGURES:
        content = (output / 'figures' / name).read_bytes()
        assert content.startswith(b'\x89PNG\r\n\x1a\n') and len(content) > 1000
    for name in ['model_metrics.txt', 'cluster_summary.txt', 'text_insights.md']:
        assert (output / name).stat().st_size > 100
    nb = json.loads(notebook.read_text())
    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    assert code_cells and all(cell['execution_count'] is not None for cell in code_cells)
    assert [cell['execution_count'] for cell in code_cells] == list(range(1, len(code_cells) + 1))
    assert all(item['output_type'] != 'error' for cell in code_cells for item in cell['outputs'])
    assert any('SUCCESS: all analysis' in ''.join(item.get('text', []))
               for cell in code_cells for item in cell['outputs'])
    print(f'PASS: {len(cleaned)} rows; original text preserved; no repeated-text leakage;')
    print(f'      {len(predictions)} held-out predictions match reported metrics; clusters and exports reconcile;')
    print(f'      {len(code_cells)} sequentially executed code cells; required figures and written insights exist.')
    return summary


def compare(output, reference):
    filenames = ['cleaned_data.csv', 'test_predictions.csv', 'model_metrics.txt', 'cluster_summary.txt',
                 'model_comparison.csv', 'sampling_distributions.csv', 'k_experiment.csv',
                 'text_insights.md', 'run_summary.json']
    for name in filenames:
        assert (output / name).read_bytes() == (reference / name).read_bytes(), f'Reproduction differs: {name}'
    print(f'PASS: {len(filenames)} deterministic artifacts exactly match the reference run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--notebook', type=Path, default=ROOT / 'notebook/analysis.ipynb')
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    verify(args.output, args.notebook)
    if args.reference:
        compare(args.output, args.reference)


if __name__ == '__main__':
    main()
