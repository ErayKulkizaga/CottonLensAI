"""Freeze model_v2.json: model_v1 with the Texas area intervals replaced by the CAL1 choice.
Everything else (momentum, anchoring coefficients) is unchanged. model_v1.json stays as it is.
Usage: python research/monitor/build_model_v2.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    v1 = json.loads((ROOT / 'research/monitor/model_v1.json').read_text(encoding='utf-8'))
    cal = json.loads((ROOT / 'research/evidence/cal1-intervals-20261117.json').read_text(encoding='utf-8'))
    assert cal['chosen'], 'CAL1 chose no method; model_v2 must not be built'
    f = cal['frozen']
    v2 = {**v1, 'version': 'model_v2', 'intervals_tx_area': {
        'q80_log': f['q80_log'], 'q95_log': f['q95_log'], 'method': f['method'],
        'source': 'CAL1 (docs/CAL1_INTERVAL_CALIBRATION_PREREG_20261117.md): real-time coverage 2011-2025 '
                  f"{cal[f['method']]['coverage80']:.2f} (80%), {cal[f['method']]['coverage95']:.2f} (95%)"}}
    (ROOT / 'research/monitor/model_v2.json').write_text(json.dumps(v2, indent=1, sort_keys=True), encoding='utf-8')
    print(json.dumps(v2['intervals_tx_area'], indent=1))


if __name__ == '__main__':
    main()
