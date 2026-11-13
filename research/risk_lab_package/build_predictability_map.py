"""Summarize completed price experiments in research/registry.json into a predictability map (zero fit)."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_LABEL = {'weather': 'Hava (NASA POWER)', 'nass': 'USDA NASS ürün durumu', 'wasde-text': 'WASDE metni',
                'wasde': 'WASDE sayısal', 'cftc': 'CFTC pozisyonları', 'fx': 'Döviz', 'crop': 'Mısır/soya fiyatları',
                'ams': 'USDA AMS spot/kontrat', 'fas': 'FAS ihracat satışları', 'oncall': 'On-call', 'fundamental': 'Temel veri birleşimi',
                'agri-transfer': 'Tarım piyasaları transfer', 'return-path': 'Fiyat yolu (nonlinear)',
                'availability-clock': 'DXY/WTI zamanlama', 'price-only': 'Yalnız fiyat geçmişi', 'recency': 'Yakın dönem ağırlığı'}


def label(profile):
    for k, v in SOURCE_LABEL.items():
        if profile.startswith(k):
            return v
    return re.sub(r'-v\d+$', '', profile)


def main():
    reg = ROOT / 'research/registry.json'
    recs = json.loads(reg.read_text(encoding='utf-8'))['records']
    rows = []
    for x in recs:
        m = x.get('metrics') or {}
        sel, raw = m.get('selected') or {}, m.get('raw') or {}
        if 'gain_pct' not in sel and 'gain_pct' not in raw:
            continue
        rows.append({'profile': x.get('profile'), 'source': label(x.get('profile') or ''), 'group': x.get('group'),
                     'horizon': x.get('horizon'), 'families': x.get('model_families'), 'origins': m.get('count'),
                     'selected_gain_pct': sel.get('gain_pct'), 'raw_gain_pct': raw.get('gain_pct'),
                     'raw_direction_pct': raw.get('all_origin_direction_pct'), 'classification': x.get('classification')})
    sel = [r['selected_gain_pct'] for r in rows if r['selected_gain_pct'] is not None]
    out = {'source_registry_sha256': hashlib.sha256(reg.read_bytes()).hexdigest(),
           'experiments': len(rows), 'selected_positive': sum(v > 0 for v in sel), 'selected_ge_5pct': sum(v >= 5 for v in sel),
           'max_selected_gain_pct': max(sel), 'rows': rows}
    dest = ROOT / 'research/risk_lab_package/predictability_map.json'
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(out['experiments'], out['selected_positive'], out['selected_ge_5pct'], out['max_selected_gain_pct'])


if __name__ == '__main__':
    main()
