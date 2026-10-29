"""Bounded manual capture of archived USDA PDF copies for contract-identity review."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from research.weekly_named_source import printed_date


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(plan_path, root, sample):
    from PIL import Image
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError
    plan = json.loads(plan_path.read_text()); days = plan['candidate_publication_dates']
    if (days != sorted(set(days)) or len(days) > plan['max_files'] or plan['max_files'] > 40
            or any(date.fromisoformat(d).isoformat() != d for d in days)
            or plan['max_file_bytes'] != 5000000 or plan['max_total_bytes'] != 150000000
            or plan['new_fits'] != 0 or not plan['manual_one_off']):
        raise ValueError('Fixed bounded historical PDF source plan required')
    identity = {'plan_sha256': sha(plan_path), 'source_sha256': sha(Path(__file__)),
                'sample_sha256': sha(sample), 'new_fits': 0, 'scheduler_enabled': False}
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root/'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Preserve capture identity; source or plan differs')
    else:
        with identity_path.open('x') as f:
            json.dump(identity, f, indent=2)
    used = 0; deadline = time.monotonic()+15*60
    for day in days:
        folder = root/'reports'/day; receipt = folder/'receipt.json'; pdf = folder/'report.pdf'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            for name, expected in saved['files'].items():
                if sha(folder/name) != expected:
                    raise ValueError('Captured PDF/review artifact changed; no refetch')
            used += saved.get('bytes', 0); continue
        if time.monotonic() >= deadline:
            return {'status': 'checkpoint_pause', 'bytes': used}
        if folder.exists():
            raise ValueError('Unreceipted partial capture; preserve and inspect')
        folder.mkdir(parents=True)
        url = 'https://s3.amazonaws.com/barchart-news-store-prod/external_links/USDA/'+day.replace('-', '')+'-cnwwcmr.pdf'
        result = {'url': url, 'expected_publication_date': day, 'attempted_at': datetime.now(UTC).isoformat(),
                  'historical_availability_verified': False, 'first_vintage_verified': False,
                  'training_allowed': False, 'publisher': 'USDA AMS', 'host_role': 'archived mirror copy'}
        try:
            if day == '2018-08-03':
                data = sample.read_bytes(); result['reused_verified_sample'] = True
            else:
                with urllib.request.urlopen(url, timeout=20) as response:
                    if urlsplit(response.url).hostname != 's3.amazonaws.com':
                        raise ValueError('Unapproved source redirect')
                    data = response.read(plan['max_file_bytes']+1)
            if len(data) > plan['max_file_bytes'] or used+len(data) > plan['max_total_bytes']:
                raise ValueError('Frozen PDF byte budget exceeded')
            if not data.startswith(b'%PDF'):
                raise ValueError('Not an archived PDF')
            pdf.write_bytes(data); used += len(data); result['bytes'] = len(data)
            reader = PdfReader(pdf); first = reader.pages[0]; stamp = printed_date(first.extract_text())
            result.update(printed_publication_date=stamp.isoformat(), pages=len(reader.pages),
                          status='candidate_for_value_identity_review' if stamp.isoformat() == day else 'wrong_printed_publication')
            images = [i for i in first.images if i.image.width > 500 and 300 < i.image.height < i.image.width]
            if images:
                image = max(images, key=lambda i: i.image.width*i.image.height)
                (folder/'embedded-image.bin').write_bytes(image.data)
                image.image.save(folder/'table.png')
                w, h = image.image.size; box = (0, int(h*.44), int(w*.80), int(h*.73))
                crop = image.image.crop(box); scale = min(3., 1800/crop.width)
                crop.resize((int(crop.width*scale), int(crop.height*scale)), Image.Resampling.LANCZOS).save(folder/'futures-review.png')
                result.update(image_name=image.name, image_size=[w, h], preview_crop=list(box),
                              preview_method='heuristic crop for visual review only; no automatic cell admission')
        except (ValueError, OSError, urllib.error.URLError, PyPdfError) as error:
            result.update(status='not_verified', error=f'{type(error).__name__}: {error}')
        result['files'] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        with receipt.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2); f.write('\n')
        print(json.dumps({'date': day, 'status': result['status'], 'bytes': used}), flush=True)
        if result.get('error') == 'Frozen PDF byte budget exceeded':
            raise ValueError('Stop at frozen source budget')
    rows = [json.loads(p.read_text()) for p in sorted((root/'reports').glob('*/receipt.json'))]
    return {'status': 'completed_source_capture', 'planned': len(days), 'attempted': len(rows),
            'candidates': sum(r['status'] == 'candidate_for_value_identity_review' for r in rows),
            'bytes': used, 'new_fits': 0, 'training_allowed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'root', 'sample'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(capture(a.plan, a.root, a.sample)))
