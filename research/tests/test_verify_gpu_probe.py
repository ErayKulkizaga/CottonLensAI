import io
import zipfile

import pytest

from research.verify_gpu_probe import members


def test_probe_member_whitelist_and_content():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('complete.json', '{}')
    assert members(buffer.getvalue(), ['complete.json']) == {'complete.json': b'{}'}
    with pytest.raises(ValueError, match='Unexpected'):
        members(buffer.getvalue(), ['different.json'])


def test_duplicate_probe_members_are_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('complete.json', '{}')
        with pytest.warns(UserWarning, match='Duplicate'):
            z.writestr('complete.json', '{}')
    with pytest.raises(ValueError, match='duplicate'):
        members(buffer.getvalue(), ['complete.json'])
