import io
import json
import zipfile

import pytest

from research.verify_gpu_restart import native, record, semantic_config, sha


def test_record_tamper_and_valid_control():
    value = {'state': 'generated-fixture', 'optimizer_step': 2}
    identity = sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
    data = json.dumps({**value, 'record_id': identity}).encode()
    assert record(data)['optimizer_step'] == 2
    with pytest.raises(ValueError, match='identity mismatch'):
        record(json.dumps({**value, 'optimizer_step': 3, 'record_id': identity}).encode())
    assert record(data)['optimizer_step'] == 2


def test_only_keras_object_alias_id_is_ignored():
    before = {'optimizer': {'learning_rate': .001}, 'policy': {'shared_object_id': 123, 'name': 'float32'}}
    after = {'optimizer': {'learning_rate': .001}, 'policy': {'name': 'float32'}}
    assert semantic_config(before) == semantic_config(after)
    after['optimizer']['learning_rate'] = .002
    assert semantic_config(before) != semantic_config(after)


def test_parent_external_hdf5_link_rejected_before_traversal(tmp_path):
    h5py = pytest.importorskip('h5py')
    state = io.BytesIO()
    with h5py.File(state, 'w') as f:
        f['layers'] = h5py.ExternalLink(str(tmp_path/'must-not-open.h5'), '/')
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('metadata.json', '{}')
        z.writestr('config.json', '{}')
        z.writestr('model.weights.h5', state.getvalue())
    with pytest.raises(TypeError, match='External/soft'):
        native(archive.getvalue())
