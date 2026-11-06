import gpu_session_probe as probe
import pytest
from cottonlens_ml.sprint import freeze_record


def prepared(root):
    checkpoint = root/'step.keras'
    checkpoint.write_bytes(b'generated fixture; not a model')
    freeze_record(root/'prepared.json', {'source_sha256': 'source', 'boot_identity': 'old-boot',
                                        'files': {'step.keras': probe.sha(checkpoint)}})


def test_local_denial_precedes_boot_query_and_files(tmp_path, monkeypatch):
    def deny(gpu):
        raise RuntimeError('Colab required')
    monkeypatch.setattr(probe, 'device', deny)
    monkeypatch.setattr(probe, 'boot_identity', lambda: pytest.fail('Unexpected boot query'))
    with pytest.raises(RuntimeError, match='Colab required'):
        probe.run(tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_resume_requires_new_vm_and_missing_local_marker(tmp_path):
    prepared(tmp_path)
    assert probe.phase(tmp_path, 'new-boot', False, 'source') == 'resume'
    for boot, marker in [('old-boot', False), ('new-boot', True), ('old-boot', True)]:
        with pytest.raises(ValueError, match='Delete the Colab runtime'):
            probe.phase(tmp_path, boot, marker, 'source')
    with pytest.raises(ValueError, match='source changed'):
        probe.phase(tmp_path, 'new-boot', False, 'changed')


def test_corrupt_checkpoint_and_incomplete_attempt_never_retrain(tmp_path):
    assert probe.phase(tmp_path, 'boot', False, 'source') == 'prepare'
    freeze_record(tmp_path/'started.json', {'generated': True})
    with pytest.raises(ValueError, match='Incomplete'):
        probe.phase(tmp_path, 'boot', False, 'source')
    prepared(tmp_path)
    (tmp_path/'step.keras').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Corrupt'):
        probe.phase(tmp_path, 'new-boot', False, 'source')


def test_completed_cache_checks_payload_and_does_not_require_new_session(tmp_path):
    path = tmp_path/'model.keras'
    path.write_bytes(b'generated fixture')
    freeze_record(tmp_path/'complete.json', {'source_sha256': 'source', 'files': {'model.keras': probe.sha(path)}})
    assert probe.phase(tmp_path, 'same-boot', True, 'source') == 'cached'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Corrupt'):
        probe.phase(tmp_path, 'same-boot', True, 'source')
