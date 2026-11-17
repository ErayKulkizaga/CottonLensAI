"""Host tests only assert refusal; they never train or claim GPU validation."""
import a100_preflight as probe
import pytest


def test_local_probe_fails_before_device_query_or_output(tmp_path, monkeypatch):
    def deny():
        raise RuntimeError('Colab required')
    monkeypatch.setattr(probe, 'require_colab_training', deny)
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **k: pytest.fail('Host must not query GPU'))
    target = tmp_path/'probe'
    with pytest.raises(RuntimeError, match='Colab required'):
        probe.run(target)
    assert not target.exists()


def test_wrong_gpu_rejected_before_tensorflow_import_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'require_colab_training', lambda: None)
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **k: 'Tesla T4, 16384 MiB, fixture')
    target = tmp_path/'probe'
    with pytest.raises(RuntimeError, match='A100 required'):
        probe.run(target)
    assert not target.exists()


@pytest.mark.parametrize('gpu,hardware', [('A100', 'NVIDIA A100'), ('L4', 'NVIDIA L4'),
                                        ('G4', 'NVIDIA RTX PRO 6000 Blackwell Server Edition')])
def test_incomplete_probe_preserved_without_retry(tmp_path, monkeypatch, gpu, hardware):
    monkeypatch.setattr(probe, 'require_colab_training', lambda: None)
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **k: hardware+', 40960 MiB, fixture')
    (tmp_path/'partial.txt').write_text('preserve')
    with pytest.raises(ValueError, match='Incomplete probe preserved'):
        probe.run(tmp_path, expected_gpu=gpu)
    assert (tmp_path/'partial.txt').read_text() == 'preserve'
