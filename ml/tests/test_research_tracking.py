from types import SimpleNamespace

import pytest
from cottonlens_ml.research.tracking import project_record


class Client:
    def __init__(self):
        self.runs, self.params, self.metrics = [], {}, {}

    def search_runs(self, *args, **kwargs):
        return self.runs

    def create_run(self, experiment, tags):
        run = SimpleNamespace(info=SimpleNamespace(run_id='one'), data=SimpleNamespace(tags=tags))
        self.runs.append(run)
        return run

    def log_param(self, run, key, value):
        self.params[key] = value

    def log_metric(self, run, key, value, step):
        self.metrics[key] = value

    def set_tag(self, run, key, value):
        self.runs[0].data.tags[key] = value

    def set_terminated(self, *args, **kwargs):
        pass


def test_projection_idempotent_and_flat_numeric_metrics_only():
    client = Client()
    record = {'experiment_id': 'a' * 64, 'identity': {'data': 'x'},
              'specification': {'role': 'inner', 'recipe': {'family': 'xgboost', 'seed': 42}},
              'result': {'metrics': {'mae': 1., 'nested': {}, 'invalid': float('nan')}},
              'payload_root': 'payloads/abc'}
    assert project_record(client, 'exp', record) == project_record(client, 'exp', record)
    assert len(client.runs) == 1 and client.metrics['mae'] == 1.
    assert 'invalid' not in client.metrics and 'nested' not in client.metrics
    assert client.params['family'] == 'xgboost'
    client.runs.append(client.runs[0])
    with pytest.raises(ValueError, match='Duplicate'):
        project_record(client, 'exp', record)
