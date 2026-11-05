"""Prepare the successor to the interrupted Cotton forward study before its first decision."""

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record

from research import ct_forward_daily as daily
from research import ct_manual_forward as forward
from research.ct_forward_backup import backup
from research.storage import restore


def prepare(repo, old, study, old_runtime):
    old_ready, old_design = forward.verify(old, old/'source-snapshot')
    if study.exists():
        raise ValueError('Preserve an existing study or interrupted preparation')
    first = old/'captures/20261031T000522224911Z'
    second = old/'captures/20261103T000526258866Z'
    inputs = study/'inputs'
    bootstrap = inputs/'bootstrap'
    bootstrap.mkdir(parents=True)
    for src, dest in (
        (old/'inputs/seed.parquet', bootstrap/'parent-seed.parquet'),
        (first/'chart.json', bootstrap/'first-chart.json'),
        (first/'http-receipt.json', bootstrap/'first-receipt.json'),
        (second/'chart.json', inputs/'initial-chart.json'),
        (second/'http-receipt.json', inputs/'initial-receipt.json'),
    ):
        shutil.copyfile(src, dest)
    shutil.copytree(old/'calendar-extensions', bootstrap/'parent-calendar-extensions')
    calendar = forward.calendar_body(old)
    (inputs/'calendar.json').write_text(json.dumps(calendar, indent=2)+'\n', encoding='utf-8')
    seed, revisions = forward.replay_first_observed_seed(study)
    previous, _ = forward.current_history(old)
    pd.testing.assert_frame_equal(seed.iloc[:len(previous)].reset_index(drop=True), previous, check_exact=True)
    if str(seed.date.iloc[-1].date()) != '2026-11-02' or len(seed) != len(previous)+1:
        raise ValueError('Expected the verified November 2 bar after the unchanged parent history')
    seed.to_parquet(inputs/'seed.parquet', index=False)
    freeze_record(inputs/'bootstrap-revisions.json', {'transitions': revisions})
    freeze_record(inputs/'parent-study.json', {'identity': old_ready['identity'],
        'cohort_lock_sha256': digest(old/'cohort-lock.json'),
        'status_at_preparation': forward.status(old, old/'source-snapshot')})
    shutil.copytree(old/'native-learning', study/'native-learning')
    shutil.copyfile(old/'human-authorization.json', study/'human-authorization.json')
    shutil.copyfile(old/'history-check.json', study/'history-check.json')
    design = {**old_design, 'profile': 'ct-own-level-first-observed-forward-t1-v2',
              'overlap_price_policy': 'retain-first-observed-v1',
              'source': 'Yahoo CT=F proxy. Each date keeps the value first admitted to the seed or from a timely snapshot; later conflicting quotes are archived.',
              'deliberate_repeat_reason': 'Fresh prospective cohort after a real source revision prevented publication in the original locked cohort. Same learner and EWMA comparison with separate fit identities.'}
    manifest = forward.source(repo)
    source = study/'source-snapshot'
    for name in {**manifest['ml']['files'], **manifest['helpers']}:
        dest = source/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, dest)
    identity = {'profile': design['profile'], 'source_id': manifest['source_id'], 'design_id': content_id(design),
                'seed_sha256': digest(inputs/'seed.parquet'), **forward.environment()}
    frozen = {p.relative_to(study).as_posix(): digest(p) for directory in ('inputs', 'native-learning')
              for p in (study/directory).rglob('*') if p.is_file()}
    freeze_record(study/'source-manifest.json', manifest)
    freeze_record(study/'decision-contract.json', design)
    freeze_record(study/'learning-control.json', forward.native_learning(study/'native-learning', source))
    freeze_record(study/'ready.json', {'identity': identity, 'frozen_hashes': frozen})
    forward.verify(study, source)
    daily.verify_core_before_import(study, source, identity['source_id'])

    execution = daily.verify_execution(old_runtime, old_runtime.name)
    execution = {k: v for k, v in execution.items() if k not in ('execution_source_id', 'repository_commit')}
    execution.update(core_source_id=identity['source_id'], source_policy=design['overlap_price_policy'])
    eid = daily.fingerprint(execution)
    runtime = old_runtime.parent/eid
    shutil.copytree(old_runtime, runtime)
    (runtime/'execution-manifest.json').write_text(json.dumps({**execution, 'execution_source_id': eid}, indent=2)+'\n', encoding='utf-8')
    daily.verify_execution(runtime, eid)
    return identity, design, runtime, {**execution, 'execution_source_id': eid, 'runtime_path': str(runtime)}


def restore_before_lock(study, identity, design, execution):
    run_id = uuid.uuid4().hex
    delivered = backup(study, execution, run_id, 'manual')
    fresh = study/'backups/attempts'/run_id/'fresh-download'
    archive, manifest = [fresh/name for name in delivered['assets']]
    recovered = fresh/'restored'
    restore(archive, manifest, recovered)
    for name, sha in delivered['source_files'].items():
        if digest(recovered/'bundle/study'/name) != sha:
            raise ValueError('Private restoration differs from the prepared study')
    verified = {'ready_sha256': digest(study/'ready.json'), 'source_id': identity['source_id'],
                'design_id': identity['design_id'], 'fit_budget': design['fit_budget'],
                'fresh_restore_verified': True, 'private_archive_sha256': delivered['archive_sha256'],
                'backup_id': delivered['backup_id'], 'human_authorization_sha256': digest(study/'human-authorization.json'),
                'scope': 'Successor source revision repair requested by the user before 03:05 TR'}
    freeze_record(study/'authorization.json', verified)


if __name__ == '__main__':
    base = Path('C:/Users/Eray/SEZAR/PROJECTS/CottonLensAI/output')
    new = base/'own-level-forward-v2-first-observed'
    old_execution = base/'manual-risk-runtime/2747450ff4d46b022493cc604c565aa941e0378c5c3bd741179d962cff04a644'
    identity, design, runtime, execution = prepare(Path.cwd(), base/'own-level-forward-v1', new, old_execution)
    print(json.dumps({'stage': 'prepared', 'study': str(new), 'runtime': str(runtime), **identity}), flush=True)
    restore_before_lock(new, identity, design, execution)
    source = new/'source-snapshot'
    lock = forward.lock(new, source, new/'authorization.json')
    if lock['first_origin'] != '2026-11-03':
        raise ValueError('Tonight’s planned first origin was missed')
    print(json.dumps({'stage': 'locked', 'first_origin': lock['first_origin']}), flush=True)
    result = forward.fit_for_origin(new, source, new/'authorization.json', lock['first_origin'])
    freeze_record(new/'initial-state-receipt.json', {'result': result, 'first_origin': lock['first_origin'],
                  'source_id': identity['source_id'], 'completed_at': forward.now().isoformat()})
    print(json.dumps({'stage': 'state_ready', 'runtime': str(runtime), **identity, **result}), flush=True)
