import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_optional_prediction import optional_config

from latency_meta_mdp.io.fqi_publication import checkpoint_rows, sync_checkpoints
from latency_meta_mdp.meta.information.config import identity_hash, sha256
from latency_meta_mdp.meta.information.validation import validation_metrics


class Hub:
    def __init__(self, root):
        self.root = root
        self.gated = 'manual'
        self.uploads = 0
        self.fail = False

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=False, gated=self.gated, sha='revision')

    def file_exists(self, repo_id, filename, **kwargs):
        return (self.root / filename).exists()

    def hf_hub_download(self, repo_id, filename, **kwargs):
        return str(self.root / filename)

    def upload_folder(self, *, folder_path, path_in_repo, **kwargs):
        if self.fail:
            raise ConnectionError('simulated outage')
        for f in Path(folder_path).iterdir():
            p = self.root / path_in_repo / f.name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f.read_bytes())
        self.uploads += 1
        return SimpleNamespace(oid=f'rev{self.uploads}')

    def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
        if self.fail:
            raise ConnectionError('simulated outage')
        p = self.root / path_in_repo
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(path_or_fileobj)
        return SimpleNamespace(oid='index-rev')


def campaign(tmp_path):
    cfg = optional_config()
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    recipe = tmp_path / 'recipe.json'
    recipe.write_text('{}')
    root = tmp_path / 'run'
    c = root / 'campaign'
    c.mkdir(parents=True)
    (c / 'campaign-state.json').write_text(json.dumps(dict(
        binding=dict(config_sha256=sha256(config), recipe={}, code_hash='code'), phases={})))
    for iteration in (1, 2):
        p = c / f'startup/iteration-{iteration:04d}'
        p.mkdir(parents=True)
        (p / 'state.pt').write_bytes(f'weights {iteration}'.encode())
        (p / 'complete.json').write_text(json.dumps(dict(
            files={'state.pt': sha256(p / 'state.pt')}, iteration=iteration,
            metrics=dict(optimizer_updates=64, ancestor_optimizer_updates=iteration * 64),
            binding=dict(config_hash=identity_hash(cfg), schema_version=3,
                         experiment_id=cfg['experiment_id'], trainer_code_hash='code',
                         initialization=dict(kind='scratch')))))
    (c / 'startup/events.jsonl').write_text(''.join(
        json.dumps(dict(event=dict(phase='iteration_complete', iteration=i,
                                   actual_update=i * 64))) + '\n' for i in (1, 2)))
    return root, config, recipe


def test_publish_every_complete_state_and_late_exact_validation(tmp_path):
    root, config, recipe = campaign(tmp_path)
    hub = Hub(tmp_path / 'hub')
    def sync():
        return sync_checkpoints(root, config, recipe, repo_id='owner/model',
                                run_id='run1', wandb_url='https://wandb.test/run1', api=hub)
    first = sync()
    assert (first['uploaded'], first['pending']) == (2, 0)
    assert hub.uploads == 2
    assert sync()['uploaded'] == 2
    assert hub.uploads == 2
    # Recover an upload whose local acknowledgement was lost.
    (root / 'telemetry-main/checkpoint-publication.json').unlink()
    assert sync()['uploaded'] == 2
    assert hub.uploads == 2
    c = root / 'campaign/startup'
    episodes = [dict(
        episode=dict(scene=scene, regime=regime, behavior='fqi', epsilon=0, split='validation'),
        report=dict(task_reward=0.4, terminated=True, truncated=False, starvation_ticks=0,
                    parcels=dict(spawned=10, successes=4, misses=6, timeouts=0, active=0)))
        for scene in (1, 2, 3) for regime in ('zero', 'nominal', 'family')]
    summary = c / 'validation/iteration-0002/rollout-summary.json'
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({'episodes': episodes}))
    validation = dict(protocol={'scenes': [1, 2, 3]}, completed={'2': dict(
        checkpoint_receipt_sha256=sha256(c / 'iteration-0002/complete.json'),
        evaluation_sha256=sha256(summary), metrics=validation_metrics(episodes, scenes=[1, 2, 3]))})
    (c / 'validation-state.json').write_text(json.dumps(validation))
    result = sync()
    rows = result['checkpoints']
    assert rows[0]['validation'] is None
    assert rows[1]['validation']['metrics']['zero/reward'] == pytest.approx(0.4)
    assert rows[1]['total_updates'] == 128
    index = json.loads((hub.root / 'runs/run1/index.json').read_text())
    assert index['checkpoints'][1]['validation'] == rows[1]['validation']
    assert hub.uploads == 2
    validation['completed']['2']['checkpoint_receipt_sha256'] = 'wrong'
    (c / 'validation-state.json').write_text(json.dumps(validation))
    with pytest.raises(ValueError, match='validation'):
        sync()
    validation['completed']['2']['checkpoint_receipt_sha256'] = sha256(
        c / 'iteration-0002/complete.json')
    validation['completed']['2']['metrics']['zero/reward'] = 0.99
    (c / 'validation-state.json').write_text(json.dumps(validation))
    with pytest.raises(ValueError):
        sync()


def test_failed_upload_stays_pending_and_gate_is_required(tmp_path):
    root, config, recipe = campaign(tmp_path)
    hub = Hub(tmp_path / 'hub')
    hub.fail = True
    result = sync_checkpoints(root, config, recipe, repo_id='owner/model', run_id='run1', api=hub)
    assert result['pending'] == 2
    assert result['uploaded'] == 0
    assert result['error'] == 'ConnectionError'
    assert len(list(root.glob('campaign/startup/iteration-*/state.pt'))) == 2
    hub.fail = False
    hub.gated = False
    with pytest.raises(ValueError, match='manual'):
        sync_checkpoints(root, config, recipe, repo_id='owner/model', run_id='run1', api=hub)


def test_moved_checkpoint_mtime_is_not_corruption_but_changed_bytes_are(tmp_path):
    import os
    root, config, recipe = campaign(tmp_path)
    hub = Hub(tmp_path / 'hub')
    def sync():
        return sync_checkpoints(root, config, recipe, repo_id='owner/model', run_id='run1', api=hub)
    sync()
    path = root / 'campaign/startup/iteration-0001/state.pt'
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000))
    assert sync()['pending'] == 0
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        sync()


def test_index_counts_reexecuted_updates_and_phase_start_lineage(tmp_path):
    root, _, _ = campaign(tmp_path)
    c = root / 'campaign'
    events = c / 'startup/events.jsonl'
    events.write_text(events.read_text().replace('"actual_update": 128', '"actual_update": 150'))
    parent = c / 'startup/iteration-0002'
    start = c / 'cycle-0001/fit/iteration-0000'
    start.mkdir(parents=True)
    (start / 'complete.json').write_text(json.dumps(dict(
        iteration=0, metrics=None, files={'state.pt': 'copied'},
        binding=dict(parent=dict(path=str(parent),
                                 receipt_sha256=sha256(parent / 'complete.json'))))))
    rows = checkpoint_rows(c)
    assert [r['total_updates'] for r in rows] == [64, 150, 150]
    assert rows[-1]['ancestor_updates'] == 128
    # A complete state may precede the durable iteration-complete event.
    events.write_text(events.read_text().splitlines()[0] + '\n' + '{"partial":')
    assert checkpoint_rows(c)[1]['total_updates'] is None


def test_lost_ack_repairs_absent_payload_and_rejects_changed_payload(tmp_path):
    root, config, recipe = campaign(tmp_path)
    hub = Hub(tmp_path / 'hub')
    def sync():
        return sync_checkpoints(root, config, recipe, repo_id='owner/model', run_id='run1', api=hub)
    sync()
    ack = root / 'telemetry-main/checkpoint-publication.json'
    payload = hub.root / 'runs/run1/checkpoints/startup/iteration-0001/state.pt'
    ack.unlink()
    payload.unlink()
    assert sync()['pending'] == 0
    assert payload.is_file()
    assert hub.uploads == 3
    ack.unlink()
    payload.write_bytes(b'other weights')
    with pytest.raises(ValueError, match='remote.*payload'):
        sync()


def test_startup_initial_state_follows_scratch_initialization(tmp_path):
    root, _, _ = campaign(tmp_path)
    c = root / 'campaign'
    initial = c / 'initial/iteration-0000'
    initial.mkdir(parents=True)
    receipt = dict(iteration=0, metrics=None, files={'state.pt': 'initial'},
                   binding=dict(parent=None))
    (initial / 'complete.json').write_text(json.dumps(receipt))
    startup = c / 'startup/iteration-0000'
    startup.mkdir()
    receipt['binding']['parent'] = dict(path=str(initial),
                                       receipt_sha256=sha256(initial / 'complete.json'))
    (startup / 'complete.json').write_text(json.dumps(receipt))
    assert [r['ancestor_updates'] for r in checkpoint_rows(c)[:2]] == [0, 0]
