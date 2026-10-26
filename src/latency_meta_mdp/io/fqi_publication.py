"""Incremental publication of completed Meta states, independent of GPU training."""

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from latency_meta_mdp.io.fqi_handoff import checkpoint_payload
from latency_meta_mdp.meta.information.config import sha256
from latency_meta_mdp.meta.information.validation import verified_validation_metrics


def _read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def _ancestor_updates(receipt, campaign):
    metrics = receipt.get('metrics') or {}
    if 'ancestor_optimizer_updates' in metrics:
        return metrics['ancestor_optimizer_updates']
    parent = receipt['binding'].get('parent')
    if not parent:
        return 0
    path = Path(parent['path']) / 'complete.json'
    if not path.is_file() and 'campaign' in path.parts:
        path = campaign.joinpath(*path.parts[path.parts.index('campaign') + 1:])
    if not path.is_file():
        return None
    if sha256(path) != parent['receipt_sha256']:
        raise ValueError('checkpoint parent receipt changed')
    return _ancestor_updates(_read(path), campaign)


def checkpoint_rows(campaign):
    """Only atomically completed iteration directories; never aliases or temporary saves."""
    folders = [campaign / 'initial', campaign / 'startup']
    folders += [p / 'fit' for p in sorted(campaign.glob('cycle-[0-9]*'))]
    from latency_meta_mdp.meta.campaign_tracking import read_events

    rows, updates = [], 0
    for folder in folders:
        events = [item['event'] for item in read_events(folder / 'events.jsonl')]
        completed_updates = {e['iteration']: e['actual_update'] for e in events
                             if e['phase'] == 'iteration_complete' and 'actual_update' in e}
        for path in sorted(folder.glob('iteration-[0-9]*/complete.json')):
            receipt = _read(path)
            iteration = receipt['iteration']
            local_updates = 0 if iteration == 0 else completed_updates.get(iteration)
            state = _read(folder / 'validation-state.json', {})
            validation = state.get('completed', {}).get(str(iteration))
            digest = sha256(path)
            if validation and validation['checkpoint_receipt_sha256'] != digest:
                raise ValueError('validation does not belong to this checkpoint')
            if validation:
                verified_validation_metrics(path.parent, folder / 'validation' / path.parent.name,
                                            validation, state['protocol'])
            rows.append(dict(
                checkpoint=str(path.parent.relative_to(campaign)),
                cycle=int(folder.parent.name.split('-')[-1]) if folder.name == 'fit' else 0,
                iteration=iteration,
                total_updates=None if local_updates is None else updates + local_updates,
                ancestor_updates=_ancestor_updates(receipt, campaign),
                state_sha256=receipt['files']['state.pt'], receipt_sha256=digest,
                validation=validation,
            ))
        updates += max((e.get('actual_update', 0) for e in events), default=0)
    return rows


def sync_checkpoints(root, config_path, recipe_path, *, repo_id, run_id, wandb_url=None, api=None):
    """Serialize manual retries and the supervisor's uploader for this run."""
    import fcntl

    lock_path = Path(root) / "telemetry-main/checkpoint-publication.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _sync_checkpoints(root, config_path, recipe_path, repo_id=repo_id,
                                 run_id=run_id, wandb_url=wandb_url, api=api)


def _sync_checkpoints(root, config_path, recipe_path, *, repo_id, run_id, wandb_url=None, api=None):
    """Retryable single-writer sync; completed weights are immutable, index can gain evals.

    Network faults retain local states and a visible pending count. Validation and
    publication configuration errors fail closed. A selected state is not a full
    replay/journal backup.
    """
    from huggingface_hub import HfApi

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('plain run ID required')
    root = Path(root).resolve()
    campaign = root / 'campaign'
    state_path = root / 'telemetry-main/checkpoint-publication.json'
    rows = checkpoint_rows(campaign)
    binding = dict(repo_id=repo_id, run_id=run_id, config_sha256=sha256(config_path),
                   recipe_sha256=sha256(recipe_path))
    state = _read(state_path, dict(binding=binding, published={}))
    if state['binding'] != binding:
        raise ValueError('publication identity changed')
    api = api or HfApi()
    info = api.repo_info(repo_id, repo_type='model')
    if info.private or info.gated != 'manual':
        raise ValueError('checkpoint destination must be public with manual gating')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = None
    error = None
    for row in rows:
        name = row['checkpoint']
        checkpoint = campaign / name
        stat = (checkpoint / 'state.pt').stat()
        fingerprint = [row['receipt_sha256'], stat.st_size, stat.st_mtime_ns]
        acknowledged = state['published'].get(name)
        if acknowledged:
            if acknowledged['fingerprint'] != fingerprint:
                if (acknowledged['fingerprint'][0] != row['receipt_sha256'] or
                        sha256(checkpoint / 'state.pt') != row['state_sha256']):
                    raise ValueError('published checkpoint changed on disk')
                acknowledged['fingerprint'] = fingerprint
            row.update(acknowledged['remote'])
            continue
        files = checkpoint_payload(checkpoint, config_path, recipe_path, allow_live=True)
        digests = {key: sha256(path) for key, path in files.items()}
        prefix = f'runs/{run_id}/checkpoints/{name}'
        try:
            missing = True
            if api.file_exists(repo_id, f'{prefix}/manifest.json', repo_type='model'):
                revision = api.repo_info(repo_id, repo_type='model').sha
                manifest = _read(Path(api.hf_hub_download(
                    repo_id, f'{prefix}/manifest.json', repo_type='model', revision=revision)))
                if manifest != digests:
                    raise ValueError('remote checkpoint label contains different files')
                missing = False
                for key, expected in digests.items():
                    filename = f'{prefix}/{key}'
                    if not api.file_exists(repo_id, filename, repo_type='model', revision=revision):
                        missing = True
                        continue
                    payload = api.hf_hub_download(repo_id, filename, repo_type='model',
                                                  revision=revision)
                    if sha256(Path(payload)) != expected:
                        raise ValueError('remote checkpoint payload differs from manifest')
            if missing:
                with tempfile.TemporaryDirectory(prefix='meta-publish-') as temp:
                    temp = Path(temp)
                    for key, path in files.items():
                        shutil.copy2(path, temp / key)
                    _write(temp / 'manifest.json', digests)
                    result = api.upload_folder(repo_id=repo_id, repo_type='model',
                                               folder_path=temp, path_in_repo=prefix,
                                               commit_message=f'Save {run_id} {name}')
                    revision = result.oid
            remote = dict(hf_revision=revision, hf_path=prefix,
                          hf_url=f'https://huggingface.co/{repo_id}/tree/{revision}/{prefix}')
            state['published'][name] = dict(fingerprint=fingerprint, remote=remote)
            row.update(remote)
            _write(state_path, state)
        except ValueError:
            raise
        except Exception as exc:
            error = type(exc).__name__
            break
    # Include already acknowledged later entries even when an earlier retry failed.
    for row in rows:
        if row['checkpoint'] in state['published']:
            row.update(state['published'][row['checkpoint']]['remote'])
    index = dict(version=1, **binding, git_commit=commit, wandb_url=wandb_url, checkpoints=rows)
    content = (json.dumps(index, indent=2) + '\n').encode()
    import hashlib
    digest = hashlib.sha256(content).hexdigest()
    index_pending = state.get('index_sha256') != digest
    if index_pending:
        try:
            api.upload_file(repo_id=repo_id, repo_type='model', path_or_fileobj=content,
                            path_in_repo=f'runs/{run_id}/index.json',
                            commit_message=f'Index {run_id} checkpoints and validations')
            state['index_sha256'] = digest
            index_pending = False
        except Exception as exc:
            error = type(exc).__name__
    status = dict(
        **index, uploaded=sum('hf_revision' in row for row in rows),
        pending=sum('hf_revision' not in row for row in rows), index_pending=index_pending,
        index_url=f'https://huggingface.co/{repo_id}/blob/main/runs/{run_id}/index.json',
        error=error, checked_at=time.time(),
    )
    _write(state_path, state)
    _write(root / 'telemetry-main/checkpoint-index.json', status)
    return status


class BackgroundPublisher:
    """One bounded worker; network IO cannot delay the learner or console polling."""

    def __init__(self, root, config, recipe, *, repo_id, run_id):
        from concurrent.futures import ThreadPoolExecutor

        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-upload')
        self.args = (root, config, recipe)
        self.kwargs = dict(repo_id=repo_id, run_id=run_id)
        self.future = None
        self.last_submit = -60.0
        self.final_submitted = False
        self.status = {}

    def poll(self, *, wandb_url=None, final=False):
        if self.future is not None and self.future.done():
            try:
                self.status = self.future.result()
            except Exception as exc:
                self.status = dict(self.status, error=type(exc).__name__)
            self.future = None
        now = time.monotonic()
        if self.future is None and (
            (final and not self.final_submitted) or (not final and now - self.last_submit >= 60)
        ):
            self.future = self.pool.submit(sync_checkpoints, *self.args, **self.kwargs,
                                           wandb_url=wandb_url)
            self.last_submit = now
            self.final_submitted = final
        return self.future is not None

    def close(self):
        self.pool.shutdown(wait=True)
