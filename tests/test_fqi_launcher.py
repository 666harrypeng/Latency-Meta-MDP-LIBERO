"""The portable launcher must consume the project-restored model cache."""

import json
import os
import shutil
import subprocess


def test_launcher_uses_project_cache_despite_inherited_host_defaults(tmp_path):
    (tmp_path / "scripts").mkdir()
    launcher = tmp_path / "scripts/launch_optional_fqi.sh"
    shutil.copy2("scripts/launch_optional_fqi.sh", launcher)
    interpreter = tmp_path / "capture-python"
    interpreter.write_text(
        "#!/usr/bin/env python3\nimport os,json,sys\n"
        "print(json.dumps({k:os.environ.get(k) for k in "
        "('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE')}))\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    interpreter.chmod(0o700)
    env = dict(
        os.environ, GPUS="0", RECIPE="recipe.json", RUN_DIR=str(tmp_path / "run"),
        SCENE_AUDIT="audit.json", PYTHON=str(interpreter),
        HF_HOME="/unrelated/cache", HF_HUB_CACHE="/unrelated/hub",
        HUGGINGFACE_HUB_CACHE="/unrelated/legacy", NOFILE_LIMIT="8192",
        EPISODE_CACHE_SIZE="384", HF_TOKEN="test-token", WANDB_API_KEY="test-key",
    )
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True,
                            text=True, check=True)
    actual, command = map(json.loads, result.stdout.splitlines())
    cache = tmp_path / "outputs/cache/huggingface"
    assert actual == dict(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"),
                          HUGGINGFACE_HUB_CACHE=str(cache / "hub"))
    assert command[command.index("--episode-cache-size") + 1] == "384"


def test_launcher_rejects_missing_credentials_before_creating_run(tmp_path):
    (tmp_path / 'scripts').mkdir()
    launcher = tmp_path / 'scripts/launch_optional_fqi.sh'
    shutil.copy2('scripts/launch_optional_fqi.sh', launcher)
    env = dict(os.environ, GPUS='0', RECIPE='recipe.json', RUN_DIR=str(tmp_path / 'run'))
    for key in ('HF_TOKEN', 'WANDB_API_KEY'):
        env.pop(key, None)
    result = subprocess.run(['bash', str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'HF_TOKEN' in result.stderr
    assert not (tmp_path / 'run').exists()
