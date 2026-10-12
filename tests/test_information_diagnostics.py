import numpy as np
from test_action_conditioned_jepa_rollout import _normalization
from test_information_perception import Decoder, Encoder, Predictor
from test_rtc_forecast_provider import _context, _obs

from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
from latency_meta_mdp.meta.information.diagnostics import SnapshotProbe
from latency_meta_mdp.meta.information.perception import PerceptionAccess
from latency_meta_mdp.meta.information.records import StageRecorder


def test_probe_compares_identical_real_history_and_runs_once(tmp_path):
    tmp_path = tmp_path / "probe"
    engine = FrozenForecastEngine(Predictor(), Decoder(), device="cpu")
    encoder = Encoder()
    access = PerceptionAccess(engine, encoder, _normalization(), StageRecorder("main"))
    probe = SnapshotProbe(
        access,
        engine=engine,
        encoder=encoder,
        normalization=_normalization(),
        output_dir=tmp_path,
        infer=lambda c, f: np.full((50, 7), c.estimated_delay_ticks),
    )
    for tick in range(11):
        probe.ingest(_obs(tick), None if tick == 0 else np.zeros(7))
    probe.observe(_context(10), prepare_history=True)
    output = probe.predict(_context(10))
    assert output.available
    assert [r["q"] for r in probe.report["comparisons"]] == [0, 9, 20]
    for row in probe.report["comparisons"]:
        for comparison in row["paths"].values():
            assert comparison["forecast_max_abs"] == 0
            assert comparison["actions_max_abs"] == 0
    assert (tmp_path / "source.npz").is_file()
    calls = len(encoder.batches)
    probe.predict(_context(10))
    assert len(encoder.batches) == calls
    with np.load(tmp_path / "source.npz", allow_pickle=False) as data:
        assert data["images"].shape[:2] == (9, 2)
        assert data["controls"].shape == (8, 7)
