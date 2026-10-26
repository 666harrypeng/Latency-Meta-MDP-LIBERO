import json


def test_metadata_cache_skips_json_parse_and_invalidates_changed_sources(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information.replay_metadata import load_metadata

    source, cache = tmp_path / "records.json", tmp_path / "cache"
    original = {"rows": [{"x": [1.23456789012345, True, None]}], "operations": [1, 2]}
    source.write_text(json.dumps(original))
    assert load_metadata(source, cache) == original
    real_loads = json.loads
    monkeypatch.setattr(
        json, "loads", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("reparsed"))
    )
    assert load_metadata(source, cache) == original
    monkeypatch.setattr(json, "loads", real_loads)
    source.write_text('{"operations": [3]}')
    assert load_metadata(source, cache, fields=("operations",)) == {"operations": [3]}
    assert load_metadata(source, cache) == {"operations": [3]}


def test_selected_metadata_omits_large_unused_fields_and_recovers_partial_cache(tmp_path):
    from latency_meta_mdp.meta.information.replay_metadata import load_metadata

    source, cache = tmp_path / "audit.json", tmp_path / "cache"
    source.write_text(json.dumps(dict(parcel_events=[{"kind": "delivery"}], ignored="x" * 10000)))
    expected = dict(parcel_events=[{"kind": "delivery"}])
    assert load_metadata(source, cache, fields=("parcel_events",)) == expected
    files = list(cache.glob("*.marshal"))
    assert len(files) == 1 and files[0].stat().st_size < 1000
    files[0].write_bytes(b"incomplete")
    assert load_metadata(source, cache, fields=("parcel_events",)) == expected
