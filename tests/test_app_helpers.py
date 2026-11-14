"""Unit tests for webui/app.py pure helpers (no MPS, no model)."""
from webui.app import SIZE_PRESETS, _clamp_size, _seed_from, _timestamped_path


def test_seed_from():
    assert _seed_from(123) == 123
    assert _seed_from(0) == 0
    for sentinel in (-1, None):
        s = _seed_from(sentinel)
        assert s >= 0 and s < 2**63


def test_clamp_size():
    assert _clamp_size(512, 512) == (512, 512)
    assert _clamp_size(250, 250) == (256, 256)  # floor 256
    assert _clamp_size(505, 500) == (496, 496)  # round down to /16


def test_size_presets_shape():
    assert "Custom" in SIZE_PRESETS
    for label, (w, h) in SIZE_PRESETS.items():
        if label == "Custom":
            continue
        assert w % 16 == 0 and h % 16 == 0, label
        assert 256 <= w <= 1536 and 256 <= h <= 1536, label


def test_timestamped_path_unique():
    paths = {_timestamped_path("qwen") for _ in range(50)}
    assert len(paths) == 50
