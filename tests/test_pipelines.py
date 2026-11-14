"""Unit tests for PipelineManager (no MPS, no real models — fake loaders)."""
import threading

from webui.pipelines import PipelineManager


class FakePipe:
    def __init__(self, tag: str):
        self.tag = tag


def make_manager() -> PipelineManager:
    return PipelineManager(device="cpu")


def test_ensure_loads_and_sets_active():
    m = make_manager()
    report = m._ensure("qwen", lambda: FakePipe("a"))
    assert m.active_name == "qwen"
    assert m._pipe is not None
    assert report.name == "qwen"
    assert report.seconds >= 0.0


def test_ensure_same_name_is_noop():
    m = make_manager()
    first = FakePipe("first")
    m._ensure("qwen", lambda: first)
    report = m._ensure("qwen", lambda: FakePipe("second"))
    assert m._pipe is first  # loader NOT called again
    assert report.seconds < 0.1  # "already loaded" fast path


def test_ensure_swap_unloads_previous():
    m = make_manager()
    m._ensure("qwen", lambda: FakePipe("qwen"))
    m._ensure("other", lambda: FakePipe("other"))
    assert m.active_name == "other"
    assert m._pipe.tag == "other"


def test_unload_clears_state():
    m = make_manager()
    m._ensure("qwen", lambda: FakePipe("qwen"))
    m._unload()
    assert m._pipe is None
    assert m.active_name is None
    m._unload()  # idempotent


def test_concurrent_ensure_loads_once():
    m = make_manager()
    calls = []

    def slow_loader():
        calls.append(1)
        return FakePipe("slow")

    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        m._ensure("qwen", slow_loader)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert m._pipe.tag == "slow"
