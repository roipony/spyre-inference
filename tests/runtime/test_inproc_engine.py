# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SPYRE_INPROC_ENGINE: run the offline engine in-process at world_size 1."""

import os
from types import SimpleNamespace

import pytest
import torch
from vllm.platforms.cpu import CpuPlatform

from spyre_inference import platform, threading_config
from spyre_inference.models import gemma4
from spyre_inference.platform import (
    TorchSpyrePlatform,
    _maybe_inproc_engine,
    _set_inproc_inductor_config,
    _set_inproc_torch_threads,
    maybe_set_inproc_threads,
)

MP_ENV = "VLLM_ENABLE_V1_MULTIPROCESSING"


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # setenv first so monkeypatch restores the original state of MP_ENV, including
    # removing a value the helper set itself.
    monkeypatch.setenv(MP_ENV, "x")
    monkeypatch.delenv(MP_ENV)
    monkeypatch.setattr(platform, "_inproc_env_pid", None)
    # So maybe_set_inproc_threads() leaves the global Inductor config alone.
    monkeypatch.delenv("TORCHINDUCTOR_CPP_DYNAMIC_THREADS", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)


@pytest.fixture
def thread_calls(monkeypatch):
    """Record torch.set_num_threads calls; OMP_NUM_THREADS=8 and 64 threads now."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    calls = []
    monkeypatch.setattr(platform.torch, "get_num_threads", lambda: 64)
    monkeypatch.setattr(platform.torch, "set_num_threads", calls.append)
    return calls


def _parallel_config(world_size: int, backend):
    # The backend as ParallelConfig left it, before CpuPlatform ran.
    return SimpleNamespace(world_size=world_size, distributed_executor_backend=backend)


@pytest.mark.parametrize("backend", [None, "uni"])
def test_switch_on_disables_multiprocessing_at_world_size_1(monkeypatch, backend):
    monkeypatch.setenv("SPYRE_INPROC_ENGINE", "1")
    assert _maybe_inproc_engine(_parallel_config(1, backend)) is True
    assert os.environ[MP_ENV] == "0"
    # Idempotent for a second eligible engine in the same process.
    assert _maybe_inproc_engine(_parallel_config(1, backend)) is True
    assert os.environ[MP_ENV] == "0"


@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_switch_off_leaves_env_alone(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SPYRE_INPROC_ENGINE", raising=False)
    else:
        monkeypatch.setenv("SPYRE_INPROC_ENGINE", value)
    assert _maybe_inproc_engine(_parallel_config(1, "uni")) is False
    assert MP_ENV not in os.environ


@pytest.mark.parametrize("user_value", ["1", "0"])
def test_user_setting_is_respected(monkeypatch, user_value):
    monkeypatch.setenv("SPYRE_INPROC_ENGINE", "1")
    monkeypatch.setenv(MP_ENV, user_value)
    assert _maybe_inproc_engine(_parallel_config(1, "uni")) is False
    assert os.environ[MP_ENV] == user_value


@pytest.mark.parametrize(
    "world_size, backend", [(4, "mp"), (2, None), (1, "mp"), (1, "external_launcher")]
)
def test_not_eligible_keeps_multiprocessing(monkeypatch, world_size, backend):
    monkeypatch.setenv("SPYRE_INPROC_ENGINE", "1")
    assert _maybe_inproc_engine(_parallel_config(world_size, backend)) is False
    assert MP_ENV not in os.environ


@pytest.mark.parametrize("world_size, backend", [(4, "mp"), (1, "mp")])
def test_later_ineligible_engine_raises(monkeypatch, world_size, backend):
    # vLLM's env cache and the worker's RANK/WORLD_SIZE cannot be undone in-process.
    monkeypatch.setenv("SPYRE_INPROC_ENGINE", "1")
    assert _maybe_inproc_engine(_parallel_config(1, "uni")) is True
    with pytest.raises(ValueError, match="Build it in a new process"):
        _maybe_inproc_engine(_parallel_config(world_size, backend))
    assert os.environ[MP_ENV] == "0"


def test_value_inherited_from_parent_process_is_not_ours(monkeypatch, thread_calls):
    # e.g. an EngineCore forked from the process that set it: a different pid.
    monkeypatch.setenv("SPYRE_INPROC_ENGINE", "1")
    monkeypatch.setenv(MP_ENV, "0")
    monkeypatch.setattr(platform, "_inproc_env_pid", os.getpid() + 1)
    assert _maybe_inproc_engine(_parallel_config(4, "mp")) is False
    maybe_set_inproc_threads()
    assert thread_calls == []


@pytest.mark.parametrize(
    "omp, current, expected",
    [("8", 64, [8]), ("8", 8, []), (None, 64, []), ("abc", 64, []), ("0", 64, [])],
)
def test_set_inproc_torch_threads(monkeypatch, omp, current, expected):
    if omp is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", omp)
    calls = []
    monkeypatch.setattr(platform.torch, "get_num_threads", lambda: current)
    monkeypatch.setattr(platform.torch, "set_num_threads", calls.append)
    _set_inproc_torch_threads()
    assert calls == expected


@pytest.mark.parametrize(
    "dynamic, compile_threads, expected",
    [
        ("1", "1", (True, 1)),
        (None, None, (False, 32)),
        ("0", "abc", (False, 32)),
        (None, "0", (False, 32)),
    ],
)
def test_set_inproc_inductor_config(monkeypatch, dynamic, compile_threads, expected):
    import torch._inductor.config as inductor_config

    for name, value in (
        ("TORCHINDUCTOR_CPP_DYNAMIC_THREADS", dynamic),
        ("TORCHINDUCTOR_COMPILE_THREADS", compile_threads),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with inductor_config.patch({"cpp.dynamic_threads": False, "compile_threads": 32}):
        _set_inproc_inductor_config()
        got = (inductor_config.cpp.dynamic_threads, inductor_config.compile_threads)
    assert got == expected


class _StopAfterSuper(Exception):
    pass


@pytest.mark.parametrize(
    "switch, world_size, has_model, in_process",
    [
        ("1", 1, True, True),
        ("0", 1, True, False),
        ("1", 2, True, False),
        # A bare VllmConfig() (no model) builds no engine.
        ("1", 1, False, False),
    ],
)
def test_check_and_update_config(
    monkeypatch, thread_calls, switch, world_size, has_model, in_process
):
    monkeypatch.setenv("SPYRE_INPROC_ENGINE", switch)
    # Stub everything around the switch; the rest of the hook has its own tests.
    for name in (
        "log_server_boot",
        "_warn_if_not_in_registry",
        "_maybe_pad_head_dim",
        "_maybe_pad_intermediate_size",
    ):
        monkeypatch.setattr(TorchSpyrePlatform, name, lambda *a: None)
    monkeypatch.setattr(gemma4, "repair_head_dim_access", lambda *a: None)
    monkeypatch.setattr(threading_config, "configure_threading", lambda *a: None)

    seen_by_super = []

    def fake_super(cls, vllm_config):
        seen_by_super.append(os.environ.get(MP_ENV))
        raise _StopAfterSuper

    monkeypatch.setattr(CpuPlatform, "check_and_update_config", classmethod(fake_super))

    model_config = (
        SimpleNamespace(dtype=torch.float16, quantization=None, hf_config=None)
        if has_model
        else None
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            tensor_parallel_size=world_size,
            world_size=world_size,
            distributed_executor_backend="uni" if world_size == 1 else "mp",
            worker_cls="auto",
        ),
        scheduler_config=SimpleNamespace(scheduler_cls=None, disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1),
        attention_config=SimpleNamespace(),
        use_v2_model_runner=False,
    )

    with pytest.raises(_StopAfterSuper):
        TorchSpyrePlatform.check_and_update_config(vllm_config)

    # The variable is in place before super() (CpuPlatform) reads it.
    assert seen_by_super == (["0"] if in_process else [None])
    # The hook never sets the thread count; the worker does, only when in-process.
    assert thread_calls == []
    maybe_set_inproc_threads()
    assert thread_calls == ([8] if in_process else [])
