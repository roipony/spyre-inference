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

"""SPYRE_UNIPROC_EXECUTOR: undo CpuPlatform's "uni" -> "mp" rewrite at world_size 1."""

from types import SimpleNamespace

import pytest

from spyre_inference.platform import _maybe_uniproc


def _parallel_config(world_size: int, backend):
    # The backend as CpuPlatform.check_and_update_config left it.
    return SimpleNamespace(world_size=world_size, distributed_executor_backend=backend)


@pytest.mark.parametrize("requested", [None, "uni"])
def test_switch_on_restores_uni_at_world_size_1(monkeypatch, requested):
    monkeypatch.setenv("SPYRE_UNIPROC_EXECUTOR", "1")
    pc = _parallel_config(1, "mp")
    _maybe_uniproc(pc, requested)
    assert pc.distributed_executor_backend == "uni"


@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_switch_off_keeps_mp(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SPYRE_UNIPROC_EXECUTOR", raising=False)
    else:
        monkeypatch.setenv("SPYRE_UNIPROC_EXECUTOR", value)
    pc = _parallel_config(1, "mp")
    _maybe_uniproc(pc, "uni")
    assert pc.distributed_executor_backend == "mp"


def test_explicit_mp_is_kept(monkeypatch):
    monkeypatch.setenv("SPYRE_UNIPROC_EXECUTOR", "1")
    pc = _parallel_config(1, "mp")
    _maybe_uniproc(pc, "mp")
    assert pc.distributed_executor_backend == "mp"


def test_tensor_parallel_keeps_mp(monkeypatch):
    monkeypatch.setenv("SPYRE_UNIPROC_EXECUTOR", "1")
    pc = _parallel_config(4, "mp")
    _maybe_uniproc(pc, None)
    assert pc.distributed_executor_backend == "mp"


@pytest.mark.parametrize("backend", ["uni", "external_launcher"])
def test_backend_not_rewritten_to_mp_is_left_alone(monkeypatch, backend):
    # e.g. VLLM_ENABLE_V1_MULTIPROCESSING=0, where CpuPlatform keeps "uni".
    monkeypatch.setenv("SPYRE_UNIPROC_EXECUTOR", "1")
    pc = _parallel_config(1, backend)
    _maybe_uniproc(pc, backend)
    assert pc.distributed_executor_backend == backend
