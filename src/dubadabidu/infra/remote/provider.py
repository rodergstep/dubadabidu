# SPDX-License-Identifier: AGPL-3.0-or-later
"""The seam between the provider-neutral lifecycle (orchestrator, state,
diagnostics) and one GPU rental provider. A second provider (Vast.ai,
docs/MIGRATION_VAST.md) is one more class implementing Provider."""
from __future__ import annotations

from typing import Protocol

Pod = dict[str, object]          # the provider's pod object, as its API returns it
Settings = dict[str, object]     # the merged runpod block (typed in a later commit)


class NoDirectSsh(RuntimeError):
    """The host gave the pod no public TCP mapping — a property of the HOST,
    so the remedy is a different pod, not a longer wait."""


class Provider(Protocol):
    name: str                    # "runpod"
    key_env: str                 # read locally, NEVER sent to the instance

    def create(self, rp: Settings) -> Pod: ...
    def get(self, pid: str) -> Pod: ...
    def destroy(self, pid: str) -> None: ...      # raises RuntimeError ("HTTP 404" = gone)
    def list_all(self) -> list[Pod]: ...
    def status(self, pod: Pod) -> str: ...
    def container_started(self, pod: Pod) -> bool: ...
    def ssh_target(self, pod: Pod) -> tuple[str, int] | None: ...
    def price(self, pod: Pod) -> float | None: ...
    def vcpu(self, pod: Pod) -> float | None: ...
    def ram_gb(self, pod: Pod) -> float | None: ...
    def arm_watchdog(self, rp: Settings, host: str, port: int, seconds: float) -> None: ...


def active() -> Provider:
    """The provider remote commands use. RunPod is the only one so far."""
    from dubadabidu.infra.remote.runpod import RunPodProvider
    return RunPodProvider()
