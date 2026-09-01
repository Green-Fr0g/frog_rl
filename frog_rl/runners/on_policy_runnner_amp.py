"""Backward-compatible AMP runner import."""

from __future__ import annotations

from .on_policy_runner import OnPolicyRunner

AmpOnPolicyRunner = OnPolicyRunner

__all__ = ["AmpOnPolicyRunner", "OnPolicyRunner"]
