"""Fraud Defence Workbench engine.

Eight switchable stages (capture → provenance → reuse → pixel forensics → AI
detection → documents → reasoning → human review), each returning plain-language
findings linked to the check that produced them. Shared by the batch job and the
live Databricks App so offline metrics predict online behaviour.
"""
from .catalog import PRESETS, STAGES, default_config, preset_config
from .engine import ClaimRun, Finding, StageResult, run_claim

__all__ = ["STAGES", "PRESETS", "default_config", "preset_config",
           "run_claim", "ClaimRun", "StageResult", "Finding"]
