"""Block-sparse SLA attention for MiniMax-H3.

Adapted from PlagueKind/ComfyUI-PlagueKind-Nodes (MIT), whose Triton kernel and
block routing are in turn based on ModelTC/LightX2V (Apache-2.0). See the
repository's THIRD_PARTY_NOTICES.md and LICENSES directory.
"""

from __future__ import annotations

from .patch import patch_h3_sla

__all__ = ["patch_h3_sla"]
