"""Atlas profile-to-recommendation engine + API facade + local GUI.

Entry points:
  * `RecommendationService` — typed service/API facade
  * `RecommendationPolicy` — deterministic versioned recommendation policy
  * `render_gui`/`write_gui` — server-less local single-html GUI
"""

from model_atlas.recommend.api import RecommendationService
from model_atlas.recommend.gui import ATLAS_PROFILE_DEFAULT_DIR, render_gui, write_gui
from model_atlas.recommend.policy import (
    RECOMMENDATION_POLICY_VERSION,
    AtlasProfile,
    CompressionIntent,
    MethodFamily,
    MethodRecommendation,
    ProfileExecutionBinding,
    RecBlock,
    RecConfidence,
    Recommendation,
    RecommendationPolicy,
    RecTarget,
    StageEvidence,
)

__all__ = [
    "ATLAS_PROFILE_DEFAULT_DIR",
    "AtlasProfile",
    "CompressionIntent",
    "MethodFamily",
    "MethodRecommendation",
    "ProfileExecutionBinding",
    "RECOMMENDATION_POLICY_VERSION",
    "RecBlock",
    "RecConfidence",
    "RecTarget",
    "Recommendation",
    "RecommendationPolicy",
    "RecommendationService",
    "StageEvidence",
    "render_gui",
    "write_gui",
]
