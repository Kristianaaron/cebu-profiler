"""Built-in canonical recipes for the Atlas control plane."""

from model_atlas.recipes.builtin import (
    GLM52_HARDWARE,
    GLM52_SOURCE_PATH,
    glm52_no_pruning_recipe,
    tenp_pruning_optin_recipe,
)

__all__ = [
    "GLM52_HARDWARE",
    "GLM52_SOURCE_PATH",
    "glm52_no_pruning_recipe",
    "tenp_pruning_optin_recipe",
]
