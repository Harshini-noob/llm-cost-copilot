"""
eval/routers

Intelligent routing models, encoders, and evaluation components for LLM Cost Copilot Phase M3.
"""

from eval.routers.data_prep import (
    DataSufficiencyReport,
    ModelOutcomeRecord,
    PreparedDataset,
    PromptRecord,
    STATUS_INSUFFICIENT,
    STATUS_SUFFICIENT,
    assess_data_sufficiency,
    prepare_routing_dataset,
)
from eval.routers.encoders import (
    BaseEncoder,
    DeterministicTestEncoder,
    ModelNotCachedError,
    SentenceTransformerEncoder,
)
from eval.routers.decision import (
    CostNormalizer,
    LambdaDecisionEngine,
    ModelUtilityBreakdown,
    PROVISIONAL_PRESETS,
    RoutingDecision,
    resolve_preset_lambda,
)
from eval.routers.knn_router import (
    KNNRouter,
    NeighborInfo,
)
from eval.routers.learned_router import (
    LearnedRouter,
    ModelFitStatus,
    PromptFeatureExtractor,
)

__all__ = [
    "BaseEncoder",
    "DeterministicTestEncoder",
    "ModelNotCachedError",
    "SentenceTransformerEncoder",
    "PromptRecord",
    "ModelOutcomeRecord",
    "DataSufficiencyReport",
    "PreparedDataset",
    "STATUS_INSUFFICIENT",
    "STATUS_SUFFICIENT",
    "assess_data_sufficiency",
    "prepare_routing_dataset",
    "KNNRouter",
    "NeighborInfo",
    "LearnedRouter",
    "ModelFitStatus",
    "PromptFeatureExtractor",
    "CostNormalizer",
    "LambdaDecisionEngine",
    "ModelUtilityBreakdown",
    "PROVISIONAL_PRESETS",
    "RoutingDecision",
    "resolve_preset_lambda",
]
