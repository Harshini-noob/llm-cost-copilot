from dataclasses import dataclass, field


@dataclass
class CandidateModel:
    """A single model that was considered (or rejected) during routing."""
    model: str
    estimated_cost: float = None
    expected_quality: float = None
    avg_latency_sec: float = None
    reason: str = None  # only set for rejected candidates

    def __repr__(self):
        if self.reason:
            return f"CandidateModel(model={self.model!r}, rejected_reason={self.reason!r})"
        return (f"CandidateModel(model={self.model!r}, cost=${self.estimated_cost:.6f}, "
                f"quality={self.expected_quality}, latency={self.avg_latency_sec}s)")


@dataclass
class RoutingResult:
    """The full result of a routed query — the answer plus the complete
    reasoning behind which model was chosen, at what cost, and what
    the alternatives were."""
    answer: str
    model: str
    routing_mode: str
    routing_reason: str
    total_cost_usd: float
    quality_score: int
    escalated: bool
    escalation_reason: str
    request_id: str
    candidates_considered: list = field(default_factory=list)
    rejected_candidates: list = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)  # full original response, for anything not modeled above

    def __repr__(self):
        return (f"RoutingResult(model={self.model!r}, cost=${self.total_cost_usd:.6f}, "
                f"quality={self.quality_score}/5, escalated={self.escalated})")

    def summary(self) -> str:
        """A human-readable printout of the full routing decision."""
        lines = [
            f"Answer: {self.answer[:150]}{'...' if len(self.answer) > 150 else ''}",
            "",
            f"Model chosen: {self.model}",
            f"Why: {self.routing_reason}",
            f"Routing mode: {self.routing_mode}",
            f"Cost: ${self.total_cost_usd:.6f}",
            f"Quality score: {self.quality_score}/5",
            f"Escalated: {self.escalated}" + (f" ({self.escalation_reason})" if self.escalated else ""),
        ]
        if self.candidates_considered:
            lines.append("\nCandidates considered:")
            for c in self.candidates_considered:
                lines.append(f"  {c}")
        if self.rejected_candidates:
            lines.append("\nRejected:")
            for r in self.rejected_candidates:
                lines.append(f"  {r}")
        return "\n".join(lines)