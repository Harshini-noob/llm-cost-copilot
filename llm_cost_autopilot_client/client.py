import requests
from .models import RoutingResult, CandidateModel


class LLMCostAutopilotClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def ask(self, prompt: str, routing_mode: str = "balanced",
            max_cost_usd: float = None, min_quality: float = None) -> RoutingResult:
        params = {"prompt": prompt, "routing_mode": routing_mode}
        if max_cost_usd is not None:
            params["max_cost_usd"] = max_cost_usd
        if min_quality is not None:
            params["min_quality"] = min_quality

        response = requests.post(
            f"{self.base_url}/query",
            params=params,
            headers={"X-API-Key": self.api_key},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        candidates = [
            CandidateModel(model=c["model"], estimated_cost=c.get("estimated_cost"),
                            expected_quality=c.get("expected_quality"),
                            avg_latency_sec=c.get("avg_latency_sec"))
            for c in data.get("candidates_considered", [])
        ]
        rejected = [
            CandidateModel(model=r["model"], reason=r.get("reason"))
            for r in data.get("rejected_candidates", [])
        ]

        return RoutingResult(
            answer=data.get("answer", ""),
            model=data.get("model", ""),
            routing_mode=data.get("routing_mode", routing_mode),
            routing_reason=data.get("routing_reason", ""),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            quality_score=data.get("quality_score"),
            escalated=data.get("escalated", False),
            escalation_reason=data.get("escalation_reason"),
            request_id=data.get("request_id", ""),
            candidates_considered=candidates,
            rejected_candidates=rejected,
            raw=data,
        )

    def compare_models(self, prompt: str, routing_mode: str = "balanced") -> list:
        result = self.ask(prompt, routing_mode=routing_mode)
        return result.candidates_considered