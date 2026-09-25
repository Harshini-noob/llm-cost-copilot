from abc import ABC, abstractmethod


class ModelProvider(ABC):
    """
    Common interface every LLM provider must implement. The router only
    ever talks to this interface — it never knows or cares whether a
    model call is actually going to Groq, Gemini, OpenAI, etc.
    """

    @abstractmethod
    def generate(self, prompt: str, model: str, max_tokens: int,
                 temperature: float = None, reasoning_effort: str = None) -> dict:
        """
        Must return a dict shaped like:
        {
            "answer": str,
            "model": str,
            "input_tokens": int,
            "output_tokens": int,
            "latency_sec": float,
        }
        Cost is calculated separately by the router using registry pricing,
        so providers only need to report raw token usage.

        temperature is optional — pass 0 for deterministic output
        (e.g. classification, confidence-scoring); leave as None to use
        the provider's own default for normal generation calls.

        reasoning_effort is optional and only meaningful for reasoning-style
        models (e.g. Groq's gpt-oss family) — pass "low" for fast, direct
        answers on classification/scoring tasks where internal reasoning
        would otherwise consume the entire token budget without producing
        a visible answer.
        """
        raise NotImplementedError


from abc import ABC, abstractmethod


class ModelProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, model: str, max_tokens: int,
                        temperature: float = None, reasoning_effort: str = None) -> dict:
        """
        Same contract as before, but now a coroutine — callers must
        `await` this instead of calling it directly.
        """
        raise NotImplementedError