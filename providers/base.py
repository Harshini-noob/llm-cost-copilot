from abc import ABC, abstractmethod


class ModelProvider(ABC):
    """
    Common interface every LLM provider must implement. The router only
    ever talks to this interface — it never knows or cares whether a
    model call is actually going to Groq, Gemini, OpenAI, etc.
    """

    @abstractmethod
    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float = None) -> dict:
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
        """
        raise NotImplementedError