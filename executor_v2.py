import time
from model_registry_v2 import get_model_by_id


async def execute_model(
    model_id: str,
    prompt: str,
    adapters: dict | None = None,
) -> dict:

    model_config = get_model_by_id(model_id)
    provider = model_config["provider"]

    start_time = time.perf_counter()

    try:
        if adapters is None:
            adapters = {}

        adapter = adapters.get(provider)

        if adapter is None:
            raise NotImplementedError(
                f"No adapter configured for provider: {provider}"
            )

        response = await adapter(
            model_config=model_config,
            prompt=prompt,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        input_tokens = response["input_tokens"]
        output_tokens = response["output_tokens"]

        input_cost = (
            input_tokens
            * model_config["input_price_per_1m"]
            / 1_000_000
        )

        output_cost = (
            output_tokens
            * model_config["output_price_per_1m"]
            / 1_000_000
        )

        total_cost = input_cost + output_cost

        return {
            "output": response["output"],
            "model_id": model_config["id"],
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": round(total_cost, 10),
            "latency_ms": round(latency_ms, 2),
            "success": True,
            "error": None,
        }

    except Exception as e:

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "output": None,
            "model_id": model_config["id"],
            "provider": provider,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "latency_ms": round(latency_ms, 2),
            "success": False,
            "error": str(e),
        }