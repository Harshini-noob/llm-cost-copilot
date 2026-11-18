import asyncio

from executor_v2 import execute_model
from adapters_v2 import PROVIDER_ADAPTERS


async def main():

    result = await execute_model(
        model_id="gemini_3_8_flash",
        prompt="Explain binary search in 3 simple sentences.",
        adapters=PROVIDER_ADAPTERS,
    )

    print("\n" + "=" * 50)
    print("REAL MODEL EXECUTION RESULT")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())