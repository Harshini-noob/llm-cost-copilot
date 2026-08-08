import os
from dotenv import load_dotenv
from llm_cost_autopilot_client import LLMCostAutopilotClient

load_dotenv()

client = LLMCostAutopilotClient(api_key=os.getenv("LLM_AUTOPILOT_API_KEY"))

result = client.ask("Explain how vaccines work", routing_mode="economy")

print("Answer:", result["answer"][:100], "...")
print("Model used:", result["model"])
print("Cost:", result["total_cost_usd"])
print("Quality score:", result["quality_score"])

print("\n--- Comparing all models ---")
comparison = client.compare_models("What is 2+2?", routing_mode="balanced")
for c in comparison:
    print(c["model"], "→ $", c["estimated_cost"], "| quality:", c["expected_quality"])