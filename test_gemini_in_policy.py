# test_gemini_in_policy.py
from routing_policy import select_model

decision = select_model("What is 2+2?", routing_mode="economy")
print("Economy mode picked:", decision["model"])
print("All candidates:")
for c in decision["candidates_considered"]:
    print(" ", c)