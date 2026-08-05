from router import classify, classify_llm
from test_prompts import TEST_PROMPTS

print(f"{'PROMPT':<50} {'RULE-BASED':<12} {'LLM-BASED'}")
print("-" * 80)

for item in TEST_PROMPTS:
    prompt = item["prompt"]
    rule_tier = classify(prompt)
    llm_tier = classify_llm(prompt)
    match = "✓" if rule_tier == llm_tier else "✗ DIFFERENT"
    print(f"{prompt[:47]:<50} {rule_tier:<12} {llm_tier:<12} {match}")