from router import classify, classify_llm
from test_prompts import TEST_PROMPTS

rule_correct = 0
llm_correct = 0
total = len(TEST_PROMPTS)

print(f"{'PROMPT':<55} {'EXPECTED':<10} {'RULE':<10} {'LLM'}")
print("-" * 95)

for item in TEST_PROMPTS:
    prompt = item["prompt"]
    expected = item["expected_tier"]

    rule_tier = classify(prompt)
    llm_tier = classify_llm(prompt)

    if rule_tier == expected:
        rule_correct += 1
    if llm_tier == expected:
        llm_correct += 1

    print(f"{prompt[:52]:<55} {expected:<10} {rule_tier:<10} {llm_tier}")

print("-" * 95)
print(f"Rule-based accuracy: {rule_correct}/{total} ({rule_correct/total*100:.1f}%)")
print(f"LLM-based accuracy:  {llm_correct}/{total} ({llm_correct/total*100:.1f}%)")