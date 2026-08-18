from embedding_classifier import classify_embedding
from router import classify, classify_llm
from test_prompts import TEST_PROMPTS

rule_correct = embed_correct = llm_correct = 0
total = len(TEST_PROMPTS)

print(f"{'PROMPT':<50} {'EXPECTED':<10} {'RULE':<10} {'EMBED':<10} {'LLM'}")
print("-" * 100)

for item in TEST_PROMPTS:
    prompt = item["prompt"]
    expected = item["expected_tier"]

    rule_tier = classify(prompt)
    embed_tier = classify_embedding(prompt)
    llm_tier = classify_llm(prompt)

    if rule_tier == expected: rule_correct += 1
    if embed_tier == expected: embed_correct += 1
    if llm_tier == expected: llm_correct += 1

    print(f"{prompt[:47]:<50} {expected:<10} {rule_tier:<10} {embed_tier:<10} {llm_tier}")

print("-" * 100)
print(f"Rule-based accuracy:      {rule_correct}/{total} ({rule_correct/total*100:.1f}%)")
print(f"Embedding-based accuracy: {embed_correct}/{total} ({embed_correct/total*100:.1f}%)")
print(f"LLM-based accuracy:       {llm_correct}/{total} ({llm_correct/total*100:.1f}%)")