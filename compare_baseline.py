from models import call_model
from router import classify_llm_with_cost, MODEL_MAP, MAX_TOKENS_BY_TIER
from test_prompts import TEST_PROMPTS

BASELINE_MODEL = "llama-3.3-70b-versatile"
BASELINE_MAX_TOKENS = 500  # fixed baseline cap, unrelated to tier logic

router_total_cost = 0       # includes classifier call cost, so it's an honest total
classifier_total_cost = 0   # tracked separately too, so you can see its share
baseline_total_cost = 0

simple_routed, simple_baseline = 0, 0
medium_routed, medium_baseline = 0, 0
complex_routed, complex_baseline = 0, 0

print(f"{'PROMPT':<50} {'TIER':<10} {'ROUTED $':<12} {'BASELINE $'}")
print("-" * 90)

for item in TEST_PROMPTS:
    prompt = item["prompt"]
    tier, classify_cost = classify_llm_with_cost(prompt)
    routed_model = MODEL_MAP[tier]
    max_tokens = MAX_TOKENS_BY_TIER[tier]

    routed_result = call_model(prompt, model=routed_model, max_tokens=max_tokens)
    baseline_result = call_model(prompt, model=BASELINE_MODEL, max_tokens=BASELINE_MAX_TOKENS)

    routed_cost_with_classifier = routed_result["cost_usd"] + classify_cost

    router_total_cost += routed_cost_with_classifier
    classifier_total_cost += classify_cost
    baseline_total_cost += baseline_result["cost_usd"]

    if tier == "simple":
        simple_routed += routed_cost_with_classifier
        simple_baseline += baseline_result["cost_usd"]
    elif tier == "medium":
        medium_routed += routed_cost_with_classifier
        medium_baseline += baseline_result["cost_usd"]
    else:
        complex_routed += routed_cost_with_classifier
        complex_baseline += baseline_result["cost_usd"]

    print(f"{prompt[:47]:<50} {tier:<10} {routed_cost_with_classifier:<12.8f} {baseline_result['cost_usd']:.8f}")
    print(f"   → routed tokens: {routed_result['output_tokens']}, baseline tokens: {baseline_result['output_tokens']}")

print("-" * 90)
print(f"Total router cost:   ${router_total_cost:.6f}  (of which classifier calls: ${classifier_total_cost:.6f})")
print(f"Total baseline cost: ${baseline_total_cost:.6f}")
print(f"Overall savings: {((baseline_total_cost - router_total_cost) / baseline_total_cost) * 100:.1f}%")
print()
print("--- Per-tier breakdown ---")
if simple_baseline > 0:
    print(f"Simple:  {((simple_baseline - simple_routed) / simple_baseline) * 100:.1f}% savings")
if medium_baseline > 0:
    print(f"Medium:  {((medium_baseline - medium_routed) / medium_baseline) * 100:.1f}% savings")
if complex_baseline > 0:
    print(f"Complex: {((complex_baseline - complex_routed) / complex_baseline) * 100:.1f}% savings")