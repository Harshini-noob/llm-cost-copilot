from cost_estimator import estimate_request_cost

cost = estimate_request_cost("Design a distributed caching system", "llama-3.3-70b-versatile", expected_output_tokens=500)
print(cost)