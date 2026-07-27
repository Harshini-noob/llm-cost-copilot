# Each prompt has a human-judged "expected_tier" — this is YOUR judgment call,
# used as ground truth to measure how often the LLM classifier agrees with a human.

TEST_PROMPTS = [
    # --- SIMPLE: factual lookups, definitions, short direct answers ---
    {"prompt": "What is 2+2?", "expected_tier": "simple"},
    {"prompt": "What is the capital of Japan?", "expected_tier": "simple"},
    {"prompt": "Define photosynthesis in one line.", "expected_tier": "simple"},
    {"prompt": "What year did India gain independence?", "expected_tier": "simple"},
    {"prompt": "Spell the word 'necessary'.", "expected_tier": "simple"},
    {"prompt": "What is the boiling point of water in Celsius?", "expected_tier": "simple"},
    {"prompt": "Who wrote Romeo and Juliet?", "expected_tier": "simple"},
    {"prompt": "What is the chemical symbol for gold?", "expected_tier": "simple"},
    {"prompt": "How many continents are there?", "expected_tier": "simple"},
    {"prompt": "Translate 'good morning' to Tamil.", "expected_tier": "simple"},

    # --- MEDIUM: explanations, comparisons, moderate code, some reasoning ---
    {"prompt": "Explain what a REST API is.", "expected_tier": "medium"},
    {"prompt": "Compare Python and JavaScript for beginners.", "expected_tier": "medium"},
    {"prompt": "Write code for a binary search tree in Python.", "expected_tier": "medium"},
    {"prompt": "Why do leaves change color in autumn?", "expected_tier": "medium"},
    {"prompt": "How does a car engine convert fuel into motion?", "expected_tier": "medium"},
    {"prompt": "Explain the difference between TCP and UDP.", "expected_tier": "medium"},
    {"prompt": "Write a Python function to check if a string is a palindrome.", "expected_tier": "medium"},
    {"prompt": "Summarize the plot of Romeo and Juliet.", "expected_tier": "medium"},
    {"prompt": "What are the pros and cons of remote work?", "expected_tier": "medium"},
    {"prompt": "Explain how vaccines work in the immune system.", "expected_tier": "medium"},

    # --- COMPLEX: multi-step reasoning, system design, deep analysis ---
    {"prompt": "Design a database schema for an e-commerce platform.", "expected_tier": "complex"},
    {"prompt": "Analyze the causes of the 2008 financial crisis.", "expected_tier": "complex"},
    {"prompt": "Explain how neural networks learn through backpropagation, including the math.", "expected_tier": "complex"},
    {"prompt": "Design a system architecture for a real-time chat application at scale.", "expected_tier": "complex"},
    {"prompt": "Analyze the economic and social impacts of automation on developing countries.", "expected_tier": "complex"},
    {"prompt": "Compare microservices vs monolithic architecture in detail, including tradeoffs at scale.", "expected_tier": "complex"},
    {"prompt": "Design an algorithm to detect fraud in real-time payment transactions.", "expected_tier": "complex"},
    {"prompt": "Explain the full process of how a compiler translates code into machine instructions.", "expected_tier": "complex"},
    {"prompt": "Analyze the tradeoffs between different consensus algorithms in distributed systems.", "expected_tier": "complex"},
    {"prompt": "Design a recommendation system architecture for an e-commerce platform, including data pipeline.", "expected_tier": "complex"},
]