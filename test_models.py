from models import call_model
from logger import log_request

result = call_model("What is 2+2?", model="llama-3.1-8b-instant")
log_request("What is 2+2?", result, tier="manual")
print(result)