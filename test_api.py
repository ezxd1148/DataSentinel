import requests
import json

# Test the API endpoint
url = "http://127.0.0.1:8000/insight"
headers = {"Content-Type": "application/json"}

data = {
    "abandonment_score": 0.72,
    "shap_reasons": [
        {"feature": "n_addtocart", "shap_value": 0.18},
        {"feature": "session_velocity", "shap_value": 0.12}
    ],
    "cash_flow_delta": -12000.0,
    "cash_flow_horizon_days": 90,
    "forecast_trend": "declining"
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Success! Response keys:", list(result.keys()))
        print("LLM Output:", result.get("llm_output", {}))
    else:
        print("Error:", response.text)
except Exception as e:
    print(f"Request failed: {e}")