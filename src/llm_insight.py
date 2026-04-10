import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

LLM_PROVIDER: Optional[str] = None
OPENAI_API_KEY: Optional[str] = None
ANTHROPIC_API_KEY: Optional[str] = None
OPENAI_MODEL: str = "gpt-4o-mini"
ANTHROPIC_MODEL: str = "claude-3.0-mini"


@dataclass
class LLMInsightResult:
    parsed: Dict[str, Any]
    raw: str
    prompt: str


def load_env_vars(env_path: str = ".env") -> None:
    global LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_MODEL, ANTHROPIC_MODEL

    load_dotenv(env_path)

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL)

    if LLM_PROVIDER not in {"openai", "anthropic"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'anthropic'.")

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")


def _build_reason_lines(shap_reasons: List[Dict[str, Any]]) -> str:
    lines = []
    for item in shap_reasons:
        feature = item.get("feature")
        value = float(item.get("shap_value", 0.0))
        direction = "increases" if value > 0 else "decreases"
        lines.append(f"- {feature}: {value:+.4f} ({direction} abandonment risk)")
    return "\n".join(lines)


def build_insight_prompt(
    abandonment_score: float,
    shap_reasons: List[Dict[str, Any]],
    cash_flow_delta: float,
    cash_flow_horizon_days: int = 90,
    forecast_trend: Optional[str] = None,
) -> str:
    """Construct the prompt template that injects abandonment score, SHAP reasons, and cash flow delta."""
    reason_lines = _build_reason_lines(shap_reasons)
    trend_text = f"Forecast trend: {forecast_trend}." if forecast_trend else ""

    return (
        "You are DataSentinel, an AI assistant for retail operations. "
        "Provide a business-ready summary and recommendation for an e-commerce merchant.\n\n"
        "Inputs:\n"
        f"- Abandonment probability: {abandonment_score:.2%}\n"
        f"- Cash flow delta over the next {cash_flow_horizon_days} days: {cash_flow_delta:+,.2f}\n"
        f"- {trend_text}\n"
        "- Top SHAP reasons from the abandonment model:\n"
        f"{reason_lines}\n\n"
        "Your answer must be valid JSON only with the following keys:\n"
        "  1) insight: short paragraph describing the customer risk and business impact\n"
        "  2) recommendation: one practical action the merchant should take immediately\n"
        "  3) model_summary: one sentence that explains why the model flagged this session\n"
        "  4) confidence: one of low, medium, or high\n\n"
        "Use clear, direct language suitable for a business owner. "
        "Do not add any extra keys, commentary, or markdown formatting. "
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = re.search(r"(\{.*\})", text, flags=re.S)
        if candidate:
            return json.loads(candidate.group(1))
        raise


def _call_openai(prompt: str) -> str:
    try:
        import openai
    except ImportError as exc:
        raise ImportError("OpenAI SDK is not installed. Install with 'pip install openai'.") from exc

    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a retail intelligence assistant that produces concise JSON output."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return response.choices[0].message["content"]

    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a retail intelligence assistant that produces concise JSON output."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Anthropic SDK is not installed. Install with 'pip install anthropic'.") from exc

    prompt_prefix = (
        "\n\nHuman: You are a retail intelligence assistant that returns JSON only. "
        "Provide a concise summary and recommendation for the merchant.\n\nAssistant:"
    )
    full_prompt = prompt + prompt_prefix

    client = None
    if hasattr(anthropic, "Anthropic"):
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif hasattr(anthropic, "Client"):
        client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
    else:
        raise RuntimeError("Cannot initialize Anthropic client from installed library.")

    if hasattr(client, "completions"):
        response = client.completions.create(
            model=ANTHROPIC_MODEL,
            prompt=full_prompt,
            max_tokens_to_sample=300,
            temperature=0.2,
        )
        return getattr(response, "completion", response.get("completion"))

    response = client.create(
        model=ANTHROPIC_MODEL,
        prompt=full_prompt,
        max_tokens_to_sample=300,
        temperature=0.2,
    )
    return response.get("completion", response.get("completion_text", ""))


def generate_llm_insight(
    abandonment_score: float,
    shap_reasons: List[Dict[str, Any]],
    cash_flow_delta: float,
    cash_flow_horizon_days: int = 90,
    forecast_trend: Optional[str] = None,
) -> LLMInsightResult:
    if LLM_PROVIDER is None:
        raise RuntimeError("Environment variables are not loaded. Call load_env_vars() first.")

    prompt = build_insight_prompt(
        abandonment_score=abandonment_score,
        shap_reasons=shap_reasons,
        cash_flow_delta=cash_flow_delta,
        cash_flow_horizon_days=cash_flow_horizon_days,
        forecast_trend=forecast_trend,
    )

    if LLM_PROVIDER == "openai":
        raw_response = _call_openai(prompt)
    else:
        raw_response = _call_anthropic(prompt)

    parsed = _extract_json(raw_response)
    return LLMInsightResult(parsed=parsed, raw=raw_response, prompt=prompt)
