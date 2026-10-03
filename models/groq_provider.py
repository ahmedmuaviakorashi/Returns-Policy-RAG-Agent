import json
from typing import Any


class LLMProviderError(RuntimeError):
    pass


class GroqProvider:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b") -> None:
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def extract_refund_parameters(self, query: str) -> dict[str, Any]:
        prompt = (
            "Extract only values explicitly stated or directly implied in the customer message. "
            "Use null for unknown values. Allowed categories are electronics, apparel, home, "
            "books, and refurbished. Return JSON with purchase_price, days_since_delivery, "
            "opened, and category. opened must be true, false, or null.\n\n"
            f"Customer message: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=180,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMProviderError("The model returned an empty response.")
            parsed = json.loads(content)
            return {key: value for key, value in parsed.items() if value is not None}
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError("Groq parameter extraction failed.") from exc
