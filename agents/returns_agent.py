from typing import Any, Protocol

from models.groq_provider import LLMProviderError
from models.policy_retriever import PolicyRetriever
from tools.refund_engine import RefundCalculator, RefundValidationError
from utils.core import extract_parameters, load_json_file, missing_refund_parameters


class ParameterProvider(Protocol):
    def extract_refund_parameters(self, query: str) -> dict[str, Any]: ...


class ReturnsPolicyAgent:
    def __init__(self, llm_provider: ParameterProvider | None = None) -> None:
        self.config = load_json_file("data/config.json")
        self.policies = load_json_file("data/policies.json")
        self.retriever = PolicyRetriever(self.policies)
        self.calculator = RefundCalculator()
        self.llm_provider = llm_provider

    def run(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Enter a question about a return or refund.")

        params = extract_parameters(query, self.config)
        llm_status = "not_configured"
        if self.llm_provider and missing_refund_parameters(params):
            llm_status = "used"
            try:
                extracted = self.llm_provider.extract_refund_parameters(query)
                for key, value in extracted.items():
                    if key not in params:
                        params[key] = value
            except LLMProviderError:
                llm_status = "unavailable"

        wants_estimate = self._wants_refund_estimate(query, params)
        matches = self.retriever.search(query, category=params.get("category"))
        missing = missing_refund_parameters(params) if wants_estimate else []
        tool_result: dict[str, Any] | None = None
        validation_error: str | None = None

        if wants_estimate and not missing:
            try:
                tool_result = self.calculator.compute_refund(params)
            except RefundValidationError as exc:
                validation_error = str(exc)

        answer = self._compose_answer(matches, missing, tool_result, validation_error)
        return {
            "answer": answer,
            "parameters": params,
            "missing_parameters": missing,
            "policy_matches": matches,
            "tool_result": tool_result,
            "llm_status": llm_status,
        }

    @staticmethod
    def _wants_refund_estimate(query: str, params: dict[str, Any]) -> bool:
        lowered = query.lower()
        estimate_terms = ("how much", "estimate", "refund amount", "get back", "my refund")
        return any(term in lowered for term in estimate_terms) or "purchase_price" in params

    @staticmethod
    def _compose_answer(
        matches: list[dict[str, Any]],
        missing: list[str],
        result: dict[str, Any] | None,
        validation_error: str | None,
    ) -> str:
        parts: list[str] = []
        citations: list[str] = []

        if matches:
            top_policy = matches[0]["policy"]
            parts.append(top_policy["content"])
            citations.append(f"{top_policy['title']} [{top_policy['id']}]")

        if validation_error:
            parts.append(validation_error)
        elif missing:
            labels = {
                "purchase_price": "the purchase price",
                "days_since_delivery": "how many days ago it was delivered",
                "opened": "whether the item is opened or sealed",
                "category": "the item category",
            }
            parts.append(
                "To calculate the refund, I still need "
                + ", ".join(labels[item] for item in missing)
                + "."
            )
        elif result:
            if result["eligible"]:
                parts.append(
                    f"Estimated merchandise refund: ${result['refund_amount']:.2f}. "
                    f"Restocking fee: ${result['restocking_fee']:.2f}."
                )
            else:
                parts.append(f"Estimated merchandise refund: $0.00. {result['reason']}")

        if citations:
            parts.append("Source: " + "; ".join(citations))
        if not parts:
            return "Ask about a return policy or provide an item, price, condition, and delivery age for an estimate."
        return "\n\n".join(parts)
