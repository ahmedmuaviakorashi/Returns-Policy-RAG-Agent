import math
import re
from collections import Counter
from typing import Any


class PolicyRetriever:
    def __init__(self, policies: list[dict[str, str]]) -> None:
        self.policies = policies
        self.documents = [self._tokens(f"{item['title']} {item['content']}") for item in policies]
        self.document_frequency = Counter(
            token for document in self.documents for token in set(document)
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        aliases = {
            "refundability": "refund",
            "refundable": "refund",
            "returns": "return",
            "returned": "return",
            "fees": "fee",
            "damaged": "damage",
            "defective": "defect",
        }
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [aliases.get(token, token.rstrip("s")) for token in tokens if len(token) > 1]

    def search(
        self, query: str, top_k: int = 3, category: str | None = None
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return []

        query_vector = self._vector(query_tokens)
        results: list[dict[str, Any]] = []
        lowered = query.lower()

        for policy, tokens in zip(self.policies, self.documents):
            score = self._cosine(query_vector, self._vector(tokens))
            policy_id = policy["id"]
            if any(term in lowered for term in ("shipping", "delivery fee", "postage")):
                score += 1.0 if policy_id == "shipping_refundability" else 0.0
            if any(term in lowered for term in ("restocking", "opened", "sealed", "fee")):
                score += 0.6 if policy_id.startswith("restocking_") else 0.0
            if any(term in lowered for term in ("damaged", "defective", "doa")):
                score += 1.0 if policy_id == "doa_defective" else 0.0
            if category:
                category_text = f"{policy_id} {policy['title']} {policy['content']}".lower()
                score += 0.8 if category in category_text else 0.0
            elif any(term in lowered for term in ("restocking", "opened", "sealed")):
                score += 1.0 if policy_id == "restocking_general" else 0.0
            elif any(term in lowered for term in ("return", "window", "days")):
                score += 0.8 if policy_id == "returns_general" else 0.0
            if score > 0:
                results.append({"policy": policy, "score": round(score, 4)})

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        length = len(tokens) or 1
        document_count = len(self.documents)
        return {
            token: (count / length)
            * (math.log((document_count + 1) / (self.document_frequency[token] + 1)) + 1)
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)
