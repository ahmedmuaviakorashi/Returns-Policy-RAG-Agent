from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from utils.core import load_json_file


class RefundValidationError(ValueError):
    pass


class RefundCalculator:
    def __init__(self, config_path: str = "data/config.json") -> None:
        self.config = load_json_file(config_path)

    def compute_refund(self, params: dict[str, Any]) -> dict[str, Any]:
        self._validate(params)
        category = params["category"]
        condition = "opened" if params["opened"] else "sealed"
        return_window = self.config["return_window_days_by_category"][category]
        price = Decimal(str(params["purchase_price"]))

        if params["days_since_delivery"] > return_window:
            return {
                "eligible": False,
                "refund_amount": 0.0,
                "restocking_fee": 0.0,
                "restocking_rate": 0.0,
                "return_window_days": return_window,
                "reason": f"The item is outside the {return_window}-day return window.",
            }

        rate = Decimal(str(self.config["restocking_fees"][category][condition]))
        fee = (price * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        refund = (price - fee).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return {
            "eligible": True,
            "refund_amount": float(refund),
            "restocking_fee": float(fee),
            "restocking_rate": float(rate),
            "return_window_days": return_window,
            "reason": f"The {condition} {category} item is within the return window.",
        }

    def _validate(self, params: dict[str, Any]) -> None:
        required = {"purchase_price", "days_since_delivery", "opened", "category"}
        missing = sorted(required.difference(params))
        if missing:
            raise RefundValidationError(f"Missing required values: {', '.join(missing)}")

        try:
            price = Decimal(str(params["purchase_price"]))
        except Exception as exc:
            raise RefundValidationError("Purchase price must be a number.") from exc
        if not price.is_finite() or price <= 0:
            raise RefundValidationError("Purchase price must be greater than zero.")

        days = params["days_since_delivery"]
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise RefundValidationError("Days since delivery must be a non-negative integer.")
        if type(params["opened"]) is not bool:
            raise RefundValidationError("Item condition must be opened or sealed.")

        category = params["category"]
        if category not in self.config["return_window_days_by_category"]:
            raise RefundValidationError(f"Unsupported category: {category}")
