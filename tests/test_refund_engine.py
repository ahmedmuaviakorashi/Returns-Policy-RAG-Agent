import pytest

from tools.refund_engine import RefundCalculator, RefundValidationError


@pytest.fixture
def calculator() -> RefundCalculator:
    return RefundCalculator()


def test_opened_electronics_fee(calculator: RefundCalculator) -> None:
    result = calculator.compute_refund(
        {
            "purchase_price": 200,
            "days_since_delivery": 12,
            "opened": True,
            "category": "electronics",
        }
    )
    assert result["eligible"] is True
    assert result["restocking_fee"] == 20.0
    assert result["refund_amount"] == 180.0


def test_outside_return_window(calculator: RefundCalculator) -> None:
    result = calculator.compute_refund(
        {
            "purchase_price": 900,
            "days_since_delivery": 31,
            "opened": False,
            "category": "electronics",
        }
    )
    assert result["eligible"] is False
    assert result["refund_amount"] == 0.0


@pytest.mark.parametrize("price", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_price(calculator: RefundCalculator, price: float) -> None:
    with pytest.raises(RefundValidationError):
        calculator.compute_refund(
            {
                "purchase_price": price,
                "days_since_delivery": 1,
                "opened": False,
                "category": "home",
            }
        )


def test_rejects_negative_delivery_age(calculator: RefundCalculator) -> None:
    with pytest.raises(RefundValidationError):
        calculator.compute_refund(
            {
                "purchase_price": 10,
                "days_since_delivery": -1,
                "opened": False,
                "category": "books",
            }
        )
