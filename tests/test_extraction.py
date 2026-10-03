from utils.core import extract_parameters, load_json_file

CONFIG = load_json_file("data/config.json")


def test_extracts_complete_refund_request() -> None:
    params = extract_parameters(
        "Opened headphones cost $200 and arrived 12 days ago.", CONFIG
    )
    assert params == {
        "purchase_price": 200.0,
        "days_since_delivery": 12,
        "opened": True,
        "category": "electronics",
    }


def test_day_count_is_not_treated_as_price() -> None:
    params = extract_parameters("I am past 35 days. Can I still return it?", CONFIG)
    assert params == {"days_since_delivery": 35}


def test_unopened_takes_precedence_over_opened_substring() -> None:
    params = extract_parameters("An unopened phone for $500, delivered yesterday", CONFIG)
    assert params["opened"] is False
    assert params["days_since_delivery"] == 1
