from models.policy_retriever import PolicyRetriever
from utils.core import load_json_file


def make_retriever() -> PolicyRetriever:
    return PolicyRetriever(load_json_file("data/policies.json"))


def test_shipping_question_ranks_shipping_policy_first() -> None:
    matches = make_retriever().search("Is the shipping fee refundable?")
    assert matches[0]["policy"]["id"] == "shipping_refundability"


def test_home_category_ranks_home_policy_first() -> None:
    matches = make_retriever().search(
        "sealed blender refund estimate", category="home"
    )
    assert matches[0]["policy"]["id"] == "restocking_home_opened"


def test_category_free_restocking_question_uses_general_policy() -> None:
    matches = make_retriever().search("Do opened items have a restocking fee?")
    assert matches[0]["policy"]["id"] == "restocking_general"


def test_category_free_return_question_uses_general_policy() -> None:
    matches = make_retriever().search("I am past 35 days. Can I still return it?")
    assert matches[0]["policy"]["id"] == "returns_general"
