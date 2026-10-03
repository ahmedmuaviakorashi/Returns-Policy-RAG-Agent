from agents.returns_agent import ReturnsPolicyAgent


def test_agent_calculates_refund_without_api_key() -> None:
    result = ReturnsPolicyAgent().run(
        "I paid $300 for a sealed blender delivered 10 days ago. How much is my refund?"
    )
    assert result["tool_result"]["refund_amount"] == 300.0
    assert result["missing_parameters"] == []
    assert "Estimated merchandise refund: $300.00" in result["answer"]
    assert "Source:" in result["answer"]


def test_agent_requests_missing_values() -> None:
    result = ReturnsPolicyAgent().run("How much refund can I get for my opened phone?")
    assert result["tool_result"] is None
    assert set(result["missing_parameters"]) == {
        "purchase_price",
        "days_since_delivery",
    }


def test_policy_question_does_not_invent_a_price() -> None:
    result = ReturnsPolicyAgent().run("I am past 35 days. Can I still return it?")
    assert result["tool_result"] is None
    assert "purchase_price" not in result["parameters"]
