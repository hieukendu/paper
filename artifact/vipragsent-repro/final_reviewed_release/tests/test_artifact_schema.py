from vipragsent.artifacts.result_schema import make_pending_result, validate_result_schema


def test_pending_result_schema_is_valid():
    result = make_pending_result("q1")
    assert validate_result_schema(result) == []
    assert result["status"] == "pending"


def test_result_schema_rejects_unknown_status():
    result = make_pending_result("q1")
    result["status"] = "done"
    assert "status" in " ".join(validate_result_schema(result))
