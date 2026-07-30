from vipragsent.data.split_stratified import assign_splits, find_split_overlaps


def test_find_split_overlap_detects_duplicate_id_across_splits():
    records = [{"id": "a", "split": "train"}, {"id": "a", "split": "test"}, {"id": "b", "split": "dev"}]
    overlaps = find_split_overlaps(records)
    assert overlaps["test__train"] == ["a"] or overlaps["train__test"] == ["a"]


def test_assign_splits_preserves_existing_split():
    records = [{"id": str(i), "split": "unassigned"} for i in range(10)] + [{"id": "fixed", "split": "test"}]
    assigned = assign_splits(records, seed=1)
    fixed = [record for record in assigned if record["id"] == "fixed"][0]
    assert fixed["split"] == "test"
    assert {record["split"] for record in assigned} <= {"train", "dev", "test"}
