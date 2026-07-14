from __future__ import annotations


def confusion_matrix(labels: list[str], gold: list[str], pred: list[str]) -> list[list[int]]:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have equal lengths")
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, guess in zip(gold, pred):
        matrix[index[true]][index[guess]] += 1
    return matrix


def row_normalise(matrix: list[list[int]]) -> list[list[float]]:
    normalised = []
    for row in matrix:
        total = sum(row)
        normalised.append([0.0 if total == 0 else value / total for value in row])
    return normalised
