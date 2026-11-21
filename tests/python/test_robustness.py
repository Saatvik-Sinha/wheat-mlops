from __future__ import annotations

import pandas as pd

from wheat_mlops.robustness import select_search_ids, spsa_optimize


def test_spsa_finds_bounded_maximum_reproducibly():
    bounds = {"x": [-2.0, 2.0]}

    def objective(parameters):
        return -((parameters["x"] - 1.0) ** 2)

    result_a, history_a = spsa_optimize(
        objective,
        bounds,
        iterations=30,
        seed=7,
        a=0.2,
        c=0.1,
        alpha=0.602,
        gamma=0.101,
        stability_constant=4,
    )
    result_b, history_b = spsa_optimize(
        objective,
        bounds,
        iterations=30,
        seed=7,
        a=0.2,
        c=0.1,
        alpha=0.602,
        gamma=0.101,
        stability_constant=4,
    )
    assert result_a == result_b
    assert history_a == history_b
    assert -2.0 <= result_a["x"] <= 2.0
    assert abs(result_a["x"] - 1.0) < 0.5


def test_search_subset_is_seeded_disjoint_subset(tmp_path):
    split = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {"image_id": f"img{i}", "split": "val", "source": f"s{i % 2}", "n_boxes": i % 5 * 15}
            for i in range(12)
        ]
        + [{"image_id": "train", "split": "train", "source": "s0", "n_boxes": 3}]
    ).to_csv(split, index=False)
    first = select_search_ids(split, 5, 19)
    second = select_search_ids(split, 5, 19)
    assert first == second
    assert len(first) == 5
    assert "train" not in first
