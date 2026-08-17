"""Release-contract tests for the six-region trained-model grid."""

import json
from pathlib import Path

from ftw_eval.evaluation import REGIONS, evaluation_key, evaluation_targets

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"


def test_evaluation_target_modes() -> None:
    assert evaluation_targets("india") == ["india"]
    assert evaluation_targets("india", eval_country="kenya") == ["kenya"]
    assert evaluation_targets("india", eval_all_regions=True) == list(REGIONS)
    assert evaluation_key("india", "india") == "india"
    assert evaluation_key("india", "kenya") == "india->kenya"


def test_released_headline_grid_has_six_targets_per_source() -> None:
    files: list[tuple[Path, str]] = []
    for source in REGIONS:
        for seed in range(10):
            files.append((RESULTS / f"ftw_unet_{source}_seed{seed}.json", source))
            for model in ("prithvi", "terramind"):
                for mode in ("frozen_decoder", "full_finetune"):
                    files.append(
                        (
                            RESULTS
                            / f"ftw_finetune_fm_{model}_{source}_{mode}_seed{seed}.json",
                            source,
                        )
                    )

    assert len(files) == 300
    for path, source in files:
        assert path.exists(), path.relative_to(ROOT)
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {evaluation_key(source, target) for target in REGIONS}
        assert set(payload) == expected, path.relative_to(ROOT)
        for target in REGIONS:
            record = payload[evaluation_key(source, target)]
            assert record["train_country"] == source
            assert record["eval_country"] == target
