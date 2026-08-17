"""Shared region and result-key conventions for trained-model evaluation."""

REGIONS = ("india", "cambodia", "vietnam", "kenya", "france", "netherlands")


def evaluation_targets(
    source: str, eval_country: str = "", eval_all_regions: bool = False
) -> list[str]:
    """Return the canonical target regions for one trained source model."""
    if source not in REGIONS:
        raise ValueError(f"unknown source region: {source!r}")
    if eval_country and eval_all_regions:
        raise ValueError("eval_country and eval_all_regions are mutually exclusive")
    if eval_country:
        if eval_country not in REGIONS:
            raise ValueError(f"unknown evaluation region: {eval_country!r}")
        return [eval_country]
    return list(REGIONS) if eval_all_regions else [source]


def evaluation_key(source: str, target: str) -> str:
    """Return the released JSON key for a source-target evaluation."""
    return source if source == target else f"{source}->{target}"
