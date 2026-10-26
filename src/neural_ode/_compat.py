"""Small compatibility helpers for plain mappings and OmegaConf configs."""


def to_plain_dict(value):
    if hasattr(value, "items"):
        return {key: to_plain_dict(item) if hasattr(item, "items") else item for key, item in value.items()}
    return value
