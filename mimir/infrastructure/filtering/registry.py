"""Provider registry that maps Unicode scripts to filter providers."""

from __future__ import annotations

from pathlib import Path

from .provider import JsonRulePack, RulePackProvider, load_rule_packs
from .script_utils import detect_scripts, script_groups


class ProviderRegistry:
    """Maps Unicode script names to concrete FilterProvider instances.

    The registry groups CJK-related scripts (CJK, Hiragana, Katakana, Hangul)
    into a single provider so that one resource file can cover Chinese,
    Japanese, and Korean small-talk.
    """

    def __init__(self) -> None:
        self._providers: dict[str, RulePackProvider] = {}

    def register(self, provider: RulePackProvider) -> None:
        for script in provider.scripts:
            self._providers[script] = provider

    def providers_for(self, text: str) -> list[RulePackProvider]:
        """Return the providers relevant for *text*, without duplicates."""
        scripts = script_groups(detect_scripts(text))
        seen: set[int] = set()
        providers: list[RulePackProvider] = []
        for script in scripts:
            provider = self._providers.get(script)
            if provider is None:
                continue
            pid = id(provider)
            if pid in seen:
                continue
            seen.add(pid)
            providers.append(provider)
        return providers


def default_registry(
    resource_dir: Path | None = None,
    user_resource_dir: Path | None = None,
) -> ProviderRegistry:
    """Build the default registry from built-in and optional user resource files.

    User resources override built-in resources for the same script.
    """
    if resource_dir is None:
        resource_dir = Path(__file__).with_name("resources").resolve()

    packs = load_rule_packs(resource_dir)
    if user_resource_dir is not None and user_resource_dir.exists():
        user_packs = load_rule_packs(user_resource_dir)
        packs.update(user_packs)

    registry = ProviderRegistry()
    # Group packs by provider to avoid one-script-per-provider fragmentation.
    # CJK is already a single pack in the default resources.
    packs_by_group: dict[frozenset[str], list[JsonRulePack]] = {}
    for pack in packs.values():
        key = pack.scripts
        packs_by_group.setdefault(key, []).append(pack)

    for group_packs in packs_by_group.values():
        registry.register(RulePackProvider(group_packs))

    return registry
