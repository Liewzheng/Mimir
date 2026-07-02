"""Tests for the language-aware memory filtering engine."""

import pytest

from mimir.infrastructure.filtering import FilterConfig, FilterEngine
from mimir.infrastructure.filtering.engine import FilterEngine as FilterEngineType
from mimir.infrastructure.filtering.provider import JsonRulePack, RulePackProvider
from mimir.infrastructure.filtering.script_utils import detect_scripts, script_groups


@pytest.fixture
def engine() -> FilterEngineType:
    return FilterEngine(FilterConfig())


class TestScriptDetection:
    def test_detect_latin(self) -> None:
        assert detect_scripts("hello world") == {"Latin"}

    def test_detect_cjk(self) -> None:
        assert detect_scripts("你好世界") == {"CJK"}

    def test_detect_mixed(self) -> None:
        scripts = detect_scripts("hello 你好")
        assert "Latin" in scripts
        assert "CJK" in scripts

    def test_detect_unknown(self) -> None:
        assert detect_scripts("12345") == {"Unknown"}

    def test_script_groups_collapse_cjk(self) -> None:
        assert script_groups({"CJK", "Hiragana"}) == {"CJK"}


class TestRulePacks:
    def test_latin_small_talk(self) -> None:
        pack = JsonRulePack(
            scripts=frozenset({"Latin"}),
            small_talk_exact=frozenset({"ok", "hello"}),
            small_talk_prefixes=(),
            small_talk_suffixes=(),
            high_signals=frozenset({"decided"}),
            low_signals=frozenset(),
        )
        provider = RulePackProvider([pack])
        assert provider.is_small_talk("ok")
        assert provider.is_small_talk("hello")
        assert not provider.is_small_talk("I decided to use Python")

    def test_cjk_small_talk(self) -> None:
        pack = JsonRulePack(
            scripts=frozenset({"CJK"}),
            small_talk_exact=frozenset({"继续", "好的", "可以"}),
            small_talk_prefixes=(),
            small_talk_suffixes=(),
            high_signals=frozenset({"决定"}),
            low_signals=frozenset(),
        )
        provider = RulePackProvider([pack])
        assert provider.is_small_talk("继续")
        assert not provider.is_small_talk("我决定用 Python")

    def test_importance_scoring(self) -> None:
        pack = JsonRulePack(
            scripts=frozenset({"Latin"}),
            small_talk_exact=frozenset(),
            small_talk_prefixes=(),
            small_talk_suffixes=(),
            high_signals=frozenset({"decided", "prefer"}),
            low_signals=frozenset({"ok"}),
        )
        provider = RulePackProvider([pack])
        assert provider.score_importance("I decided to use Python") > 0.5
        assert provider.score_importance("ok") == pytest.approx(0.4)


class TestFilterEngine:
    def test_empty_rejected(self, engine: FilterEngine) -> None:
        result = engine.should_store("", source="hook")
        assert not result.store
        assert result.reason == "empty"

    def test_short_small_talk_rejected(self, engine: FilterEngine) -> None:
        result = engine.should_store("ok", source="hook")
        assert not result.store
        assert result.reason == "too_short"

    def test_chinese_small_talk_rejected(self, engine: FilterEngine) -> None:
        result = engine.should_store("继续", source="hook")
        assert not result.store
        assert result.reason == "too_short"

    def test_mixed_small_talk_important_content_kept(self, engine: FilterEngine) -> None:
        result = engine.should_store("可以。OK。接下来我们讨论一下架构设计。", source="hook")
        assert result.store
        assert result.reason == "passed"

    def test_pure_small_talk_filtered(self, engine: FilterEngine) -> None:
        result = engine.should_store("可以。好的。谢谢。再见。", source="hook")
        assert not result.store
        assert result.reason == "mostly_small_talk"

    def test_gibberish_rejected(self, engine: FilterEngine) -> None:
        result = engine.should_store("哈哈哈哈哈哈哈哈哈哈", source="hook")
        assert not result.store
        assert result.reason == "low_density"

    def test_high_signal_short_text_stored(self, engine: FilterEngine) -> None:
        result = engine.should_store("我决定用 Python 写后端", source="hook")
        assert result.store
        assert result.reason == "passed"

    def test_mcp_store_keeps_short_text(self, engine: FilterEngine) -> None:
        # Explicit MCP store bypasses small-talk filtering.
        result = engine.should_store("ok", source="mcp")
        assert result.store
        assert result.reason == "passed"

    def test_force_bypasses_filter(self, engine: FilterEngine) -> None:
        result = engine.should_store("ok", source="hook", force=True)
        assert result.store
        assert result.reason == "forced"

    def test_disabled_filter_passes_all(self) -> None:
        engine = FilterEngine(FilterConfig(enabled=False))
        result = engine.should_store("ok", source="hook")
        assert result.store
        assert result.reason == "filtering_disabled"

    def test_clean_small_talk(self, engine: FilterEngine) -> None:
        cleaned = engine.clean_small_talk("可以。OK。接下来我们讨论一下架构设计。")
        assert "可以" not in cleaned
        assert "OK" not in cleaned
        assert "架构设计" in cleaned


class TestDefaultResources:
    def test_default_registry_loads_latin_and_cjk(self) -> None:
        engine = FilterEngine(FilterConfig())
        assert engine.registry.providers_for("hello")
        assert engine.registry.providers_for("你好")

    def test_english_small_talk_filtered_by_default(self) -> None:
        engine = FilterEngine(FilterConfig())
        result = engine.should_store("hello", source="hook")
        assert not result.store

    def test_chinese_small_talk_filtered_by_default(self) -> None:
        engine = FilterEngine(FilterConfig())
        result = engine.should_store("继续", source="hook")
        assert not result.store
