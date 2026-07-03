from pathlib import Path

from mimir.skills.extractor import extract_skeleton
from mimir.skills.injector import InjectorConfig, SkillInjector
from mimir.skills.store import Skill, SkillStore
from mimir.skills.tracker import CommandEvent, SkillTracker, SkillTrackerConfig


class TestExtractSkeleton:
    def test_single_command_returns_unchanged(self) -> None:
        skeleton = extract_skeleton(["adb -s device1 shell reboot"])
        assert skeleton.template == "adb -s device1 shell reboot"
        assert skeleton.fixed_ratio == 1.0
        assert skeleton.variable_count == 0

    def test_extracts_device_variable(self) -> None:
        commands = [
            "adb -s ABC123 shell reboot bootloader",
            "adb -s DEF456 shell reboot bootloader",
            "adb -s GHI789 shell reboot bootloader",
        ]
        skeleton = extract_skeleton(commands)
        assert skeleton.template == "adb -s {var} shell reboot bootloader"
        assert skeleton.variable_count == 1
        assert skeleton.fixed_ratio > 0.6

    def test_pipeline_command_as_whole(self) -> None:
        commands = [
            "cat /tmp/a.log | grep error | sort | uniq -c",
            "cat /tmp/b.log | grep error | sort | uniq -c",
            "cat /tmp/c.log | grep error | sort | uniq -c",
        ]
        skeleton = extract_skeleton(commands)
        assert "cat" in skeleton.template
        assert "grep error" in skeleton.template
        assert "sort | uniq -c" in skeleton.template
        assert "{var}" in skeleton.template

    def test_short_commands_ignored(self) -> None:
        commands = ["cd /a", "cd /b", "cd /c"]
        skeleton = extract_skeleton(commands)
        # cd has a fixed part but very short; still produces template
        assert "cd" in skeleton.template
        assert "{var}" in skeleton.template


class TestSkillTracker:
    def test_empty_tracker_has_no_ready_clusters(self) -> None:
        tracker = SkillTracker()
        assert tracker.ready_clusters() == []

    def test_repetition_builds_frustration(self) -> None:
        config = SkillTrackerConfig(
            window_size=20,
            min_repetitions=5,
            frustration_threshold=50.0,
            min_fixed_ratio=0.6,
        )
        tracker = SkillTracker(config)
        for i in range(10):
            tracker.observe(CommandEvent("Shell", f"adb -s DEV{i} shell reboot bootloader"))

        ready = tracker.ready_clusters()
        assert len(ready) == 1
        assert ready[0].compute_frustration(config.min_repetitions) > 0

    def test_short_commands_stay_below_threshold(self) -> None:
        config = SkillTrackerConfig(
            window_size=20,
            min_repetitions=5,
            frustration_threshold=50.0,
            min_fixed_ratio=0.6,
        )
        tracker = SkillTracker(config)
        for i in range(10):
            tracker.observe(CommandEvent("Shell", f"cd /path/{i}"))
        assert tracker.ready_clusters() == []

    def test_reset_clears_cluster(self) -> None:
        config = SkillTrackerConfig(
            window_size=20,
            min_repetitions=5,
            frustration_threshold=50.0,
            min_fixed_ratio=0.6,
        )
        tracker = SkillTracker(config)
        for i in range(6):
            tracker.observe(CommandEvent("Shell", f"adb -s DEV{i} shell reboot bootloader"))
        tracker.reset("Shell:adb")
        assert tracker.ready_clusters() == []

    def test_state_roundtrip(self) -> None:
        tracker = SkillTracker(SkillTrackerConfig(window_size=5))
        for i in range(3):
            tracker.observe(CommandEvent("Shell", f"adb -s DEV{i} shell reboot"))

        state = tracker.state()
        restored = SkillTracker()
        restored.restore(state)
        assert restored.ready_clusters() == tracker.ready_clusters()
        assert len(restored._buffer) == len(tracker._buffer)

    def test_window_prunes_old_events(self) -> None:
        config = SkillTrackerConfig(window_size=5)
        tracker = SkillTracker(config)
        for i in range(10):
            tracker.observe(CommandEvent("Shell", f"adb -s DEV{i} shell reboot"))
        # Only the last 5 events should remain in the cluster.
        cluster = tracker._clusters.get("Shell:adb")
        assert cluster is not None
        assert cluster.repeat_count == 5

    def test_different_subcommands_are_separated(self) -> None:
        tracker = SkillTracker()
        for i in range(3):
            tracker.observe(CommandEvent("Shell", f"adb shell DEV{i}"))
            tracker.observe(CommandEvent("Shell", f"adb logcat DEV{i}"))
        # Different subcommands should not share the same cluster.
        assert "Shell:adb shell" in tracker._clusters
        assert "Shell:adb logcat" in tracker._clusters


class TestSkillStore:
    def test_add_and_load(self, tmp_path: Path) -> None:
        store = SkillStore(tmp_path / "skills.jsonl")
        skill = Skill(id="s1", type="alias", name="gs", trigger_pattern="gs", expansion="git status -sb")
        store.add(skill)
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].id == "s1"

    def test_dedup_by_id(self, tmp_path: Path) -> None:
        store = SkillStore(tmp_path / "skills.jsonl")
        skill1 = Skill(id="s1", type="alias", name="gs", trigger_pattern="gs", expansion="git status")
        skill2 = Skill(id="s1", type="alias", name="gs", trigger_pattern="gs", expansion="git status -sb")
        store.add(skill1)
        store.add(skill2)
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].expansion == "git status -sb"

    def test_get_by_id(self, tmp_path: Path) -> None:
        store = SkillStore(tmp_path / "skills.jsonl")
        skill = Skill(id="s1", type="alias", name="gs", trigger_pattern="gs", expansion="git status -sb")
        store.add(skill)
        found = store.get_by_id("s1")
        assert found is not None
        assert found.name == "gs"
        assert store.get_by_id("missing") is None


class TestSkillInjector:
    def test_select_sorts_by_confidence_and_usage(self) -> None:
        skills = [
            Skill(id="a", type="alias", name="a", trigger_pattern="a", confidence=0.9, usage_count=1),
            Skill(id="b", type="alias", name="b", trigger_pattern="b", confidence=0.95, usage_count=0),
        ]
        injector = SkillInjector(InjectorConfig(max_active=10, min_confidence=0.85))
        selected = injector.select(skills)
        assert len(selected) == 2
        assert selected[0].id == "a"  # higher confidence * usage

    def test_format_escapes_backticks(self) -> None:
        skill = Skill(id="x", type="alias", name="`bad`", trigger_pattern="x", expansion="`cmd`")
        injector = SkillInjector()
        formatted = injector.format([skill])
        assert "`bad`" not in formatted
        assert "'bad'" in formatted
        assert "'cmd'" in formatted

    def test_low_confidence_skills_filtered(self) -> None:
        skills = [
            Skill(id="a", type="alias", name="a", trigger_pattern="a", confidence=0.5),
        ]
        injector = SkillInjector(InjectorConfig(max_active=10, min_confidence=0.85))
        assert injector.select(skills) == []
