from mimir.skills.extractor import extract_skeleton
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
