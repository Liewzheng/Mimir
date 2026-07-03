"""Generate revised skill versions when observed patterns drift."""

from __future__ import annotations

from mimir.skills.extractor import Skeleton
from mimir.skills.store import Skill
from mimir.skills.tracker import PatternCluster


class SkillRevisor:
    """Generate a revised skill version from a cluster when the template drifts."""

    def __init__(self, min_repetitions: int = 5) -> None:
        self.min_repetitions = min_repetitions

    def revise(self, skill: Skill, cluster: PatternCluster) -> Skill | None:
        """Return a new skill version if the cluster skeleton differs meaningfully.

        Returns None if the cluster is too small, has no skeleton, or the
        skeleton is unchanged.
        """
        if cluster.repeat_count < self.min_repetitions or cluster.skeleton is None:
            return None

        new_skeleton: Skeleton = cluster.skeleton
        old_template = skill.template or skill.expansion
        if not old_template or new_skeleton.template == old_template:
            return None

        new_id = f"{skill.id}_v{skill.version + 1}"
        return Skill(
            id=new_id,
            type=skill.type,
            name=skill.name,
            trigger_pattern=new_skeleton.template,
            template=new_skeleton.template if skill.type == "workflow" else None,
            expansion=new_skeleton.template if skill.type == "alias" else None,
            required_context=list(skill.required_context),
            confidence=new_skeleton.fixed_ratio,
            usage_count=0,
            failure_count=0,
            version=skill.version + 1,
            deprecated=False,
        )
