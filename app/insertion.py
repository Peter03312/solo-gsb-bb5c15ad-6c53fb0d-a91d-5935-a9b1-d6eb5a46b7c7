"""待加试剂插入位置评估（纯预览，不落库）。

在已保存方案的 **首位 / 相邻步骤之间 / 末尾**（共 n+1 个位置）逐一插入同一股
待加试剂，沿用该方案当前生效规则版本（无则全局启用规则图）跑完整推演，
按现有引擎口径（有向 / 无向类别匹配、首次 high 即中止）汇总每个位置的：

- 首次危险（是否由候选试剂引发、冲突的原步骤与命中依据）；
- 被阻断的原步骤；
- 候选试剂加入步自身的命中；
- 警示 / 提示命中数。

推荐口径：
1. 优先选择推演中不出现 high 命中的位置；
2. 若所有位置都有 high，选**被阻断原步骤最少**（即未被阻断的原步骤最多）者；
3. 其余按警示（medium）命中数少、位置靠前决胜。

本模块只做内存演算，不写步骤、不写推演记录，也不检查安全确认门槛。
"""
from sqlalchemy.orm import Session

from . import deduce, engine


def position_label(position: int, total_original: int) -> str:
    """位置的中文说明（原步骤序号为 1 基口语）。"""
    n = total_original
    if n == 0:
        return "首位（空方案，唯一位置）"
    if position == 0:
        return "首位（第 1 步之前）"
    if position == n:
        return f"末尾（第 {n} 步之后）"
    return f"第 {position} 步与第 {position + 1} 步之间"


def _clone_step(step: engine.InputStep, order_index: int) -> engine.InputStep:
    return engine.InputStep(
        order_index=order_index,
        time_offset_min=step.time_offset_min,
        amount_ml=step.amount_ml,
        note=step.note,
        reagent=engine.InputReagent(
            name=step.reagent.name,
            categories=list(step.reagent.categories),
        ),
    )


def _to_original_index(new_index: int, inserted_at: int):
    """新序列序号 → 原方案序号（候选试剂插入步返回 None）。"""
    if new_index == inserted_at:
        return None
    return new_index - 1 if new_index > inserted_at else new_index


def _annotate_position(report: dict, inserted_at: int, total_original: int) -> dict:
    """把引擎报告（新序列口径）标注为某个插入位置的评估结果。"""
    for item in report["timeline"]:
        item["is_inserted"] = item["order_index"] == inserted_at
        item["original_index"] = _to_original_index(item["order_index"], inserted_at)

    first_danger = None
    if report.get("first_danger"):
        first_danger = dict(report["first_danger"])
        first_danger["involves_inserted"] = first_danger["step"] == inserted_at
        first_danger["step_original_index"] = _to_original_index(
            first_danger["step"], inserted_at
        )
        first_danger["conflicts_with_original_index"] = _to_original_index(
            first_danger["conflicts_with_step"], inserted_at
        )

    # 插入步本身永远不会是 blocked（high 时它是 danger 步），其余 blocked 均为原步骤
    blocked_original_steps = sorted(
        {
            _to_original_index(idx, inserted_at)
            for idx in report["blocked_steps"]
            if idx != inserted_at
        }
    )

    inserted_step = next(
        item for item in report["timeline"] if item["order_index"] == inserted_at
    )
    summary = report["summary"]
    n = total_original

    return {
        "position": inserted_at,
        "position_label": position_label(inserted_at, n),
        "status": report["status"],
        "status_label": report["status_label"],
        "halted": report["halted"],
        "has_high": first_danger is not None,
        "first_danger": first_danger,
        "inserted_step": {
            "order_index": inserted_at,
            "state": inserted_step["state"],
            "state_label": inserted_step["state_label"],
            "finding_ids": inserted_step.get("finding_ids", []),
            "interactions": inserted_step.get("interactions", []),
            "present_after": inserted_step.get("present_after", []),
        },
        "blocked_original_steps": blocked_original_steps,
        "blocked_original_count": len(blocked_original_steps),
        "executed_original_count": n - len(blocked_original_steps),
        "warning_steps": [
            _to_original_index(idx, inserted_at)
            for idx in report.get("warning_steps", [])
            if idx != inserted_at
        ],
        "high_finding_count": summary["high"],
        "warning_finding_count": summary["medium"],
        "low_finding_count": summary["low"],
        "summary": summary,
        "timeline": report["timeline"],
        "findings": report["findings"],
    }


def _build_recommendation(positions: list[dict], total_original: int) -> dict:
    """按推荐口径选出位置并生成可展示的理由。"""
    n = total_original

    def sort_key(p: dict):
        # 1) 无 high 优先；2) 被阻断原步骤少；3) 警示命中少；4) 位置靠前
        return (
            1 if p["has_high"] else 0,
            p["blocked_original_count"],
            p["warning_finding_count"],
            p["position"],
        )

    chosen = min(positions, key=sort_key)
    reasons: list[str] = []

    if not chosen["has_high"]:
        safe_pool = [p for p in positions if not p["has_high"]]
        reasons.append(
            f"候选试剂在该位置加入后，推演全程不出现严重（high）命中、不会中止"
            f"（共 {len(positions)} 个位置，其中 {len(safe_pool)} 个无严重命中）。"
        )
        min_warnings = min(p["warning_finding_count"] for p in safe_pool)
        warning_tier = [p for p in safe_pool if p["warning_finding_count"] == min_warnings]
        if min_warnings > 0:
            reasons.append(f"在无严重位置中，该位置警示（medium）命中数最少（{min_warnings} 条）。")
        else:
            reasons.append("该位置无警示（medium）命中。")
        if len(warning_tier) > 1:
            reasons.append("警示命中数并列，按位置靠前决胜，推荐此位置。")
    else:
        reasons.append(f"全部 {len(positions)} 个位置均出现严重（high）命中，不存在无危险位置。")
        min_blocked = min(p["blocked_original_count"] for p in positions)
        blocked_tier = [p for p in positions if p["blocked_original_count"] == min_blocked]
        reasons.append(
            f"该位置被阻断的原步骤最少（{min_blocked}/{n} 步被阻断，"
            f"{n - min_blocked}/{n} 个原步骤仍可执行）。"
        )
        min_warnings = min(p["warning_finding_count"] for p in blocked_tier)
        warning_tier = [p for p in blocked_tier if p["warning_finding_count"] == min_warnings]
        if min_warnings > 0:
            reasons.append(f"阻断步骤数并列时，该位置警示（medium）命中数最少（{min_warnings} 条）。")
        if len(warning_tier) > 1:
            reasons.append("警示命中数仍并列，按位置靠前决胜，推荐此位置。")

    return {
        "position": chosen["position"],
        "position_label": chosen["position_label"],
        "has_high": chosen["has_high"],
        "blocked_original_count": chosen["blocked_original_count"],
        "warning_finding_count": chosen["warning_finding_count"],
        "reasons": reasons,
    }


def evaluate_insertion(
    db: Session,
    plan,
    candidate: dict,
    db_rules,
    revision: str,
    rule_version_id: int | None,
    scope: str,
) -> dict:
    """对方案的全部 n+1 个插入位置逐一预演并给出推荐。纯预览。"""
    original_steps = deduce.plan_to_input(db, plan)
    n = len(original_steps)

    def build_steps(inserted_at: int) -> list[engine.InputStep]:
        steps = [_clone_step(s, i if i < inserted_at else i + 1)
                 for i, s in enumerate(original_steps)]
        candidate_step = engine.InputStep(
            order_index=inserted_at,
            amount_ml=candidate["amount_ml"],
            note=candidate.get("note"),
            reagent=engine.InputReagent(
                name=candidate["name"],
                categories=list(candidate["categories"]),
            ),
        )
        steps.insert(inserted_at, candidate_step)
        return steps

    # 原方案自身推演（基线，便于对照：哪些阻断是方案原有的）
    baseline_full = engine.simulate(
        [_clone_step(s, i) for i, s in enumerate(original_steps)],
        db_rules,
        revision,
    )
    baseline = {
        "status": baseline_full["status"],
        "status_label": baseline_full["status_label"],
        "halted": baseline_full["halted"],
        "first_danger": baseline_full["first_danger"],
        "blocked_steps": baseline_full["blocked_steps"],
        "warning_steps": baseline_full["warning_steps"],
        "summary": baseline_full["summary"],
    }

    positions = []
    for k in range(n + 1):
        report = engine.simulate(build_steps(k), db_rules, revision)
        positions.append(_annotate_position(report, k, n))

    recommendation = _build_recommendation(positions, n)

    return {
        "kind": "labmix-insertion-preview",
        "plan_id": plan.id,
        "plan_name": plan.name,
        "candidate": {
            "source": candidate["source"],  # archive | temporary
            "reagent_id": candidate.get("reagent_id"),
            "name": candidate["name"],
            "categories": candidate["categories"],
            "amount_ml": candidate["amount_ml"],
            "note": candidate.get("note"),
        },
        "rule_scope": scope,
        "rule_revision": revision,
        "rule_version_id": rule_version_id,
        "original_step_count": n,
        "position_count": n + 1,
        "baseline": baseline,
        "recommendation": recommendation,
        "positions": positions,
    }
