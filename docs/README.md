# RoadSense-MTL documentation index

## Project status

| Area | Status | Source of truth |
|---|---|---|
| Overall scope, target architecture and roadmap | Planned | `PLAN.md` |
| Implementation status and AI handoff | Active tracker | `AI_PROGRESS.md` |
| Data foundation | Implemented | `milestone_1.md` |
| Independent single-task baselines | Implemented | `milestone_2.md` |
| Shared conflict-aware multi-task model | Not started | Plan §4–9 and tracker M3 |

Current model input is a `640×384` letterbox canvas (`width × height`). In code
and YAML this is written as `[384, 640]` (`height × width`).

## Documents

- [Project plan](PLAN.md): authoritative scope, target architecture,
  optimization, experiment design, roadmap and final deliverables.
- [AI progress tracker](AI_PROGRESS.md): implementation state, evidence,
  decisions, known risks and exact next actions.
- [Kaggle training](KAGGLE.md): automatic discovery, preflight, pilot/full runs,
  resume behavior, artifacts and failure guardrails.
- [Milestone 1](milestone_1.md): implemented data contract, indexing,
  rasterization, auditing, splitting, metrics and visualization.
- [Milestone 2](milestone_2.md): implemented detection/drivable/lane baselines,
  losses, training engine, checkpoints and verification commands.

## Consistency rules

1. `PLAN.md` owns project scope and future design.
2. `AI_PROGRESS.md` owns current status, evidence and handoff state.
3. Milestone documents describe only code already implemented for that milestone.
4. Planned components must be explicitly marked `NOT_STARTED` or not implemented.
5. Input size is always documented with coordinate order.
6. Operational Milestone 2 baselines are not capacity-matched MTL controls.
7. Smoke metrics verify plumbing only and are not model-quality results.
8. A milestone is complete only when implementation, tests, reproducible command
   and required evidence are all available.

When implementation status changes, update `AI_PROGRESS.md` in the same change.
Update the project plan only when scope or an architectural decision changes.
