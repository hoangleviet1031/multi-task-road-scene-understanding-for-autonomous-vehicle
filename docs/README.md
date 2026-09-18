# RoadSense-MTL documentation index

## Project status

| Area | Status | Source of truth |
|---|---|---|
| Data foundation | Implemented | `milestone_1.md` |
| Independent single-task baselines | Implemented | `milestone_2.md` |
| Conflict-aware multi-task research design | Designed, not implemented | `conflict_aware_research_design_vi.md` |
| Overall project scope and roadmap | Current overview | `project_overview_vi.md` |

Current model input is a `640×384` letterbox canvas (`width × height`). In code
and YAML this is written as `[384, 640]` (`height × width`).

## Documents

- [Project overview](project_overview_vi.md): problem, dataset, current
  architecture, evaluation protocol, repository structure and roadmap.
- [Conflict-aware research design](conflict_aware_research_design_vi.md): the
  normative design for Milestone 3+, including task-specific adapters, PCGrad,
  condition slices, OOD evaluation and ablations.
- [Milestone 1](milestone_1.md): implemented data contract, indexing,
  rasterization, auditing, splitting, metrics and visualization.
- [Milestone 2](milestone_2.md): implemented detection/drivable/lane baselines,
  losses, training engine, checkpoints and verification commands.

## Consistency rules

1. An implementation document describes only code that exists in the repository.
2. Planned components must be explicitly marked `planned` or `not implemented`.
3. `conflict_aware_research_design_vi.md` owns the Milestone 3 research design.
4. `project_overview_vi.md` summarizes that design instead of redefining it.
5. Input size is always described with its coordinate order.
6. Operational Milestone 2 baselines and future capacity-matched research
   controls must not be treated as the same experiment.
7. Smoke metrics verify plumbing only and must not be reported as model quality.

When implementation status changes, update this index, the project overview,
the relevant milestone document and the root README in the same commit.
