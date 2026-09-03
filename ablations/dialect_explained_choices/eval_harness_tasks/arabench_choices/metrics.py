"""Multi-correct target for the arabench_choices ablation variant.

Each row's `accept_choices` lists every AraBench label that should count as
correct for that row's coarse ADD class. `target_indices` maps those labels
into indices over `choices`, which lm-eval consumes via `doc_to_target` and
auto-activates `multiple_target` mode (any pred-in-gold counts as correct).

YAML reference: `doc_to_target: !function metrics.target_indices`
"""


def target_indices(doc):
    return [list(doc["choices"]).index(c) for c in doc["accept_choices"]]
