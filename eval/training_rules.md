# Training Rules
Hard rules escalated from eval_learnings.md by EvalContextLoop.
A rule is added automatically when the same failure pattern appears in
three or more eval runs for the same training phase and modality.

Future eval runs load this file first via select_context() and flag any
result that would have triggered a known hard rule before reporting it as clean.

