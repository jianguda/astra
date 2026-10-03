"""Evaluation of the techniques in `gap.repair`.

Two protocols, one method interface (`python -m gap.harness -h`):

evolve  -- code-evolution tasks (PyEvo, RustEvo): repair on one sample, generate, execute its tests, restore.
tokens  -- zero-shot corpus (HumanEval): teacher-forced next-token predictions of the reference solutions of a
           held-out part, before and after a repair on another datum (side effects).
"""
