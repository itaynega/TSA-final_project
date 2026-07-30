"""Shared foundation for both stages of the project.

Everything in here is imported by the reconstruction *and* by the improvement.
That is the point: requirement C2 is pass/fail on the improved method using the
identical split and the identical metrics as the reconstruction, and the cheapest
way to guarantee that is for there to be only one copy of each.

Ownership, per the project's one-writer-per-file rule:

    data.py     Itay
    split.py    Itay
    metrics.py  Amitay
    results.py  Amitay
"""
