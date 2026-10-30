"""Synthetic tool-use data: the train/eval item split is disjoint, and generator and checker agree on both splits."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import synth_trajectories as S


def teardown_function(_):
    S.set_split("all")


def test_pools_are_disjoint():
    S.set_split("train"); train_items, train_missing = set(S.POOL), set(S.MISSING_POOL)
    S.set_split("eval"); eval_items, eval_missing = set(S.POOL), set(S.MISSING_POOL)
    assert not (train_items & eval_items) and not (train_missing & eval_missing)
    assert eval_items <= set(S.PRICES) and not (eval_missing & set(S.PRICES))


def test_checker_agrees_with_generator_on_every_split():
    for split in ("all", "train", "eval"):
        S.set_split(split)
        rng = random.Random(1)
        for _ in range(400):
            rec = S.bad(rng) if rng.random() < 0.3 else S.good(rng)
            assert S.check(rec) == rec["grade"], (split, rec)


def test_training_tasks_never_name_a_holdout_item():
    S.set_split("train")
    rng = random.Random(2)
    # price and blocked tasks name an item; note tasks are free text ("water the plants") and may share a word
    tasks = [S.good(rng)["task"] for _ in range(500)]
    text = " ".join(t for t in tasks if "Look up the price" in t)
    assert text and not any(item in text for item in S.HOLDOUT_ITEMS + S.MISSING_EVAL)
