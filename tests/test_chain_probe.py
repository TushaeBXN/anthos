import torch

from chain_probe import ARROW, ASK, K0, MAX_HOPS, N_NODES, NODE0, make_batch


def _replay(seq):
    """Re-derive the answer from the tokens alone, one hop at a time."""
    body, q = seq[: 2 * N_NODES], seq[2 * N_NODES:]
    nxt = {int(body[i]) - NODE0: int(body[i + 1]) - NODE0 for i in range(0, len(body), 2)}
    assert int(q[0]) == ASK and int(q[3]) == ARROW
    cur = int(q[1]) - NODE0
    for _ in range(int(q[2]) - K0):
        cur = nxt[cur]
    return cur


def test_the_answer_key_matches_a_replay_of_the_tokens():
    seq, ans, k = make_batch(200, torch.Generator().manual_seed(0))
    assert all(_replay(s) == int(a) for s, a in zip(seq, ans))
    assert set(k.tolist()) == set(range(1, MAX_HOPS + 1))


def test_every_node_is_a_source_exactly_once():
    seq, _, _ = make_batch(8, torch.Generator().manual_seed(1))
    assert all(sorted((s[0:2 * N_NODES:2] - NODE0).tolist()) == list(range(N_NODES)) for s in seq)


def test_a_fixed_hop_count_is_respected():
    seq, _, k = make_batch(32, torch.Generator().manual_seed(2), hops=3)
    assert (k == 3).all() and (seq[:, -2] == K0 + 3).all()
