import pytest

from slashcompute.web import sample_grants as G

T = 1e12


@pytest.fixture
def book():
    return G.sample_book()


def test_sample_book_has_public_and_pending(book):
    assert G.public(book)
    assert [g.status for g in G.pending(book)] == ["pending"]


def test_fund_updates_grant_and_pledged_without_mutating(book):
    target = G.public(book)[-1]
    after = G.fund(book, target.id, 20 * T, balance=100 * T)
    funded = G.find(after, target.id)
    assert funded.raised == target.raised + 20 * T
    assert funded.backers == target.backers + 1
    assert after.pledged == 20 * T
    assert G.find(book, target.id) == target


@pytest.mark.parametrize("amount, balance, message", [
    (0, 100 * T, "above zero"),
    (20 * T, 10 * T, "available"),
    (10_000 * T, 100_000 * T, "left to reach"),
    (float("nan"), 100 * T, "above zero"),
    (float("inf"), float("inf"), "above zero"),
])
def test_fund_rejects_bad_amounts(book, amount, balance, message):
    with pytest.raises(G.GrantError, match=message):
        G.fund(book, G.public(book)[0].id, amount, balance)


def test_fund_rejects_pending_and_fully_funded(book):
    with pytest.raises(G.GrantError, match="approved"):
        G.fund(book, G.pending(book)[0].id, T, 100 * T)
    g = G.public(book)[0]
    full = G.fund(book, g.id, g.remaining, 10_000 * T)
    with pytest.raises(G.GrantError, match="fully funded"):
        G.fund(full, g.id, T, 10_000 * T)


def test_available_is_share_plus_starter_minus_pledged(book):
    assert G.available(book, 10 * T) == 10 * T + G.STARTER_FLOPS
    spent = G.fund(book, G.public(book)[0].id, 30 * T, 1_000 * T)
    assert G.available(spent, 10 * T) == 10 * T + G.STARTER_FLOPS - 30 * T


def test_request_creates_pending_grant(book):
    after = G.request(book, "  GPU for my thesis ", "Train a parser", 120 * T)
    new = after.grants[-1]
    assert (new.title, new.status, new.raised) == ("GPU for my thesis", "pending", 0.0)
    assert new not in G.public(after)
    for title, summary, goal, msg in [("", "x", 1, "title"), ("t", " ", 1, "Describe"),
                                      ("t", "x", 0, "goal"),
                                      ("t", "x", float("nan"), "goal"),
                                      ("t", "x", float("inf"), "goal")]:
        with pytest.raises(G.GrantError, match=msg):
            G.request(book, title, summary, goal)


def test_review_approves_or_declines_once(book):
    gid = G.pending(book)[0].id
    approved = G.review(book, gid, approve=True)
    assert G.find(approved, gid) in G.public(approved)
    declined = G.review(book, gid, approve=False)
    assert G.find(declined, gid) not in G.public(declined) and not G.pending(declined)
    with pytest.raises(G.GrantError, match="review"):
        G.review(approved, gid, approve=True)


def test_sort_modes(book):
    top = G.public(book, "top")
    assert [g.raised for g in top] == sorted((g.raised for g in top), reverse=True)
    trending = G.public(book, "trending")
    assert [g.recent for g in trending] == sorted((g.recent for g in trending), reverse=True)
    least = G.public(book, "least")
    assert [g.progress for g in least] == sorted(g.progress for g in least)


def test_board_payload(book):
    b = G.board(book, "top", grant_share=5 * T)
    assert b["sample"] is True and b["share"] == 5 * T
    assert b["available"] == 5 * T + G.STARTER_FLOPS
    assert {"progress", "remaining"} <= set(b["grants"][0])
