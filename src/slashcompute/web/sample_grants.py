"""Sample community grants for the Grants tab.

The real grant system (``community.grants``) needs accounts, which are off
for now. This keeps the screens usable with demo data held in the shell's
memory. Amounts are FLOPs, matching the 1:1 FLOP credits. Every operation
returns a new ``GrantBook``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Optional

from slashcompute.launcher.dashboard import with_unit

T = 1e12
# Lets people try funding before their own contributions have earned anything.
STARTER_FLOPS = 250 * T

SORTS = ("top", "trending", "least")


class GrantError(ValueError):
    """User-facing reason a grant action was refused."""


@dataclass(frozen=True)
class Grant:
    id: str
    title: str
    author: str
    summary: str
    goal: float
    raised: float = 0.0
    backers: int = 0
    # FLOPs raised in the last week; drives the "trending" sort.
    recent: float = 0.0
    tag: str = "Project"
    status: str = "approved"  # pending | approved | declined

    @property
    def progress(self) -> float:
        return 0.0 if self.goal <= 0 else min(1.0, self.raised / self.goal)

    @property
    def remaining(self) -> float:
        return max(0.0, self.goal - self.raised)

    def view(self) -> dict:
        return {**asdict(self), "progress": self.progress, "remaining": self.remaining}


@dataclass(frozen=True)
class GrantBook:
    grants: tuple[Grant, ...]
    pledged: float = 0.0


def sample_book() -> GrantBook:
    return GrantBook(grants=(
        Grant("g1", "Irish-language study buddy", "Aoife M.",
              "Fine-tune a small model on Gaeilge study notes so first-years can practise "
              "conversation outside class.", goal=400 * T, raised=312 * T, backers=19,
              recent=90 * T, tag="Education"),
        Grant("g2", "Lecture summaries for dyslexic students", "Tomás R.",
              "Turn recorded lectures into short, plain-language summaries with key terms "
              "highlighted.", goal=600 * T, raised=210 * T, backers=11, recent=140 * T,
              tag="Accessibility"),
        Grant("g3", "Crop-disease spotter for community gardens", "Priya K.",
              "Train a leaf-photo classifier that volunteers can run on an old phone to catch "
              "blight early.", goal=800 * T, raised=95 * T, backers=4, recent=20 * T,
              tag="Research"),
        Grant("g4", "Open homework helper for secondary schools", "Ciarán D.",
              "A tutor that explains maths steps instead of giving answers, tuned on "
              "teacher-written worked examples.", goal=500 * T, raised=455 * T, backers=31,
              recent=30 * T, tag="Education"),
        Grant("g5", "Local-history guide for the city library", "Maeve O.",
              "Answer visitor questions from the library's digitised archive of street "
              "photos and parish records.", goal=300 * T, raised=48 * T, backers=3,
              recent=48 * T, tag="Community"),
        Grant("g6", "Vision model for a robotics-club sorting arm", "Jonah L.",
              "Teach a low-cost arm to sort recycling by material using a camera and a "
              "tiny on-board model.", goal=350 * T, raised=160 * T, backers=9, recent=75 * T,
              tag="Robotics"),
        Grant("g7", "Sign-language gesture captions", "Lena W.",
              "Caption short ISL clips so deaf students can search a video library by sign.",
              goal=700 * T, tag="Accessibility", status="pending"),
    ))


def available(book: GrantBook, grant_share: float) -> float:
    """FLOPs you can still pledge: your grant share plus the starter allowance."""
    return max(0.0, grant_share + STARTER_FLOPS - book.pledged)


def find(book: GrantBook, grant_id: str) -> Optional[Grant]:
    return next((g for g in book.grants if g.id == grant_id), None)


def _swap(book: GrantBook, new: Grant, **changes) -> GrantBook:
    return replace(book, grants=tuple(new if g.id == new.id else g for g in book.grants),
                   **changes)


def fund(book: GrantBook, grant_id: str, amount: float, balance: float) -> GrantBook:
    g = find(book, grant_id)
    if g is None:
        raise GrantError("That grant no longer exists.")
    if g.status != "approved":
        raise GrantError("Only approved grants can be funded.")
    if not math.isfinite(amount) or amount <= 0:
        raise GrantError("Enter an amount above zero.")
    if g.remaining <= 0:
        raise GrantError("This grant is already fully funded.")
    if amount > g.remaining:
        raise GrantError(f"Only {with_unit(g.remaining)} left to reach the goal.")
    if amount > balance:
        raise GrantError(f"You have {with_unit(balance)} available for grants.")
    funded = replace(g, raised=g.raised + amount, backers=g.backers + 1,
                     recent=g.recent + amount)
    return _swap(book, funded, pledged=book.pledged + amount)


def request(book: GrantBook, title: str, summary: str, goal: float,
            author: str = "You") -> GrantBook:
    title, summary = title.strip(), summary.strip()
    if not title:
        raise GrantError("Give your grant a title.")
    if not summary:
        raise GrantError("Describe what the compute is for.")
    if not math.isfinite(goal) or goal <= 0:
        raise GrantError("Set a goal above zero.")
    new = Grant(id=f"g{len(book.grants) + 1}", title=title, author=author, summary=summary,
                goal=float(goal), tag="New", status="pending")
    return replace(book, grants=book.grants + (new,))


def review(book: GrantBook, grant_id: str, approve: bool) -> GrantBook:
    g = find(book, grant_id)
    if g is None or g.status != "pending":
        raise GrantError("That grant is not waiting for review.")
    return _swap(book, replace(g, status="approved" if approve else "declined"))


def sort_grants(grants: list[Grant], mode: str) -> list[Grant]:
    if mode == "trending":
        return sorted(grants, key=lambda g: (-g.recent, g.title))
    if mode == "least":
        return sorted(grants, key=lambda g: (g.progress, g.title))
    return sorted(grants, key=lambda g: (-g.raised, g.title))


def public(book: GrantBook, mode: str = "top") -> list[Grant]:
    return sort_grants([g for g in book.grants if g.status == "approved"], mode)


def pending(book: GrantBook) -> list[Grant]:
    return [g for g in book.grants if g.status == "pending"]


def board(book: GrantBook, mode: str, grant_share: float) -> dict:
    """The Grants tab payload."""
    return {
        "sample": True,
        "grants": [g.view() for g in public(book, mode)],
        "pending": [g.view() for g in pending(book)],
        "pledged": book.pledged,
        "starter": STARTER_FLOPS,
        "share": grant_share,
        "available": available(book, grant_share),
    }
