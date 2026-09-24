"""Backend-neutral correctness prototype for SeedPlane speculative decoding.

This is deliberately a coordinator/reference implementation, not a fast decoder:
the target callback returns logits for one position at a time. It defines the
forest, greedy verification, and commit boundary that an optimized batched
verifier must preserve.
"""
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


@dataclass
class ForestNode:
    token: int
    children: dict[int, "ForestNode"] = field(default_factory=dict)


@dataclass
class SpeculativeForest:
    roots: dict[int, ForestNode] = field(default_factory=dict)
    paths: list[tuple[int, ...]] = field(default_factory=list)

    @classmethod
    def build(cls, branches: Iterable[Iterable[int]]) -> "SpeculativeForest":
        forest = cls()
        for branch in branches:
            path = tuple(int(token) for token in branch)
            if not path:
                continue
            forest.paths.append(path)
            children = forest.roots
            for token in path:
                node = children.get(token)
                if node is None:
                    node = children[token] = ForestNode(token)
                children = node.children
        return forest

    @property
    def node_count(self) -> int:
        def count(children):
            return sum(1 + count(node.children) for node in children.values())
        return count(self.roots)


@dataclass(frozen=True)
class Verification:
    emitted: tuple[int, ...]
    accepted_draft_tokens: int
    rejected: bool
    target_steps: int
    branch: tuple[int, ...]


def verify_greedy(committed: Sequence[int], forest: SpeculativeForest,
                  target_logits: Callable[[tuple[int, ...]], Sequence[float]]) -> Verification:
    """Verify every draft path, then choose the longest exact greedy continuation.

    `target_logits(prefix)` must return target-model logits for the next token.
    On the first disagreement, the target token is emitted as correction. Empty
    forests fall back to one target token. Ties preserve branch input order.
    """
    base = tuple(int(t) for t in committed)
    if not forest.paths:
        logits = target_logits(base)
        if not logits:
            raise ValueError("target returned empty logits")
        token = max(range(len(logits)), key=logits.__getitem__)
        return Verification((token,), 0, False, 1, ())

    best = None
    for path in forest.paths:
        accepted = []
        steps = 0
        rejected = False
        for draft_token in path:
            logits = target_logits(base + tuple(accepted))
            if not logits:
                raise ValueError("target returned empty logits")
            target_token = max(range(len(logits)), key=logits.__getitem__)
            steps += 1
            if draft_token != target_token:
                accepted.append(target_token)
                rejected = True
                break
            accepted.append(draft_token)
        result = Verification(tuple(accepted), len(accepted) - int(rejected),
                              rejected, steps, path)
        rank = (result.accepted_draft_tokens, len(result.emitted))
        if best is None or rank > (best.accepted_draft_tokens, len(best.emitted)):
            best = result
    return best


@dataclass
class CommitLedger:
    """Committed token frontier; speculative state never mutates it pre-commit."""
    tokens: list[int] = field(default_factory=list)
    commits: int = 0

    @property
    def frontier(self) -> int:
        return len(self.tokens)

    def commit(self, verification: Verification) -> tuple[int, ...]:
        if not verification.emitted:
            raise ValueError("cannot commit an empty verification")
        committed = verification.emitted
        self.tokens.extend(committed)
        self.commits += 1
        return committed
