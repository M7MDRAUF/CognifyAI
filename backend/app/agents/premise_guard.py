"""The planner's claims about evidence it has not seen (FR-002, the certified track's L-13).

The planner runs before retrieval, so it has seen no evidence, yet it writes that none was
supplied: into its normalized question, which the researcher, critic, risk, decision and evaluator
agents are all given, and into ``risk_focus`` items, which the risk agent is given. Over the final
readiness audit's 65 live runs the claim reached a later agent in 33. These two detectors let the
orchestrator keep it out of those inputs, while the planner's own recorded output keeps it, so the
measurement stays possible.
"""

import re
from typing import Final

# The certified runner's pre-registered detector, `evals.run.PLANNER_CONTAMINATION`, copied
# exactly: the application must not import the evaluation harness, and a unit test pins the two
# patterns and their flags together, so neither can drift from the other.
UNSEEN_EVIDENCE_CLAIM: Final = re.compile(
    r"\bno (?:relevant |policy |supporting )?(?:documents?|chunks?|evidence|sources?|context)\b"
    r".{0,40}?\b(?:were|was|is|are|has been|have been)\b.{0,20}?"
    r"\b(?:supplied|provided|retrieved|available|given|found)\b"
    r"|\binsufficient evidence\b|\bcannot be evidenced\b"
    r"|\b(?:evidence|documents?|chunks?) (?:is|are|was|were) (?:missing|unavailable|absent)\b",
    re.IGNORECASE | re.DOTALL,
)

# For `risk_focus` items only, where "insufficient evidence" can name a genuine risk: a statement
# that no documents, chunks, chunk ids, evidence, sources or context were supplied or provided.
# The detector's first alternative, restricted to "supplied" and "provided", and widened to accept
# `chunk_id` and any one word between "no" and the noun, so it also catches the two recorded forms
# that alternative misses: "no chunk_id= provenance was supplied" and "no retrieved context was
# provided".
EVIDENCE_SUPPLY_CLAIM: Final = re.compile(
    r"\bno\s+(?:[a-z-]+\s+)?(?:(?:documents?|chunks?|evidence|sources?|context)\b|chunk_ids?\b=?)"
    r".{0,40}?\b(?:were|was|is|are|has been|have been)\b.{0,20}?"
    r"\b(?:supplied|provided)\b",
    re.IGNORECASE | re.DOTALL,
)


def claims_unseen_evidence(text: str) -> bool:
    """Whether ``text`` makes the pre-registered detector's claim about evidence."""
    return UNSEEN_EVIDENCE_CLAIM.search(text) is not None


def claims_evidence_supply(text: str) -> bool:
    """Whether ``text`` states that evidence was or was not supplied (a ``risk_focus`` item)."""
    return EVIDENCE_SUPPLY_CLAIM.search(text) is not None
