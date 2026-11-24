"""Grounding: claim-level citations and a deterministic support check (PLV7-T02).

The definitions are those of ``docs/pharma-live/evaluation/methodology.md``, section 5:

* a **claim** is one sentence of an evidence-bearing field, split with the policy's sentence
  splitter (``policy.split_sentences``, which keeps a citation marker with its sentence);
* a claim is **not substantive** when it is a fixed text of the application, a question, a
  statement that information is missing, or has no content token;
* a substantive claim **cites** the passages its inline markers (``[E1]``, ``[E1, E2]``; a key is
  any the response recognizes: letters, digits, ``_`` and ``-``, up to 20 characters) name and
  the passages its item cites (a procedure's or a risk's ``citation_keys``);
* it is **supported** when at least ``SUPPORT_THRESHOLD`` of its content tokens (citation markers
  removed) occur in a single cited passage: the previous track's rule, with its token definition
  (``evals.pharma.responders.content_tokens``, which a test holds equal), so the figures compare;
* PLV-D-025 (PLV7-T04) adds two clauses, because the token rule never compares a token shorter
  than three characters: the same passage must also hold every numeric value the claim states
  (``claim_values``; a claim whose tokens a passage holds but whose values none holds is
  ``value_not_found``), and a passage the signal extractor reads as text addressed to the model
  (``signals.suspected_instruction``) supports no claim;
* on the application's path (RMD3-T03: DV-RMD-14, DV-RMD-20) a cited passage's register-label
  words are neutral and a word's -ing form compares as the word; the grader keeps the rule above.

``check_grounding`` is linear in the text it reads: every passage is read once. Its record holds
counts, reason codes and SHA-256 digests, never a claim's text.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from app.pharma.policy import split_sentences
from app.pharma.signals import suspected_instruction

SUPPORT_THRESHOLD: Final = 0.6
# The grader's fixed list (evals.pharma.responders.STOP_WORDS), copied so the application does not
# import the evaluation package; a test holds the two equal.
STOP_WORDS: Final = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "then", "than", "from", "into", "onto",
        "are", "was", "were", "been", "being", "have", "has", "had", "not", "but", "its", "our",
        "their", "they", "them", "there", "these", "those", "what", "which", "who", "whom",
        "when", "where", "why", "how", "does", "did", "done", "can", "could", "should", "would",
        "will", "shall", "may", "might", "must", "any", "all", "each", "every", "some", "such",
        "only", "also", "very", "more", "most", "other", "about", "before", "after", "over",
        "under", "between", "during", "while", "upon", "you", "your", "his", "her", "she",
    }
)  # fmt: skip
_TOKEN: Final = re.compile(r"[a-z0-9]+")
# A key is what the response itself recognizes (``response._KEY_IN_TEXT``), alone or in a list.
_MARKER: Final = re.compile(r"\[([A-Za-z0-9_-]{1,20}(?:\s*,\s*[A-Za-z0-9_-]{1,20})*)\]")
_MISSING: Final = re.compile(
    r"\b(?:not (?:stated|specified|found|provided|available|given|covered|described)"
    r"|no (?:document|passage|evidence|procedure|record)"
    r" (?:states|gives|specifies|covers|describes)"
    r"|do(?:es)? not (?:state|specify|say|mention|cover|give|describe))\b",
    re.IGNORECASE,
)

SUPPORTED: Final = "supported"
NO_CITATION: Final = "no_citation"
NOT_ENTAILED: Final = "not_entailed"
NOT_SUBSTANTIVE: Final = "not_substantive"
VALUE_NOT_FOUND: Final = "value_not_found"

_NUMBER: Final = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
_LEADING: Final = re.compile(r"^[^A-Za-z0-9]+")
_REVISION_WORDS: Final = frozenset({"revision", "rev"})


def content_tokens(text: str) -> set[str]:
    """Lower-case alphanumeric words of three or more characters, minus the stop words."""
    return {t for t in _TOKEN.findall(text.lower()) if len(t) >= 3 and t not in STOP_WORDS}


def ing_forms(token: str) -> frozenset[str]:
    """``token`` and the word its -ing form pairs with (DV-RMD-20), for the gate's held comparison
    only: "removing" and "remove", "setting" and "set", "filling" and "fill" compare as one word.
    Plural and past forms are not paired: "the tests passed" is not "the test" (a permission the
    passage does not grant was held with them)."""
    forms = {token}
    base = token[:-3]
    if token.endswith("ing") and len(base) >= 3:
        forms |= {base, base + "e"}
        if len(base) >= 4 and base[-1] == base[-2] and base[-1] not in "aeiou":
            forms.add(base[:-1])
    else:
        forms |= {token + "ing", token + token[-1] + "ing"}
        if token.endswith("e"):
            forms.add(token[:-1] + "ing")
    return frozenset(forms)


def _decimals(numbers: Sequence[str]) -> set[Decimal]:
    return {Decimal(number.replace(",", ".")) for number in numbers}


def claim_values(sentence: str) -> frozenset[Decimal]:
    """The numeric values a claim states (PLV-D-025): the numbers of each word that starts with a
    digit ("45", "45N", "2.0-2.2", "10:41"), citation markers removed. A number inside a word that
    starts with a letter (VF-310, SOP-SYN-FIL-310, A-202) is an identifier, and the word after
    "revision" or "rev" names a revision; neither is a value. Values compare as decimals."""
    values: set[Decimal] = set()
    previous = ""
    for raw in _MARKER.sub(" ", sentence).split():
        word = _LEADING.sub("", raw)
        if word[:1].isascii() and word[:1].isdigit() and previous not in _REVISION_WORDS:
            values |= _decimals(_NUMBER.findall(word))
        previous = word.rstrip(".,:;").casefold()
    return frozenset(values)


def passage_values(text: str) -> frozenset[Decimal]:
    """Every number of a passage's text, as a decimal."""
    return frozenset(_decimals(_NUMBER.findall(text)))


@dataclass(frozen=True)
class PassageLabel:
    """A cited passage's register label, from the stored anchors (DV-RMD-14): what names it."""

    document_number: str
    revision: str
    lifecycle_status: str

    def words(self) -> frozenset[str]:
        """The content tokens a claim names the passage with: its document number, its revision
        as written in prose (``revision 01``, ``rev 01``, ``rev01``) and its lifecycle status."""
        revision = self.revision.lower()
        return frozenset(
            content_tokens(
                f"{self.document_number} revision rev {revision} rev{revision} "
                f"{self.lifecycle_status}"
            )
        )

    def names_document(self, sentence: str) -> bool:
        # An uncontrolled passage's number is "unknown" (``anchors``): the word names no document
        # (RMD-DEF-055).
        if self.document_number.strip().casefold() in {"", "unknown"}:
            return False
        pattern = rf"(?<![A-Za-z0-9-]){re.escape(self.document_number)}(?![A-Za-z0-9-])"
        return re.search(pattern, sentence, re.IGNORECASE) is not None


@dataclass(frozen=True)
class ClaimSource:
    """One evidence-bearing text and the citation keys its item carries."""

    field: str
    text: str
    citation_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimCheck:
    field: str
    sentence: str
    cited: tuple[str, ...]
    substantive: bool
    supported: bool
    reason: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.sentence.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GroundingResult:
    claims: tuple[ClaimCheck, ...]
    # Cited passages read as text addressed to the model, which support no claim (PLV-D-025).
    instruction_passages: int = 0

    @property
    def unsupported(self) -> tuple[ClaimCheck, ...]:
        return tuple(c for c in self.claims if c.substantive and not c.supported)

    def record(self) -> dict[str, object]:
        """Counts, reason codes and digests: what a run record keeps (no claim text)."""
        return {
            "claims": len(self.claims),
            "substantive": sum(1 for c in self.claims if c.substantive),
            "unsupported": len(self.unsupported),
            "reasons": dict(sorted(Counter(c.reason for c in self.claims).items())),
            "unsupported_digests": [c.digest for c in self.unsupported],
            "instruction_passages": self.instruction_passages,
        }


def _cited(sentence: str, item_keys: Sequence[str]) -> tuple[str, ...]:
    keys: list[str] = []
    for marker in _MARKER.findall(sentence):
        keys += [key.strip() for key in marker.split(",")]
    keys += list(item_keys)
    return tuple(dict.fromkeys(keys))


# DV-RMD-24: the joints at which a sentence's clauses meet ("X; whether Y is not stated", "X, but
# the evidence does not specify Y").
_CLAUSE: Final = re.compile(r";\s*|,?\s+(?:but|while|whereas|although|though)\s+", re.IGNORECASE)
# RMD-DEF-029 and RMD-DEF-110 (``strict``): a sentence is cut into clauses: at a semicolon, a
# colon, a clause joint ("but", "while", "although", "because" ...), an "and" before a negated verb,
# and a comma before a word that opens a clause of its own or a consequence ("so", "therefore"). A
# comma before any other word goes on with the clause (a list: "does not state whether A, B or C are
# defined"), and a comma inside a number ("100,000") is no cut. A sentence with no clause stating
# missing information is the claim whole. In one with such a clause, the clause is set aside and
# the claim is each other clause that states a value or a permission or obligation ("Although X is
# not stated, the press may be held for 96 hours", "X is 72 hours, so the press may be held for 96
# hours, as Z is not stated"); a clause stating neither is the missing statement's context ("...,
# so the question is not answered", "The checklist lists the punches but does not specify ...").
_PIECE_STRICT: Final = re.compile(
    r"(;\s*|:\s+|,\s+|\s+(?=(?:but|while|whereas|however|although|though|because|since)\s)"
    r"|\s+(?=and\s+(?:(?:does|do|is|are|was|were|has|have)\s+not|no|never)\b)"
    # Round 3 (RMD-DEF-110): an "and" right after a missing-information predicate ("The dirty hold
    # time is not stated and the press may be held for 96 hours").
    r"|(?:(?<=\bstated)|(?<=\bspecified)|(?<=\bprovided)|(?<=\bgiven)|(?<=\bdescribed)|"
    r"(?<=\bknown)|(?<=\blisted)|(?<=\bdocumented)|(?<=\bunknown)|(?<=\bmissing))\s+(?=and\s))",
    re.IGNORECASE,
)
# Round 3 (RMD-DEF-110): a piece after a comma with a predicate of its own ("..., TP-201 may be
# held for 96 hours") opens a clause, unless it goes on with a list ("..., or what the entry must
# contain"); a clause opened by "but", "however", "yet", "whereas" or "while" says something of its
# own and is checked whatever it states. A clause still without a predicate takes the pieces that
# follow it ("but its document number, revision and status are not given").
_OWN_PREDICATE: Final = re.compile(
    r"\b(?:may|must|can|could|should|shall|will|might)\b", re.IGNORECASE
)
_LIST_ITEM: Final = re.compile(
    r"^(?:or|and|nor|what|who|whom|whose|how|which|whether|where|when|why)\b", re.IGNORECASE
)
_HAS_PREDICATE: Final = re.compile(
    r"\b(?:may|must|can|could|should|shall|will|might|is|are|was|were|has|have|had|be|been|"
    r"requires?|states?|sets?|gives?|lists?|covers?|shows?|takes?)\b",
    re.IGNORECASE,
)
_CONTRAST: Final = re.compile(r"^(?:but|however|yet|whereas|while)\b", re.IGNORECASE)
# A clause about the evidence that negates states what it lacks ("no such record is included in the
# evidence", "neither passage states which governs", "are not given in the register").
_ABOUT_EVIDENCE: Final = re.compile(
    r"\b(?:evidence|passages?|register|retrieved text|sources provided|documents provided)\b",
    re.IGNORECASE,
)
_NEGATED: Final = re.compile(r"\b(?:no|not|none|neither|nor|cannot|without)\b|n't\b", re.IGNORECASE)
_OWN_CLAUSE: Final = frozenset(
    {
        "and", "but", "yet", "while", "whereas", "although", "though", "however", "because",
        "since", "as", "the", "this", "that", "these", "those", "it", "its", "they", "there",
        "a", "an", "if", "when", "once", "unless", "until", "then", "each", "every", "all", "no",
        "qa", "qc", "operators", "staff",
    }
)  # fmt: skip
_CONSEQUENCE: Final = re.compile(
    r"^(?:so|therefore|hence|thus|consequently|meaning|which means)\b", re.IGNORECASE
)
# What a clause beside a missing-information statement must hold to be checked: a value, or a
# permission or obligation it does not negate ("so neither value can be presented" grants none).
_VALUE_DIGIT: Final = re.compile(r"(?<![\w.,-])\d")
_MODAL: Final = re.compile(
    r"\b(?:may|can|could|must|should|shall|requires?|required|permitted|allowed)\b", re.IGNORECASE
)
# Words that present a claim rather than state it ("72 hours in one effective document [E4] and 60
# hours in another"): under ``strict`` they count neither for nor against it.
_PRESENTATION: Final = frozenset(
    {
        "document", "documents", "passage", "passages", "evidence", "one", "another", "given",
        "stated", "states", "state", "says", "listed", "lists", "per", "respectively",
    }
)  # fmt: skip


# A modal the clause negates grants nothing ("neither value can be presented", "no single value can
# be given"); "without" beside it negates its object, not the modal (RMD-DEF-110, round 3).
_NEGATED_MODAL: Final = re.compile(
    r"\b(?:cannot|can't|couldn't|mustn't|shouldn't|won't|neither|no|nothing|none)\b|"
    r"\b(?:may|must|can|could|should|shall|will|might) not\b",
    re.IGNORECASE,
)


def _actionable(clause: str) -> bool:
    return bool(_VALUE_DIGIT.search(clause)) or bool(
        _MODAL.search(clause) and not _NEGATED_MODAL.search(clause)
    )


# What states a fact the evidence lacks. "Not found", "not available", "not covered" and "not
# recorded" say so only of the evidence ("not found in the passages"): otherwise they are the
# procedure's own words ("materials not found on the approved supplier list", "equipment not
# covered by a valid calibration", "if the balance is not available").
# Round 3 (RMD-DEF-029): "listed", "documented", "defined", "included", "indicated" and "mentioned"
# say information is missing as a clause's predicate ("the cleaning agent is not listed"), not as
# the procedure's attribute ("materials not listed on the approved supplier list"); "does not
# include" does with a subject or after a joint ("the passages do not include", "but does not
# specify"), not as an imperative ("Do not include rejected tablets"); "specifies no" is no bound
# ("specifies no more than 96 hours").
_TELLS: Final = (
    r"(?:state|specify|say|mention|cover|give|describe|list|define|include|show|confirm|indicate|"
    r"establish|contain|address)"
)
_MISSING_STRICT: Final = re.compile(
    r"\b(?:not (?:stated|specified|provided|given|described)"
    r"|(?:is|are|was|were) not (?:listed|documented|mentioned|defined|indicated|included)"
    r"(?! (?:on|under|for)\b)"
    r"|not (?:found|available|covered|recorded|listed|documented|mentioned|defined|indicated|"
    r"included) (?:in|from|within|by|on) (?:the |any |these |this )?(?:passages?|evidence|"
    r"documents?|context|sources?|procedures?|records?|text|register)\b"
    r"|no (?:document|passage|evidence|procedure|record|source)s? (?:states?|gives?|specif(?:y|ies)"
    r"|covers?|describes?|mentions?|lists?)"
    rf"|(?<=\w)\s+do(?:es)? not {_TELLS}"
    rf"|^(?:but|and|which|that|yet)\s+do(?:es)? not {_TELLS}"
    r"|(?:states?|gives?|lists?|specif(?:y|ies)|mentions?) no(?! (?:more|less|later|fewer|greater|"
    r"higher|lower|earlier) than)"
    r"|(?:is|are|remains?) (?:unknown|not known|missing|unresolved)"
    # A record the evidence lacks ("no cleaning record is present"; round 3).
    r"|no (?:[\w-]+ ){0,3}(?:records?|entr(?:y|ies)|logs?|data|results?|documents?)"
    r" (?:is|are|was|were) (?:present|available|attached|provided|included|given|shown))\b",
    re.IGNORECASE,
)


def _states_missing(clause: str) -> bool:
    return bool(_MISSING_STRICT.search(clause)) or bool(
        _ABOUT_EVIDENCE.search(clause) and _NEGATED.search(clause)
    )


def _strict_claim(bare: str) -> str:
    """The claim a sentence makes under ``strict`` (the clause rule above ``_PIECE_STRICT``)."""
    parts = _PIECE_STRICT.split(_plain(bare))
    clauses: list[str] = []
    separator = ";"
    for i, part in enumerate(parts):
        if i % 2:
            separator = part
            continue
        if not part.strip():
            continue
        first = part.split(maxsplit=1)[0].casefold().strip("([")
        opens = (
            not clauses
            or separator.strip() in {"", ";", ":"}
            or first in _OWN_CLAUSE
            or _CONSEQUENCE.match(part.strip()) is not None
            or (_OWN_PREDICATE.search(part) is not None and _LIST_ITEM.match(part) is None)
        )
        # A missing-information piece after a comma is a clause of its own ("..., which the
        # passages do not state") when the clause before it already says something.
        if not opens and _states_missing(part):
            opens = _HAS_PREDICATE.search(clauses[-1]) is not None
        # A piece that closes a bracket the clause opened goes on with it: the clause's subject is
        # the list ("Whether X (A, B, C) is not stated").
        if clauses and clauses[-1].count("(") > clauses[-1].count(")") and ")" in part:
            opens = False
        # A clause with no predicate yet takes the piece that gives it one.
        if (
            opens
            and clauses
            and separator.strip() == ","
            and _HAS_PREDICATE.search(clauses[-1]) is None
            and _states_missing(part)
        ):
            opens = False
        if opens:
            clauses.append(part)
        else:
            clauses[-1] = f"{clauses[-1]}, {part}"
    missing = [_states_missing(clause) for clause in clauses]
    if not any(missing):
        return " ".join(clauses)
    return " ".join(
        clause
        for clause, lacks in zip(clauses, missing, strict=True)
        if not lacks and (_actionable(clause) or _CONTRAST.match(clause.strip()) is not None)
    )


def _asserted(bare: str, strict: bool = False) -> str:
    """The clauses of ``bare`` that do not state missing information, joined (RMD-DEF-006)."""
    if not strict:
        parts = _CLAUSE.split(bare)
        return " ".join(part for part in parts if part.strip() and not _MISSING.search(part))
    return _strict_claim(bare)


# RMD13-T02 (``strict``, the application's path): what the token share cannot see. The checks read
# the text with the characters a model and a passage write for one thing made one: a curly
# apostrophe, the minus sign and an en dash, the micro sign (RMD-DEF-111, 115, 027).
_SAME_CHARACTER: Final = str.maketrans(
    {chr(0x2019): "'", chr(0x2018): "'", chr(0x2212): "-", chr(0x2013): "-", chr(0xB5): "u",
     chr(0x3BC): "u"}
)  # fmt: skip
_WORDS: Final = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# RMD-DEF-014 and RMD-DEF-111: the words that negate a clause, compared by the parity of their
# count within the passage clause a claim's clause matches best.
_NEGATORS: Final = frozenset(
    {
        "not", "no", "never", "without", "cannot", "nor", "neither", "none", "prohibited",
        "forbidden", "don't", "doesn't", "didn't", "mustn't", "shouldn't", "can't", "won't",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "couldn't",
        "wouldn't", "needn't",
    }
)  # fmt: skip
# Bounds and orders read as what they say before negators are counted ("no more than 72 hours" is
# at most 72; "prior to" is before; RMD-DEF-112), so a bound is no negation; the bound symbols and
# their abbreviations are the same words, and "has to" or "needs to" is "must" (round 3).
_LE, _GE = chr(0x2264), chr(0x2265)
_READINGS: Final = (
    (re.compile(rf"<=|{_LE}|\bnmt\b"), " atmost "),
    (re.compile(rf">=|{_GE}|\bnlt\b"), " atleast "),
    (re.compile(r"<"), " under "),
    (re.compile(r">"), " over "),
    (re.compile(r"\b(?:no|not) more than\b|\bnot (?:to )?(?:be )?exceed(?:s|ed|ing)?\b|\bup to\b"
                r"|\b(?:a )?maximum of\b|\bat most\b"), " atmost "),
    (re.compile(r"\b(?:no|not) (?:less|fewer) than\b|\b(?:a )?minimum of\b|\bat least\b"),
     " atleast "),
    (re.compile(r"\b(?:no|not) later than\b"), " by "),
    (re.compile(r"\b(?:more|greater|higher) than\b|\bexceeds?\b|\babove\b|\bover\b"), " over "),
    (re.compile(r"\b(?:less|fewer|lower) than\b|\bbelow\b|\bunder\b"), " under "),
    (re.compile(r"\bprior to\b|\bahead of\b"), " before "),
    (re.compile(r"\bsubsequent to\b|\bupon\b"), " after "),
    (re.compile(r"\b(?:has|have|need|needs) to\b"), " must "),
)  # fmt: skip
_OPPOSITES: Final = (
    ("over", "under"), ("atmost", "atleast"), ("over", "atmost"), ("under", "atleast"),
    ("maximum", "minimum"), ("increase", "decrease"),
)  # fmt: skip
_PERMISSION: Final = frozenset({"may", "can", "could", "optional", "optionally", "permitted"})
_OBLIGATION: Final = frozenset({"must", "shall", "required", "mandatory"})
# Negated, a word of obligation says there is none ("not required") and a word of permission or
# obligation forbids ("not permitted", "must not"): the two are opposites (round 3).
_NO_OBLIGATION: Final = frozenset({"required", "mandatory", "necessary", "need", "needs"})
_FORBIDS: Final = frozenset({"permitted", "allowed", "may", "can", "must", "shall"})
# A clause: a sentence cut at semicolons, colons and an "and" that opens a negated step ("Clean the
# press after each batch and do not use compressed air"). Commas do not cut: a negation governs
# its whole list ("Do not use compressed air, brushes or solvents"; round 3).
_FRAGMENT: Final = re.compile(
    r";\s*|:\s+|\s+and\s+(?=(?:do not|don't|never|not|no)\b)|,\s+(?=(?:but|however|yet|whereas|"
    r"while|although|though)\b)",
    re.IGNORECASE,
)
# What a role, a site or a grade is compared within: the parts a sentence joins with "and" or a
# comma ("QA verifies the record and QC tests the samples"; round 3).
_ROLE_PART: Final = re.compile(r";\s*|,\s+|\s+and\s+", re.IGNORECASE)
_ORDER_WORD: Final = re.compile(r"\b(before|after)\b")
# RMD-DEF-027 and RMD-DEF-114: the roles, the lettered designations and the equipment tags that
# change a claim's meaning. A document number's parts ("SOP-SYN-QA-114") name no role.
_ROLE_FORMS: Final = (
    ("quality assurance", "qa"), ("quality control", "qc"), ("qualified person", "qp"),
    ("qa", "qa"), ("qc", "qc"), ("qp", "qp"),
)  # fmt: skip
_DESIGNATION: Final = re.compile(r"\b(site|grade|class|zone)\s+([a-d]|[0-9]{1,3})\b", re.IGNORECASE)
# A tag is written "TP-201", and in capitals "TP201" or "TP 201" (round 3).
_TAG: Final = re.compile(
    r"(?<![A-Za-z0-9-])(?:([A-Za-z]{1,4})-|([A-Z]{1,4}) ?)([0-9]{2,5})(?![A-Za-z0-9-])"
)
# A tag the claim names only to exclude it ("not for TP-305", "other than TP-202") is no swap.
_EXCLUDED_TAG: Final = re.compile(
    r"\b(?:not (?:for|to|on|in|at|with)|except(?: for)?|other than|excluding|rather than|"
    r"instead of|unlike)\s+(?:the\s+)?(?:[a-z]+\s+){0,2}([A-Za-z]{1,4})[- ]?([0-9]{2,5})",
    re.IGNORECASE,
)
_IDENTIFIER: Final = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-[A-Za-z]*\d[\w-]*")
# RMD-DEF-028: the revision a claim names, as a number, a capital letter or both ("A1"), in brackets
# or as "v02"; "revision a" is the article (round 3). A revision named right after a document
# number is that document's.
_CLAIMED_REVISION: Final = re.compile(
    r"\b(?i:revision|rev\.?|version|ver\.?)\s*\(?\s*([0-9]+[A-Za-z]?|[A-Z][0-9]?)\s*\)?(?![\w-])|"
    r"\b(?i:v)([0-9]+)\b"
)
_DOCUMENT_REVISION: Final = re.compile(
    r"\b([A-Za-z]{2,5}(?:-[A-Za-z0-9]+)+)\s+(?i:revision|rev\.?|version|ver\.?)\s*\(?\s*"
    r"([0-9]+[A-Za-z]?|[A-Z][0-9]?)\b"
)
# RMD-DEF-027, 057, 109 and 115: numbers with thousands separators, signs and the units after them.
_NUMBER_STRICT: Final = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
_PLAIN_NUMBER: Final = re.compile(r"^[0-9]+(?:[.,][0-9]+)?$")
_THOUSANDS: Final = re.compile(r"^[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?$")
_SIGNED: Final = re.compile(rf"^[(\[<>={_LE}{_GE}]*-(?=[0-9])")
# Round 3 (RMD-DEF-115, 027): a sign written apart ("- 20", "minus 20") and a tolerance ("40 ± 2
# kN", whose 2 is no value of its own).
_SIGN_WORDS: Final = frozenset({"-", "minus"})
_TOLERANCE: Final = frozenset({chr(0xB1), "+/-"})
# A date is compared as written (RMD-DEF-109 left it no value; round 3).
_DATE: Final = re.compile(
    r"(?<![\d.])(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})(?!\d)(?!\.\d)|\b\d{4}-\d{2}-\d{2}\b"
)
_UNITS: Final = {
    "h": "hour", "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
    "min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute",
    "s": "second", "sec": "second", "secs": "second", "second": "second", "seconds": "second",
    "d": "day", "day": "day", "days": "day", "week": "week", "weeks": "week",
    "month": "month", "months": "month", "year": "year", "years": "year",
    "ug": "ug", "mcg": "ug", "ng": "ng", "mg": "mg", "g": "g", "kg": "kg", "ml": "ml", "l": "l",
    "mm": "mm", "cm": "cm", "kn": "kn", "n": "n", "kp": "kp", "kpa": "kpa", "mbar": "mbar",
    "bar": "bar", "rpm": "rpm", "c": "celsius", "°c": "celsius", "°f": "fahrenheit",
    "%": "percent", "percent": "percent", "ppm": "ppm",
    "ul": "ul", "nl": "nl", "pa": "pa", "um": "um", "nm": "nm", "ppb": "ppb", "cfu": "cfu",
    "iu": "iu",
}  # fmt: skip
_LABEL_WORDS: Final = 4
_NOT_UNIT: Final = re.compile(r"[^a-z%°]")
_UNIT_WORDS: Final = frozenset(_UNITS)


def _plain(text: str) -> str:
    return text.translate(_SAME_CHARACTER)


def _fragments(text: str) -> list[str]:
    return [part for sentence in split_sentences(text) for part in _FRAGMENT.split(sentence)]


def _roles(text: str) -> set[str]:
    lowered = re.sub(r"'s\b", "", _IDENTIFIER.sub(" ", text).lower())
    return {role for form, role in _ROLE_FORMS if re.search(rf"\b{form}\b", lowered)}


def _designations(text: str) -> set[str]:
    return {f"{kind.lower()}:{name.lower()}" for kind, name in _DESIGNATION.findall(text)}


def _tags(text: str) -> set[str]:
    return {
        f"{(hyphened or capital).lower()}-{number}"
        for hyphened, capital, number in _TAG.findall(text)
    }


def _excluded_tags(text: str) -> set[str]:
    return {f"{prefix.lower()}-{number}" for prefix, number in _EXCLUDED_TAG.findall(text)}


def _forms(tokens: set[str]) -> set[str]:
    return {form for token in tokens for form in ing_forms(token)}


def _order(clause: str) -> tuple[set[str], set[str]] | None:
    """The steps a clause orders, as the words it says come first and the words it says come
    after ("A before B" and "B after A" are one order; "do not start until X" is "start after X";
    round 3, RMD-DEF-014 and 112); words on both sides tell nothing. None without "before" or
    "after"."""
    text = f" {_EXCLUDED_TAG.sub(' ', clause.lower())} "
    for pattern, word in _READINGS:
        text = pattern.sub(word, text)
    negations = sum(1 for word in _WORDS.findall(text) if word in _NEGATORS)
    if negations % 2:
        text = re.sub(r"\buntil\b", " after ", text)
    found = _ORDER_WORD.search(text)
    if found is None:
        return None
    left = _forms(content_tokens(text[: found.start()]))
    right = _forms(content_tokens(text[found.end() :]))
    first, then = (left, right) if found.group(1) == "before" else (right, left)
    return first - then, then - first


def _marks(text: str) -> set[str]:
    """A text's content tokens with its designations and equipment tags: what it is about."""
    return content_tokens(text) | _designations(text) | {f"tag:{t}" for t in _tags(text)}


def _reading(clause: str) -> tuple[int, set[str]]:
    """A clause's negation parity and its words, its bounds and orders read as such; "not ...
    until" is "after" ("do not start until the checks pass" is "start after they pass")."""
    # A tag named to exclude it ("not for TP-305") negates the scope, not the claim (round 3).
    text = f" {_EXCLUDED_TAG.sub(' ', clause.lower())} "
    for pattern, word in _READINGS:
        text = pattern.sub(word, text)
    words = set(_WORDS.findall(text))
    negations = sum(1 for word in _WORDS.findall(text) if word in _NEGATORS)
    if "until" in words and negations % 2:
        negations -= 1
        words = (words - {"until"}) | {"after"}
    return negations % 2, words


def _decimal_strict(number: str) -> Decimal | None:
    """A number as a value, its thousands separators read as such (RMD-DEF-057). A number with two
    or more separators ("5.2.1", "28.10.2026", "3,4,5") is a section number, a date or a list, not
    a value (RMD-DEF-109)."""
    if _THOUSANDS.match(number):
        return Decimal(number.replace(",", ""))
    if _PLAIN_NUMBER.match(number):
        return Decimal(number.replace(",", "."))
    return None


def _value_units(text: str) -> list[tuple[Decimal, str | None, frozenset[str]]]:
    """The values ``text`` states, each with its sign (RMD-DEF-115), the unit family after it (None
    without one) and its label: the content tokens of up to four words on each side, document and
    equipment numbers aside (RMD-DEF-113). A number inside a word that starts with a letter is an
    identifier, and the word after "revision" or "rev" names a revision, as ``claim_values`` reads
    them."""
    words = _MARKER.sub(" ", _plain(text)).split()
    spots: list[tuple[int, int, list[Decimal], str | None]] = []
    previous = ""
    signed = tolerance = False
    skip = -1
    for i, raw in enumerate(words):
        if i == skip:
            continue
        lowered = raw.casefold()
        # Round 3: a sign or a tolerance written as a word of its own governs the next number.
        if lowered in _SIGN_WORDS:
            signed = True
            continue
        if lowered in _TOLERANCE:
            tolerance = True
            continue
        word = _LEADING.sub("", raw)
        if word[:1].isascii() and word[:1].isdigit() and previous not in _REVISION_WORDS:
            numbers = _NUMBER_STRICT.findall(word)
            # "100 000": thousands written with a space (round 3, RMD-DEF-057).
            if (
                re.fullmatch(r"[0-9]{1,3}", word)
                and i + 1 < len(words)
                and re.fullmatch(r"[0-9]{3}[.,;:]?", words[i + 1])
            ):
                numbers = [word + words[i + 1].rstrip(".,;:")]
                skip = i + 1
            values = [v for v in (_decimal_strict(n) for n in numbers) if v is not None]
            if values and (signed or _SIGNED.match(raw)):
                values[0] = -values[0]
            is_tolerance = tolerance or raw[:1] in _TOLERANCE
            signed = tolerance = False
            rest = word[word.rfind(numbers[-1]) + len(numbers[-1]) :] if numbers else ""
            # A compound unit is read by its first part ("mg/swab" is mg; round 3).
            after = _NOT_UNIT.sub("", rest.casefold().split("/")[0])
            last = skip if skip > i else i
            if not after and last + 1 < len(words):
                after = _NOT_UNIT.sub("", words[last + 1].casefold().split("/")[0])
                last = last + 1 if after in _UNITS else last
            if values and not is_tolerance:
                spots.append((i, last, values, _UNITS.get(after)))
        else:
            signed = tolerance = False
        previous = word.rstrip(".,:;").casefold()
    found: list[tuple[Decimal, str | None, frozenset[str]]] = []
    for n, (i, last, values, unit) in enumerate(spots):
        start = spots[n - 1][1] + 1 if n else 0
        end = spots[n + 1][0] if n + 1 < len(spots) else len(words)
        around = words[max(start, i - _LABEL_WORDS) : i] + words[last + 1 : min(end, last + 5)]
        label = frozenset(content_tokens(_IDENTIFIER.sub(" ", " ".join(around))))
        found += [(value, unit, label) for value in values]
    return found


def _value_clauses(sentence: str) -> list[str]:
    """A sentence's clauses for values: split at semicolons, and at commas and "and" where every
    part states a value of its own ("Set the pre-compression force to 5 kN and the main compression
    force to 40 kN"; round 3, RMD-DEF-027)."""
    clauses: list[str] = []
    for part in sentence.split(";"):
        pieces = [piece for piece in re.split(r",\s+|\s+and\s+", part) if piece.strip()]
        if len(pieces) > 1 and all(_value_units(piece) for piece in pieces):
            clauses += pieces
        else:
            clauses.append(part)
    return clauses


def _passage_clauses(text: str) -> list[tuple[set[str], list[tuple[Decimal, str | None]]]]:
    """Each clause of a passage (``_value_clauses``) as what it is about (``_marks``) and the
    values it states with their units."""
    clauses = [part for s in split_sentences(_plain(text)) for part in _value_clauses(s)]
    return [
        (_marks(clause), [(v, u) for v, u, _ in _value_units(clause)])
        for clause in clauses
        if clause.strip()
    ]


def _dates(text: str) -> set[str]:
    found: set[str] = set()
    for match in _DATE.finditer(_plain(text)):
        day, month, year = match.groups()
        found.add(f"{int(day)}.{int(month)}.{year}" if day else match.group(0))
    return found


def _values_held(claimed: str, passage: str, *, pooled: bool = False) -> bool:
    """Whether every value of the claim is one the passage states, with its sign and, when both
    give one, its unit family, in a clause the claim matches no less well than any other clause
    stating a value of that family: by its label words, then by the whole claim (its tags and
    designations included). A value moved to another parameter, equipment or site is not held
    (RMD-DEF-027, RMD-DEF-114); a heading or a clause stating no such value is no rival
    (RMD-DEF-113). Round 3: clauses are matched first by the tags and designations they share with
    the claim, then by the whole claim, then by the label; a date the claim states must be one the
    passage states; and read across pooled passages (``pooled``) a value must be held in a clause
    sharing two words with the claim beyond its unit (RMD-DEF-027, 109)."""
    if _dates(claimed) - _dates(passage):
        return False
    clauses = _passage_clauses(passage)
    whole = _marks(_plain(claimed))
    named = {mark for mark in whole if ":" in mark}
    for value, unit, label in _value_units(claimed):
        holding = [
            i
            for i, (_, pairs) in enumerate(clauses)
            if any(v == value and (unit is None or u is None or u == unit) for v, u in pairs)
        ]
        if not holding:
            return False
        # A value with no unit (a step number, a count) has no parameter to be moved from.
        rivals = [
            i
            for i, (_, pairs) in enumerate(clauses)
            if unit is not None and i not in holding and any(u == unit for _, u in pairs)
        ]

        def score(i: int, label: frozenset[str] = label) -> tuple[int, int, int]:
            tokens = clauses[i][0]
            return len(named & tokens), len(whole & tokens), len(label & tokens)

        if rivals and max(map(score, rivals)) > max(map(score, holding)):
            return False
        if pooled and max(len((whole & clauses[i][0]) - _UNIT_WORDS) for i in holding) < 2:
            return False
    return True


def _inverted(claimed: str, passage: str) -> bool:
    """Whether the claim says the opposite of, or more than, the passage: a clause of the claim
    whose negation, order of steps, bound or modal (may against must) differs from the passage
    clause it matches best (RMD-DEF-014, 111, 112); a role the passage sentence it matches (or, if
    that names none, the passage) does not name; a site or grade other than the ones it names; or
    an equipment tag whose prefix the passage names with other numbers only (RMD-DEF-114)."""
    claimed, passage = _plain(claimed), _plain(passage)
    clauses = [c for c in _fragments(passage) if c.strip()]
    claim_parts = [c for c in _fragments(claimed) if c.strip()]
    for part in claim_parts:
        tokens = _marks(part)
        best = max(clauses, key=lambda c: len(tokens & _marks(c)), default="")
        # A clause is compared with the passage clause it restates: half its words, at least two.
        words = content_tokens(part)
        if len(words & content_tokens(best)) < max(2, -(-len(words) // 2)):
            continue
        claim_parity, claim_words = _reading(part)
        passage_parity, passage_words = _reading(best)
        if claim_parity != passage_parity:
            return True
        mine, theirs = _order(part), _order(best)
        if mine and theirs and (mine[0] & theirs[1] or mine[1] & theirs[0]):
            return True
        for one, other in (*_OPPOSITES, *((b, a) for a, b in _OPPOSITES)):
            if one in claim_words and other in passage_words and one not in passage_words:
                return True
        if not claim_parity:
            for own, theirs_words in ((_PERMISSION, _OBLIGATION), (_OBLIGATION, _PERMISSION)):
                if claim_words & own and passage_words & theirs_words and not passage_words & own:
                    return True
        elif _no_obligation_against_prohibition(claim_words, passage_words):
            return True
    if _contained_clause_inverted(claim_parts, clauses):
        return True
    return _named_differently(claimed, passage)


def _no_obligation_against_prohibition(claim_words: set[str], passage_words: set[str]) -> bool:
    """Negated, "not required" (no obligation) and "not permitted" or "must not" (a prohibition)
    are opposites (round 3, RMD-DEF-014)."""
    claim_free = bool(claim_words & _NO_OBLIGATION)
    passage_free = bool(passage_words & _NO_OBLIGATION)
    return (claim_free and not passage_free and bool(passage_words & _FORBIDS)) or (
        passage_free and not claim_free and bool(claim_words & _FORBIDS)
    )


def _contained_clause_inverted(claim_parts: Sequence[str], clauses: Sequence[str]) -> bool:
    """Whether a passage clause the claim states word for word (two words or more) has the other
    negation, with no such clause of the claim's negation beside it ("Do not restart the line.
    Cleaning must be verified ..." against "Restart the line once cleaning is verified ..."; round
    3, RMD-DEF-014)."""
    held = [(part, _forms(content_tokens(part))) for part in claim_parts]
    against: list[set[str]] = []
    agreeing: list[set[str]] = []
    for clause in clauses:
        own = content_tokens(clause)
        if len(own) < 2:
            continue
        holder = next((part for part, tokens in held if own <= tokens), None)
        if holder is None:
            continue
        (agreeing if _reading(holder)[0] == _reading(clause)[0] else against).append(own)
    # A clause of the claim's negation over the same words ("Do not restart the line ... Restart
    # the line once ...") answers the one against it.
    return any(not any(len(own & other) >= 2 for other in agreeing) for own in against)


def _named_differently(claimed: str, passage: str) -> bool:
    """Whether the claim names a role, a site or a grade other than the passage part it matches
    names (each part a sentence's piece joined by "and" or a comma; the sentence, then the passage,
    when that part names none of the kind), or an equipment tag whose prefix the passage names with
    other numbers only (RMD-DEF-027, 114; round 3)."""
    pieces = [
        (piece, sentence)
        for sentence in split_sentences(passage)
        if sentence.strip()
        for piece in _ROLE_PART.split(sentence)
        if piece.strip()
    ]
    for part in _ROLE_PART.split(claimed):
        roles, designations = _roles(part), _designations(part)
        if not (roles or designations) or not pieces:
            continue
        tokens = _marks(part)
        piece, sentence = max(pieces, key=lambda found: len(tokens & _marks(found[0])))
        if roles - (_roles(piece) or _roles(sentence) or _roles(passage)):
            return True
        for designation in designations:
            kind = f"{designation.split(':')[0]}:"
            stated = next(
                (
                    found
                    for found in (
                        {d for d in _designations(text) if d.startswith(kind)}
                        for text in (piece, sentence, passage)
                    )
                    if found
                ),
                set[str](),
            )
            if stated and designation not in stated:
                return True
    stated_tags = _tags(passage)
    for tag in _tags(claimed) - _excluded_tags(claimed):
        same = {t for t in stated_tags if t.split("-")[0] == tag.split("-")[0]}
        if same and tag not in same:
            return True
    return False


def _revision_name(written: str) -> str:
    # Compared as text, leading zeros aside: a number of any length never raises (round 3).
    return (written.lstrip("0") or "0") if written.isdigit() else written.upper()


_COMPARABLE_REVISION: Final = re.compile(r"[0-9]+[A-Za-z]?|[A-Za-z][0-9]?")


def _other_revision(claimed: str, label: PassageLabel) -> bool:
    """Whether the claim names a revision other than the cited passage's (RMD-DEF-028): numbers
    compared as numbers ("02" is "2"), letters as letters; a revision named right after a document
    number is that document's (round 3), so another document's revision is not compared with this
    passage's; a passage whose revision is not one of these shapes holds no revision a claim
    names."""
    own = label.revision.strip()
    comparable = _COMPARABLE_REVISION.fullmatch(own) is not None
    tied = [
        revision
        for document, revision in _DOCUMENT_REVISION.findall(claimed)
        if document.casefold() == label.document_number.strip().casefold()
    ]
    if tied:
        return not comparable or _revision_name(own) not in {_revision_name(r) for r in tied}
    untied = _DOCUMENT_REVISION.sub(" ", claimed)
    named = {_revision_name(a or b) for a, b in _CLAIMED_REVISION.findall(untied)}
    if not named:
        return False
    return not comparable or _revision_name(own) not in named


def _substantive(
    sentence: str, trusted: frozenset[str], *, clauses: bool = False, strict: bool = False
) -> bool:
    bare = _MARKER.sub(" ", sentence).strip()
    if not bare or bare in trusted or bare.endswith("?"):
        return False
    if clauses:
        return bool(content_tokens(_asserted(bare, strict)))
    if _MISSING.search(bare):
        return False
    return bool(content_tokens(bare))


def _marked_parts(sentence: str) -> list[tuple[str, list[str]]]:
    """The parts of ``sentence`` that end at a citation marker and state a value, each with the
    keys its marker names ("72 hours in one document [E4]", "and 60 hours in another [E6]")."""
    parts = re.split(r"(\[[^\]]+\])", _plain(sentence))
    marked = [
        (text, [key.strip() for key in marker.strip("[]").split(",")])
        for text, marker in zip(parts[::2], parts[1::2], strict=False)
        if _value_units(text)
    ]
    # A value after the last marker rests on no marker of its own: any cited passage must hold it
    # (round 3).
    if len(parts) % 2 and _value_units(parts[-1]):
        marked.append((parts[-1], ["*"]))
    return marked


def _pooled(
    sentence: str,
    claimed: str,
    claim: set[str],
    cited: Sequence[str],
    passages: Mapping[str, str],
    labels: Mapping[str, PassageLabel],
    evidence: Callable[[str], tuple[set[str], frozenset[Decimal]] | None],
    held_share: Callable[[set[str], set[str]], float],
) -> bool | None:
    """A claim citing two or more passages, read against them together (``strict``): a comparison
    or a conflict takes a value from each ("72 hours in one effective document [E4] and 60 hours in
    another [E6]"), which no one passage holds. None when its words are not held by the passages
    together or it inverts them. Else, when the claim puts its values beside two or more markers,
    whether each value is held by a passage of the marker that follows it (``_marked_parts``: a
    value given to the wrong document is not); otherwise whether each value is held in the clause
    it matches best across the passages (``_values_held``)."""
    usable = [
        key
        for key in cited
        if evidence(key) is not None
        and not (key in labels and _other_revision(claimed, labels[key]))
    ]
    if len(usable) < 2:
        return None
    read = [found for key in usable if (found := evidence(key)) is not None]
    tokens = set[str]().union(*(found[0] for found in read))
    named = set[str]().union(*(labels[key].words() for key in usable if key in labels))
    rest = claim - (named - tokens)
    if rest and held_share(rest, tokens) < SUPPORT_THRESHOLD:
        return None
    joined = "\n".join(passages[key] for key in usable)
    if _inverted(claimed, joined):
        return None
    marked = _marked_parts(sentence)
    if len(marked) >= 2:
        return all(
            any(
                _values_held(text, passages[key])
                for key in (usable if keys == ["*"] else keys)
                if key in usable
            )
            for text, keys in marked
        )
    return _values_held(claimed, joined, pooled=True)


def _misnamed(sentence: str, cited: Sequence[str], labels: Mapping[str, PassageLabel]) -> bool:
    """Whether the claim names the document of a passage it does not cite, and of none it does."""
    named = {key for key, label in labels.items() if label.names_document(sentence)}
    return bool(named) and not named & set(cited)


def check_grounding(
    sources: Sequence[ClaimSource],
    passages: Mapping[str, str],
    *,
    trusted: frozenset[str] = frozenset(),
    labels: Mapping[str, PassageLabel] | None = None,
    inflections: bool = False,
    clauses: bool = False,
    strict: bool = False,
) -> GroundingResult:
    """Each claim of ``sources`` with its citations and its support, against ``passages`` (the
    context's text by citation key).

    ``labels`` holds the passages' register labels by key (DV-RMD-14): a claim's words that name
    the passage it is checked against (its document number, revision and status) and that the
    passage's text does not hold count neither for nor against it, so a claim left with no other
    word states only that passage's register entry; a claim that names the document of a passage
    it does not cite, and of none it does, is not entailed. With ``inflections`` (DV-RMD-20), a
    claim token is held when the passage holds one of its ``ing_forms``; values are still compared
    exactly. With ``clauses`` (DV-RMD-24, RMD-DEF-006), a sentence's clauses that state missing
    information are set aside and the rest is the claim, instead of the whole sentence being no
    claim. With ``strict`` (RMD13-T02, the application's path), a passage holds a claim only if the
    claim does not invert the passage sentence it matches (``_inverted``: negation, order of steps,
    a role or a site), names no other revision than the passage's (``_other_revision``), and states
    each value with its unit in the clause its label words match (``_values_held``, thousands
    separators read as such); the clause split also takes concessive openers and ", so", ", and",
    "however" and colon joins. Without labels, inflections, clauses or strict, the rule is the
    previous track's (``evals.pharma.grading_v2``)."""
    labels = labels or {}
    read: dict[str, tuple[set[str], frozenset[Decimal]] | None] = {}

    def evidence(key: str) -> tuple[set[str], frozenset[Decimal]] | None:
        """A passage's tokens and values, read once; None for an unknown key or a passage
        addressed to the model (PLV-D-025)."""
        if key not in passages:
            return None
        if key not in read:
            text = passages[key]
            instruction = suspected_instruction(text) is not None
            read[key] = None if instruction else (content_tokens(text), passage_values(text))
        return read[key]

    def held_share(rest: set[str], tokens: set[str]) -> float:
        held = [t for t in rest if t in tokens or (inflections and ing_forms(t) & tokens)]
        return len(held) / len(rest)

    checks: list[ClaimCheck] = []
    for source in sources:
        for raw in split_sentences(source.text):
            sentence = raw.strip()
            if not sentence:
                continue
            cited = _cited(sentence, source.citation_keys)
            if not _substantive(sentence, trusted, clauses=clauses, strict=strict):
                checks.append(
                    ClaimCheck(source.field, sentence, cited, False, False, NOT_SUBSTANTIVE)
                )
                continue
            if not cited:
                checks.append(ClaimCheck(source.field, sentence, cited, True, False, NO_CITATION))
                continue
            bare = _MARKER.sub(" ", sentence)
            claimed = _asserted(bare, strict) if clauses else bare
            claim = content_tokens(claimed) - _PRESENTATION if strict else content_tokens(claimed)
            values = claim_values(claimed)
            tokens_held = held = False
            misnamed = _misnamed(sentence, cited, labels)
            for key in cited:
                found = evidence(key)
                if found is None or misnamed:
                    continue
                label = labels.get(key)
                if strict and label is not None and _other_revision(claimed, label):
                    continue
                # A label word the passage's own text holds counts as held, like any other word.
                rest = claim - (label.words() - found[0]) if label is not None else claim
                if rest and held_share(rest, found[0]) < SUPPORT_THRESHOLD:
                    continue
                if strict and _inverted(claimed, passages[key]):
                    continue
                tokens_held = True
                if _values_held(claimed, passages[key]) if strict else values <= found[1]:
                    held = True
                    break
            if strict and not held and not misnamed:
                pooled = _pooled(
                    sentence, claimed, claim, cited, passages, labels, evidence, held_share
                )
                tokens_held = tokens_held or pooled is not None
                held = bool(pooled)
            reason = SUPPORTED if held else VALUE_NOT_FOUND if tokens_held else NOT_ENTAILED
            checks.append(ClaimCheck(source.field, sentence, cited, True, held, reason))
    instructions = sum(1 for found in read.values() if found is None)
    return GroundingResult(tuple(checks), instructions)
