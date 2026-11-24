"""What an answer may not repeat from its prompt (RMD8-T02; RE2E-DEF-043, RE2E-DEF-044).

Asked to copy out its context, a live answer reproduced the whole evidence block: the fences, the
``chunk_id=``/``source=`` provenance lines and the passage text (TC-SEC-INJ-U-FENCES). Others
quoted a document's planted instruction, canary included, in a note saying it was ignored. The
prompt rules ask the model not to; this filter makes sure, after the model and before storage,
and the cleaned answer says what was left out (DV-RMD-56).

Three removals, all deterministic:

- a line whose text, after list, quote and code markers, is a fence delimiter or begins with a
  provenance prefix the application writes into the evidence block;
- a sentence sharing eight or more consecutive words with a passage sentence that
  ``suspected_instruction`` reads as an instruction. Eight words is longer than the instruction
  phrases themselves, so a sentence that only names the attack is kept, and a quotation of a
  passage that holds no instruction is never touched;
- in a General answer, a sentence that names the evidence block's tokens inline or copies eight
  words of the General rules: declining to copy its context, a live answer restated the rules
  instead (RMD-DEF-008, DV-RMD-60).
"""

import re
from collections.abc import Callable, Iterable, Sequence

from app.agents.prompts import SAFETY_RULES
from app.pharma.signals import suspected_instruction

CONTEXT_NOTICE = (
    "Lines of the evidence context that the answer repeated were left out of this answer."
)
INSTRUCTION_NOTICE = (
    "Quoted instructions from a document were left out of this answer; instructions found in "
    "documents are never followed."
)
PROMPT_NOTICE = "Text restating the application's own instructions was left out of this answer."
# Appended to a shorter text (an uncertainty statement, a next step) that lost context lines.
SHORT_NOTICE = "(Evidence context lines were left out.)"

COPY_WORDS = 8

_LEADING_MARKERS = re.compile(r"^\s*(?:(?:[-*+]|\d{1,3}[.)])\s+|>\s?|`+\s*)*")
_FENCE_DELIMITER = re.compile(r"^<<<(?:DATA|END):[^>\n]*>>>`*\s*$")
_PROVENANCE_PREFIXES = ("chunk_id=", "label chunk_id=", "source=", "text: ")
_CODE_FENCE = re.compile(r"^\s{0,3}(?:`{3,}|~{3,})")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
# What may close a sentence after its full stop: a straight or typographic quotation mark, a
# parenthesis or a bracket. A line's last words are a sentence even without a full stop, as a
# bullet often is.
_CLOSERS = re.escape("\"')]" + chr(0x201D) + chr(0x2019))
_SENTENCE = re.compile(rf"[^.!?\n]+(?:[.!?]+[{_CLOSERS}]*|$)", re.MULTILINE)
_WORD = re.compile(r"[A-Za-z0-9]+")
_BRACKETED = re.compile(r"\[[^\[\]\n]*\]")


def _is_context_line(line: str) -> bool:
    text = _LEADING_MARKERS.sub("", line, count=1)
    return bool(_FENCE_DELIMITER.match(text)) or text.startswith(_PROVENANCE_PREFIXES)


def _without_empty_code_blocks(lines: list[str]) -> list[str]:
    """Drops a fenced code block left holding nothing but blank lines."""
    kept: list[str] = []
    opened: int | None = None
    for line in lines:
        if not _CODE_FENCE.match(line):
            kept.append(line)
            continue
        if opened is None:
            opened = len(kept)
            kept.append(line)
            continue
        if all(not item.strip() for item in kept[opened + 1 :]):
            del kept[opened:]
        else:
            kept.append(line)
        opened = None
    return kept


def _tidy(text: str) -> str:
    return _EXTRA_BLANK_LINES.sub("\n\n", text).strip()


def strip_context_lines(text: str) -> tuple[str, int]:
    """``text`` without its fence and provenance lines, and how many there were. A text holding
    none is returned exactly as it is."""
    lines = text.split("\n")
    kept = [line for line in lines if not _is_context_line(line)]
    removed = len(lines) - len(kept)
    if not removed:
        return text, 0
    return _tidy("\n".join(_without_empty_code_blocks(kept))), removed


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


def _shingles(words: Sequence[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + COPY_WORDS]) for i in range(len(words) - COPY_WORDS + 1)}


def _planted_shingles(passages: Iterable[str]) -> set[tuple[str, ...]]:
    shingles: set[tuple[str, ...]] = set()
    for passage in dict.fromkeys(passages):
        if suspected_instruction(passage) is None:
            continue
        for sentence in _SENTENCE.findall(passage):
            if suspected_instruction(sentence) is not None:
                shingles |= _shingles(_words(sentence))
    return shingles


def drop_planted_copies(text: str, passages: Iterable[str]) -> tuple[str, int]:
    """``text`` without the sentences that copy an instruction planted in a passage, and how many
    there were. A text holding none is returned exactly as it is."""
    planted = _planted_shingles(passages)
    if not planted:
        return text, 0
    return _drop_sentences(
        text, lambda sentence: not _shingles(_words(sentence)).isdisjoint(planted)
    )


# RMD-DEF-008 (DV-RMD-60): the General rules as one word sequence, so a copied rule counts even
# where the rules break it across lines.
_RULE_SHINGLES = _shingles(_words(SAFETY_RULES))
_FORMAT_TOKEN = re.compile(r"<<<(?:DATA|END):|chunk_id=", re.IGNORECASE)


def restates_prompt(sentence: str) -> bool:
    """Whether ``sentence`` names the evidence block's own tokens inline (a fence delimiter or the
    provenance prefix) or copies eight or more consecutive words of the General rules."""
    return bool(_FORMAT_TOKEN.search(sentence)) or not _shingles(_words(sentence)).isdisjoint(
        _RULE_SHINGLES
    )


def drop_prompt_copies(text: str) -> tuple[str, int]:
    """``text`` without the sentences that restate the application's own prompt, and how many
    there were (RMD-DEF-008). A text holding none is returned exactly as it is."""
    return _drop_sentences(text, restates_prompt)


def _drop_sentences(text: str, drop: Callable[[str], bool]) -> tuple[str, int]:
    parts: list[str] = []
    position = 0
    removed = 0
    opened_line = False
    for match in _SENTENCE.finditer(text):
        if not drop(match.group(0)):
            continue
        start = match.start()
        # A sentence opens its line when it starts one, or follows a removed one that did.
        opened_line = start == 0 or text[start - 1] == "\n" or (opened_line and position >= start)
        parts.append(text[position:start])
        position = match.end()
        if opened_line:
            # The next sentence then does without the space before it.
            while position < len(text) and text[position] in " \t":
                position += 1
        removed += 1
    if not removed:
        return text, 0
    parts.append(text[position:])
    return _tidy("".join(parts)), removed


def notice_for(lines: int, copies: int) -> str:
    """The paragraph saying what a cleaned answer left out, or "" when it left nothing out."""
    return " ".join(
        notice for count, notice in ((lines, CONTEXT_NOTICE), (copies, INSTRUCTION_NOTICE)) if count
    )


def clean_answer(text: str, passages: Iterable[str]) -> tuple[str, int, int]:
    """``text`` without context lines, planted copies and restatements of the General rules, with
    a notice when it lost any; and two counts, the lines and the copied instructions (a document's
    and the application's own together, RMD-DEF-008). A text that loses nothing is returned
    exactly as it is."""
    stripped, lines = strip_context_lines(text)
    without_planted, planted = drop_planted_copies(stripped, passages)
    cleaned, restated = drop_prompt_copies(without_planted)
    notice = " ".join(
        part for part in (notice_for(lines, planted), PROMPT_NOTICE if restated else "") if part
    )
    if not notice:
        return text, 0, 0
    return (f"{cleaned}\n\n{notice}" if cleaned else notice), lines, planted + restated


def with_short_notice(text: str) -> tuple[str, int]:
    """A shorter text without its context lines, saying so when it lost any; and the count."""
    stripped, lines = strip_context_lines(text)
    if not lines:
        return text, 0
    return (f"{stripped} {SHORT_NOTICE}" if stripped else SHORT_NOTICE), lines


def removed_instruction_sentences(drafted: str, shown: str) -> int:
    """How many instruction-like sentences of ``drafted`` (context lines already left out), a
    document's or a restatement of the General rules, the ``shown`` text no longer holds. A
    citation marker is not part of the comparison, since the shown text may have lost or renamed
    it."""
    shown_words = f" {' '.join(_words(_BRACKETED.sub(' ', shown)))} "
    count = 0
    for sentence in _SENTENCE.findall(drafted):
        if suspected_instruction(sentence) is None and not restates_prompt(sentence):
            continue
        words = _words(_BRACKETED.sub(" ", sentence))
        if words and f" {' '.join(words)} " not in shown_words:
            count += 1
    return count
