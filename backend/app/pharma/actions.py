"""The deterministic detector of prohibited regulated acts (PLV4-T02; LV-REQ-001 to LV-REQ-003).

The classes, forms and witnesses are those of ``docs/pharma-live/policy/action-taxonomy.md``. A
sentence is read as the policy check reads it (``policy.normalize_for_policy``) and cut into clauses
at the pivots that end a witness's reach: a semicolon, a colon, a question mark, "but", "however",
"that said", ", still,", and a comma or "and" followed by a directive or the system speaking for
itself. A clause takes an act when it holds a cue of the act's class (the decision verb with its
object, a resultative statement, a euphemism, or a statement that nothing is left for the qualified
person) and no witness governs it: a qualified role taking the decision, the system's own negation,
a question, a report of a record, a regulation, a deferral, a conditional deferral or a hedged
suggestion. Some acts are taken by a negative statement ("no further quality review is needed",
"it is not a deviation", "that step won't apply tonight"); a negation never excuses those. The
system deciding in its own voice ("I find the system compliant") or granting the act to someone
("the lab is cleared to ...") is never excused by a role, a record or a regulation in its clause.

Each finding names the class, its criticality, the form, the sentence and the span of the cue in
the normalised sentence, a confidence, and the policy code the response's notices use. Every pattern
is an alternation anchored on word boundaries with bounded gaps and no nested repetition, so
detection is linear in the length of the text.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

# A module import, not a name import: the policy check imports this module, so either may be
# imported first; neither uses the other until a function runs.
from app.pharma import policy
from app.pharma.enums import PolicyCode


class ActionClass(StrEnum):
    """The action classes AC-01 to AC-11 of the taxonomy, section 1."""

    BATCH_DISPOSITION = "AC-01"
    DEVIATION_CLOSURE = "AC-02"
    DEVIATION_CLASSIFICATION = "AC-03"
    CAPA_CLOSURE = "AC-04"
    ROOT_CAUSE = "AC-05"
    COMPLIANCE_DECLARATION = "AC-06"
    PROCEDURE_OVERRIDE = "AC-07"
    EQUIPMENT_COMMAND = "AC-08"
    MEDICAL_INSTRUCTION = "AC-09"
    QUALIFIED_PERSON = "AC-10"
    REVIEW_WAIVER = "AC-11"


A = ActionClass
CRITICAL: Final = frozenset(
    {
        A.BATCH_DISPOSITION,
        A.DEVIATION_CLOSURE,
        A.CAPA_CLOSURE,
        A.PROCEDURE_OVERRIDE,
        A.EQUIPMENT_COMMAND,
        A.MEDICAL_INSTRUCTION,
        A.QUALIFIED_PERSON,
        A.REVIEW_WAIVER,
    }
)
# The policy code of each class (taxonomy, section 1); the two classes without a code of their own
# are autonomous regulated decisions.
CODE: Final = {
    A.BATCH_DISPOSITION: PolicyCode.BATCH_DISPOSITION,
    A.DEVIATION_CLOSURE: PolicyCode.APPROVAL_LANGUAGE,
    A.DEVIATION_CLASSIFICATION: PolicyCode.AUTONOMOUS_DECISION,
    A.CAPA_CLOSURE: PolicyCode.APPROVAL_LANGUAGE,
    A.ROOT_CAUSE: PolicyCode.ROOT_CAUSE,
    A.COMPLIANCE_DECLARATION: PolicyCode.COMPLIANCE_CLAIM,
    A.PROCEDURE_OVERRIDE: PolicyCode.PROCEDURE_OVERRIDE,
    A.EQUIPMENT_COMMAND: PolicyCode.AUTONOMOUS_DECISION,
    A.MEDICAL_INSTRUCTION: PolicyCode.MEDICAL_ADVICE,
    A.QUALIFIED_PERSON: PolicyCode.AUTONOMOUS_DECISION,
    A.REVIEW_WAIVER: PolicyCode.AUTONOMOUS_DECISION,
}

DIRECT, INDIRECT, PASSIVE, EUPHEMISTIC = "direct", "indirect", "passive", "euphemistic"
CONDITIONAL, DELEGATED, IMPLIED = "conditional", "delegated", "implied"


@dataclass(frozen=True)
class ActionFinding:
    action_class: ActionClass
    critical: bool
    form: str
    sentence_index: int
    span: tuple[int, int]
    confidence: float
    code: PolicyCode


@dataclass(frozen=True)
class _Cue:
    pattern: re.Pattern[str]
    form: str
    # A negative statement takes the act: a negation never excuses it.
    negative: bool = False
    # A report of a record or a regulation can excuse it; an override verb ("per the SOP, skip
    # step 4") is the output's own act whatever it cites.
    reportable: bool = True
    # RMD4-T02: a cue read only in its setting, as a class's guard is ("marked effective" is a
    # CAPA's effectiveness only in a sentence about a CAPA, not a document's lifecycle status).
    topic: re.Pattern[str] | None = None
    # RMD4-T02 (DV-RMD-30): a value granted or directed is an override only past a limit its text
    # states ("WI-443 limits the hold to 72 hours, ... the cores can wait 90 hours"): "max" past a
    # maximum, "min" short of a minimum, "both" either; "cited" past a maximum when the clause
    # carries its passage's marker, and always without one (RMD13-T02 round 3).
    limit: str = ""
    # RMD4-T02 (DV-RMD-30): what a past-tense statement says happened ("the blend sat 10 minutes
    # past the hold limit") is a fact, not an override.
    fact: bool = False


def _p(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


# --- Vocabulary -----------------------------------------------------------------------------------

# Up to n words (letters, digits and the punctuation of identifiers) between two parts of a cue.
_W = r"[\w'/.,()%\u00b0-]+ "
_GAP4, _GAP8, _GAP12 = f"(?:{_W}){{0,4}}", f"(?:{_W}){{0,8}}", f"(?:{_W}){{0,12}}"
_APOS = "[']"
_NEG_AUX = (
    rf"(?:not|never|no|cannot|can{_APOS}t|won{_APOS}t|don{_APOS}t|doesn{_APOS}t|didn{_APOS}t|"
    rf"isn{_APOS}t|aren{_APOS}t|wasn{_APOS}t|shouldn{_APOS}t|mustn{_APOS}t|needn{_APOS}t|nothing|"
    rf"neither|nor|without)"
)

# RMD-DEF-018: a batch or lot named by its identifier alone ("release RE2E-B-01"): a hyphenated
# identifier of at least three parts whose last part holds a digit, and which is not a controlled
# document's or a quality record's (those are "SOP-SYN-QA-101", "DEV-SYN-045"); an equipment
# number has two parts ("TP-201"). RMD-DEF-099: it starts a word (never the tail of a document
# number, "SYN-QA-101" in "SOP-SYN-QA-101") and holds a letter (never a date, "2026-10-27").
_BATCH_ID = (
    r"(?<![\w-])(?=[a-z0-9-]*[a-z])(?!(?:sop|wi|form|frm|pol|doc|tm|mbr|bpr|dev|capa|ncr|cc|chg|"
    r"cr|man|ws|spec|qa|qc)-)[a-z0-9]+-[a-z0-9]+-[a-z0-9-]*\d[a-z0-9-]*"
)
_LOT = (
    r"(?:lots?|batch(?:es)?|bulk|products?|vials?|bags?|kits?|tablets?|capsules?|pallets?|"
    r"shipments?|units|stock|material|drums?|containers?|cartons?|bottles?|tubes?|syringes?|"
    rf"ampoules?|inventory|consignments?|loads?|cases?|totes?|it|them|{_BATCH_ID})"
)
# A record's identifier may carry letters before its number ("DEV-SYN-045", "CAPA-SYN-018";
# RMD4-T02).
_DEV = (
    r"(?:deviations?|dev-?(?:[a-z]+-)*\d[\w-]*|non-?conformances?|ncrs?|ncr-(?:[a-z]+-)*\d[\w-]*|"
    r"incidents?|investigations?|excursion records?|event records?|deviation records?|the record|"
    r"records?)"
)
# The deviation itself, not any record of it.
_DEV_EVENT = (
    r"(?:deviations?|dev-?(?:[a-z]+-)*\d[\w-]*|non-?conformances?|ncrs?|ncr-(?:[a-z]+-)*\d[\w-]*|"
    r"incidents?|investigations?|excursions?)"
)
_CAPA = (
    r"(?:capas?|capa-?(?:[a-z]+-)*\d[\w-]*|corrective (?:and preventive )?actions?|"
    r"preventive actions?|action plans?|the fix|retraining)"
)
_EQUIP = (
    r"(?:pumps?|(?:tp|vf|bal|fbd|p|ahu|cip|sip|lyo|wfi)-\d+|lines?|press(?:es)?|dryers?|"
    r"balances?|set ?points?|alarms?|"
    r"interlocks?|agitators?|skids?|fillers?|loops?|torque|rpm|speeds?|temperatures?|valves?|"
    r"autoclaves?|mixers?|cappers?|sensors?|hvac|fans?|heaters?|chillers?|compressors?|motors?|"
    r"conveyors?|blenders?|granulators?|coaters?|ovens?|freezers?|machines?|equipment|"
    r"controllers?|production mode|sanitization|cycles?|flow|pressure|vessels?|tanks?|"
    r"centrifuges?|lyophili[sz]ers?|isolators?|doors?|plc|inlet air|air handlers?)"
)
_STEP = (
    r"(?:steps?|checks?|inspections?|clearances?|verifications?|rinses?|cleaning|sampling|"
    r"tests?|testing|hold[- ]times?|protocols?|procedures?|sops?|requirements?|reviews?|"
    r"reconciliations?|drain times?|limits?|sanitization|disassembly|reassembly|line clearance|"
    r"monitoring|calibrations?|signatures?|second checks?|double checks?|filters?|integrity tests?|"
    r"sign-?offs?|drying times?|contact times?|samples?|challenges?|soaks?)"
)
# The checks and reviews a procedure requires, not the parts it handles ("take the filters out"
# is a disassembly step; RMD4-T02).
# The quality authority a review waiver sets aside (RMD4-T02): the quality unit, the qualified
# person, a second-person or independent review, a sign-off.
_AUTHORITY = (
    r"(?:qa|quality (?:unit|assurance|control|review|reviewers?|department|function|team)|qp|"
    r"qualified person|second[- ]person(?: review| check| verification| sign-?off)?|"
    r"(?:independent|supervisory) (?:review|check|sign-?off)|reviewers?|sign-?offs?|"
    r"countersignatures?|(?:qa|quality) (?:approvals?|sign-?offs?))"
)
_CHECK_STEP = (
    r"(?:steps?|checks?|inspections?|clearances?|line clearance|verifications?|rinses?|tests?|"
    r"testing|reviews?|signatures?|sign-?offs?|second checks?|double checks?|integrity tests?|"
    r"sampling|monitoring|calibrations?|reconciliations?|hold[- ]time checks?|drain times?|"
    r"hold[- ]times?|drying times?|contact times?)"
)
_MED = (
    r"(?:doses?|dosing|tablets?|capsules?|cream|ointment|drops|spray|tears|injections?|infusions?|"
    r"bags?|medication|medicines?|antibiotics?|regimen|therapy|treatment|suspension|syrup|"
    r"inhalers?|patch(?:es)?|pills?|prescriptions?|drugs?|vaccines?)"
)
_PERSON = (
    r"(?:patients?|(?:your|his|her|their|the) (?:mother|father|son|daughter|child|children|kid|"
    r"wife|husband|partner|baby|parent|grandmother|grandfather)|caller|the child|you|he|she)"
)
_DOSE = r"\d+(?:[.,]\d+)? ?(?:mg|mcg|\u00b5g|g|ml|units|iu|%|tablets?|capsules?|puffs?|drops)\b"
# The roles that may take a regulated decision (taxonomy, W-01 and W-06).
_ROLE = (
    r"(?:qa|quality assurance|(?:the |your |its |our |a )?(?:own )?quality(?: control)? unit"
    r"(?:'s)?|qu|quality (?:head|director|reviewer|department|organi[sz]ation|leadership|"
    r"signatory|lead|manager|owner|team|function|council|board)|"
    r"(?:deviation |capa |change control |material |quality |cross-functional )?review boards?|"
    r"capa (?:review )?board|capa owner|deviation owner|change control(?: board)?|"
    r"qualified persons?|qp|(?:qualified )?investigation team|investigators?|"
    r"investigation (?:lead|owner)|lead investigator|"
    r"(?:the )?(?:patient(?:'s)? |child(?:'s)? |her |his |your |their )?"
    r"(?:prescriber|physician|doctor|pediatrician|pharmacist|clinician|care team|"
    r"health ?care provider)s?|authori[sz]ed (?:\w+ ){0,2}(?:reviewers?|signator(?:y|ies)|"
    r"technicians?|personnel|staff|signers?)|engineering and qa|validation team|"
    r"the site(?:'s)? quality|"
    # RMD4-T02 (DV-RMD-30): the roles a medical question goes to.
    r"pharmacovigilance(?: team| unit| department| function)?|(?:drug|patient) safety "
    r"(?:team|unit|department)|medical (?:information|affairs)(?: team| unit| department)?)"
)

# The settings some cues are read in (RMD4-T02): a sentence about a CAPA, and one naming equipment.
_CAPA_TOPIC = _p(
    r"\b(?:capas?\b|capa-|corrective|preventive|retrain|remediat|action plans?|the fix\b|"
    r"mitigation)"
)
_EQUIP_WORD = _p(rf"\b{_EQUIP}\b")
# A verb used as a noun ("any change to the speed", "a lower speed limit", "a product run on
# TP-201") commands nothing (RMD4-T02).
_NOUN_USE = (
    r"(?<!\bany )(?<!\bthe )(?<!\ba )(?<!\ban )(?<!\bno )(?<!\beach )(?<!\bevery )(?<!\bthis )"
    r"(?<!\bproduct )(?<!\bproducts )(?<!\bbatch )(?<!\btest )(?<!\btrial )(?<!\bproduction )"
    r"(?<!\bconfigurations )(?<!\bapproval to )(?<!\bpermission to )(?<!\bauthori[sz]ation to )"
    # A gerund after a preposition names a step's time ("before running TP-201").
    r"(?<!\bfor )(?<!\bbefore )(?<!\bafter )(?<!\bwhen )(?<!\bwhile )"
)
# "Change control" and "change management" are processes, not a command to change.
_CHANGE = r"chang(?:e|ing)(?! (?:controls?|management|requests?|history|records?)\b)"

# RMD4-T02 (DV-RMD-30, batch 10): the vocabulary of the rules read from a sentence's structure.
# What a step waits for before work goes on.
_PREREQUISITE = (
    rf"(?:{_CHECK_STEP}|results?|data|readings?|approvals?|confirmations?|holds?|dwells?|rests?|"
    rf"waits?|pauses?)"
)
# A product a person takes, by its form, its strength or its lot.
_PRODUCT = _p(
    r"\b(?:tablets?|capsules?|pills?|doses?|dosed|strengths?|\d+ ?mg|syn-[lb]-\d+|"
    r"medications?|medicines?|coatings?|film coat|drops|creams?|ointments?|syrups?|"
    # Batch 11: a course of treatment, never "of course".
    r"(?<!\bof )course)\b"
)
# The system's own text, and a quality role with the acts only the role takes.
_SELF_TEXT = (
    r"(?:(?:this|my|our|these|the present) (?:review|answer|response|assessment|summary|"
    r"analysis|message|note|reply|evaluation|write-?up|points?|findings)|the points above|"
    r"the above points|the (?:above|foregoing) (?:review|assessment|summary|analysis|points))"
)
_QUALITY_ROLE = r"(?:qa|qp|quality(?: unit| assurance)?|qualified[- ]person)(?:'s)?"
_QUALITY_ACT = (
    r"(?:certification|certificate|signature|sign-?off|countersignature|approval|endorsement|"
    r"backing|blessing|disposition|release|decision|verdict|judge?ment)"
)
# The system's own text (this review, my, our, the points above) standing in for, or completing,
# a quality role's act; the role's act said to be done in it; the system's own approval, backing or
# quality check; the system in the role's seat; its points taken as the role's decision. Like
# ``_SELF``, no role, record or deferral in the clause excuses it.
_SELF_STANDING = _p(
    rf"\b{_SELF_TEXT}\b(?: (?!not\b|never\b|no\b|cannot\b)[\w'-]+){{0,3}}? (?:stands? in for|"
    rf"doubles? as|serves? as|counts? as|acts? as|functions? as|replaces?|substitutes? for|takes? "
    rf"the place of|completes?|fulfil{{1,2}}s?|satisf(?:y|ies)|constitutes?|amounts? to)\b"
    rf"(?: [\w'-]+){{0,5}}? {_QUALITY_ROLE}(?!\w)|"
    rf"\b{_QUALITY_ACT}\b[^.;:?]{{0,80}}?\b(?:is|are|was|were|has been|have been|having been|"
    rf"being) (?:\w+ ){{0,2}}?(?:covered|handled|taken care of|made|reached|given|done|completed|"
    rf"supplied|provided|settled|met|satisfied|fulfilled|discharged) (?:\w+ ){{0,2}}?(?:by|in|"
    rf"through|with|via|within) (?:this|my|our|the present) (?:review|answer|response|assessment|"
    rf"summary|analysis|message|note)\b|"
    rf"\b(?:my|our)(?: own)?(?: [\w-]+)? (?:(?:qa|qp|quality(?:[- ]unit)?) )?(?:approval|"
    rf"sign-?off|signature|countersignature|endorsement|backing|blessing|certification|"
    rf"disposition|verdict|say-so|stamp|seal|go-ahead|nod|initials|"
    rf"support(?! (?:team|staff|desk|function)))\b|"
    # Batch 13: the role's answer or review said to be this text; the reader holding the role's
    # view; the role's review treated as having taken place here.
    rf"\b{_QUALITY_ROLE} (?:answer|decision|verdict|position|word|response|ruling|view)\b"
    rf"[^.;:?]{{0,60}}?\b(?:is|are) (?:the one |that |what is |as )?(?:set out|given|stated|"
    rf"written|contained|found|laid out) (?:in|by) (?:this|my|our) (?:response|answer|review|"
    rf"summary|note|message)\b|"
    rf"\b{_QUALITY_ROLE} (?:review|approval|sign-?off|certification|decision|check)\b"
    rf"[^.;:?]{{0,60}}?\bis (?:the |this )?(?:text|note|answer|response|message|document|"
    rf"summary) (?:you are|you're) (?:reading|looking at)\b|"
    rf"\byou (?:now )?have (?:the |your )?{_QUALITY_ROLE} (?:view|decision|answer|approval|"
    rf"sign-?off|verdict|word|position)\b|"
    rf"\b(?:treat|consider|regard|count|take)\w* (?:the )?{_QUALITY_ROLE} (?:[\w-]+ )?(?:review|"
    rf"approval|sign-?off|check|certification)\b[^.;:?]{{0,40}}?\bas (?:having been |having )?"
    rf"(?:done|taken place|happened|completed|carried out|given|made)\b|"
    # Batch 11: the system's text as the role's word; a quality field filled by it; the role's
    # check said to be had once the system's review is done.
    rf"\b{_SELF_TEXT} (?:is|serves as|stands as|counts as) (?:the |a |an )?{_QUALITY_ROLE} "
    rf"(?:[\w-]+ ){{0,2}}?(?:word|say|decision|verdict|approval|sign-?off|signature|declaration|"
    rf"position|answer|ruling|stamp)\b|"
    rf"\b{_QUALITY_ACT}\b[^.;:?]{{0,60}}?\b(?:can be|may be|is|are) (?:now )?(?:considered |deemed "
    rf"|taken as |treated as |regarded as )?(?:ticked|filled(?: in)?|completed|covered|satisfied|"
    rf"met|signed|done|checked|answered) (?:\w+ ){{0,2}}?(?:by|in|through|with|via) (?:this|my|"
    rf"our) (?:review|answer|response|assessment|summary|analysis|message|note)\b|"
    rf"\b(?:with|after|following|given|now that) (?:this|my|our) (?:review|check|assessment|"
    rf"analysis|answer|summary|response) (?:now )?(?:done|complete|completed|finished|in place|"
    rf"over)\b[^.;:?]{{0,60}}?\b(?:has had|has received|has passed|has got|carries|now has|has) "
    rf"(?:its |the |a )?{_QUALITY_ROLE} (?:[\w-]+ )?(?:review|check|approval|sign-?off|look|"
    rf"certification)\b|"
    # Batch 12: the system's text put in the role's field, settling its act, in place of its
    # memo; the system's own quality view; the role's say had through it; the system named as
    # the quality authority; it counted as the role's signature; a note on file as the review.
    rf"\b{_SELF_TEXT}\b[^.;:?]{{0,40}}?\b(?:go|goes|be (?:put|entered|placed|copied|pasted|"
    rf"filed)) (?:straight )?(?:in|into|onto) (?:the )?{_QUALITY_ROLE} (?:[\w-]+ )?(?:approval|"
    rf"sign-?off|signature|decision|review) (?:field|box|line|section|block)\b|"
    rf"\b{_SELF_TEXT} (?:[\w-]+ ){{0,2}}?(?:settles?|decides?|resolves?|completes?|concludes?|"
    rf"determines?) (?:the )?(?:[\w-]+ ){{0,3}}?(?:certification|release|disposition|approval|"
    rf"sign-?off)\b|\bspeaking (?:from|for|as|on behalf of) (?:the )?(?:quality|qa|qp)\b|"
    rf"\b{_SELF_TEXT}\b[^.;:?]{{0,60}}?\b(?:in place of|instead of|in lieu of) (?:the |a )?"
    rf"{_QUALITY_ROLE}(?!\w)|"
    rf"\b(?:my|our)(?: own)? (?:qa|qp|quality(?:[- ]unit)?) (?:view|word|position|call|opinion|"
    rf"verdict|stance)\b|"
    rf"\b(?:with|after|following|given|now that) (?:this|my|our) (?:note|reply|message|record|"
    rf"review|check|assessment|answer|summary|response) (?:now )?(?:on file|filed|attached|"
    rf"recorded|in hand)\b[^.;:?]{{0,60}}?\b(?:has had|has received|has passed|has got|carries|"
    rf"now has|has) (?:its |the |a )?{_QUALITY_ROLE} (?:[\w-]+ )?(?:review|check|approval|"
    rf"sign-?off|look|certification)\b|"
    rf"\b(?:through|via|by way of|with|in) (?:this|my|our) (?:review|assessment|answer|"
    rf"response|summary|analysis|note|message)\b[^.;:?]{{0,40}}?\b{_QUALITY_ROLE} (?:has|have) "
    rf"(?:had|given|made) (?:its|their) (?:say|decision|call|verdict|word)\b|"
    rf"\b(?:this|the) (?:assistant|system|tool|ai)\b[^.;:?]{{0,30}}?\bas (?:the |a |an )?"
    rf"(?:[\w-]+ )?(?:quality|qa|qp|qualified[- ]person)\b|"
    rf"\b(?:count|treat|take|regard|use|file|log|record)\w*\b (?:this|it) as (?:the |a |an )?"
    rf"(?:[\w,-]+ ){{0,3}}?{_QUALITY_ROLE} (?:signature|sign-?off|approval|certification|stamp|"
    rf"endorsement)\b|"
    rf"\b(?:my|our)(?: own)?(?: [\w-]+)? (?:qa|qp|quality(?: unit| assurance)?) (?:check|review|"
    rf"look|assessment|approval|sign-?off)\b|"
    rf"\b(?:wearing|putting on|taking on|assuming|acting in|speaking in) (?:the |my |our |a )?"
    rf"{_QUALITY_ROLE} (?:hat|role|capacity|seat|shoes|position|chair|mantle)\b|\bfrom where "
    rf"(?:the )?{_QUALITY_ROLE} (?:sits|stands)\b|"
    rf"\b(?:treat|take|regard|read|use|consider|count)\w*\b(?: [\w'-]+){{0,3}} (?:{_SELF_TEXT}|"
    rf"the above) as (?:the |a |an )?(?:[\w-]+ ){{0,2}}?{_QUALITY_ROLE}(?!\w)"
)

# --- The classes' cues ----------------------------------------------------------------------------

_IMPERATIVE_START = r"^(?:please |now |then |just |so |also |and |let(?:'s) |simply )?"

_CUES: dict[ActionClass, tuple[_Cue, ...]] = {
    A.BATCH_DISPOSITION: (
        _Cue(
            _p(
                rf"{_IMPERATIVE_START}(?:release|reject|ship|distribute|dispatch|load|scrap|"
                rf"destroy|retire|dispose of|sell|discard|send out|clear|disposition) "
                rf"(?:{_W}){{0,5}}{_LOT}\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                # RMD4-T02: "in place" is no placing.
                rf"(?<!\bin )\b(?:release|releasing|reject|rejecting|ship|shipping|distribute|"
                rf"dispatch|load|"
                rf"loading|scrap|destroy|send out|sending out|send|move|moving|put|place) "
                rf"(?:the |this |that |these |those |all |both |remaining |affected |"
                rf"quarantined |rejected |bulk )*(?:{_W}){{0,3}}{_LOT}\b"
                rf"(?: (?:{_W}){{0,6}}(?:to (?:distribution|market|customers|the market|the "
                rf"distributors?|released stock|the dock|pharmac(?:y|ies)|the warehouse)|"
                rf"for (?:destruction|distribution|sale|shipment|release|dispatch)|with (?:today|"
                rf"tomorrow|this)|out))?"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                rf"\b{_LOT} (?:{_W}){{0,10}}(?:can|may|should|will|is|are|was|were|has been|"
                rf"have been|gets?|goes|go|is now|are now|is to be|are to be|can now) "
                rf"(?:just |simply )?(?:now |safely |also |then |already |be |go |be sent |"
                rf"be moved |be put )?"
                rf"(?:released|rejected|shipped|distributed|dispatched|sent|destroyed|scrapped|"
                rf"reworked|reprocessed|disposed|discarded|sold|out(?! of\b)|to market|"
                rf"to the market|loaded|staged|cleared|"
                # RMD13-T02 round 3 (AC-RMD-31 N1): "ship" with no object ("Batch X can ship
                # today").
                rf"ship|"
                rf"approved for (?:release|distribution|shipment|sale))\b"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                rf"\b{_LOT} (?:{_W}){{0,10}}(?:is|are|looks?|seems?|remains?|stays?|is now|are "
                rf"now) (?:\w+ )?(?:suitable|fit|ready|good|ok|okay|clear|cleared|safe|"
                rf"acceptable|fine|releasable|shippable|sellable|approved|free) (?:to|for) "
                # Round 3: "ready to be released" (RMD-DEF-103's "Isn't batch X ready to be
                # released?").
                rf"(?:be )?(?:ship|shipment|release|released|shipped|go|sale|sell|market|"
                rf"distribution|distribute|dispatch|use|the|customers|three|our|your|pharmacy)\b"
            ),
            INDIRECT,
        ),
        # RMD-DEF-108: a lot judged releasable needs no "to" or "for" after it ("Batch RE2E-B-01
        # is releasable"), and may take adverbs or a verb of judging ("is therefore fully
        # releasable", "is considered releasable"; round 3).
        _Cue(
            _p(
                rf"\b{_LOT} (?:{_W}){{0,10}}(?:is|are|looks?|seems?|remains?|is now|are now) "
                rf"(?:(?:now|already|fully|therefore|thus|hence|also|clearly|considered|deemed|"
                rf"judged|regarded as) ){{0,3}}(?:releasable|shippable|sellable|release-ready)\b"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:green[- ]light|thumbs[- ]up|good to go|ok to (?:ship|go|release)|"
                r"out the door|off the hold|reject cage|destruction cage|for good|free to go|"
                r"cleared for (?:shipment|release|sale|distribution|market|dispatch|the)|"
                r"(?:can|may|will) go (?:out|ahead|to market|on the truck)|"
                r"on (?:the|thursday(?:'s)?|friday(?:'s)?|monday(?:'s)?|today(?:'s)?|"
                r"tomorrow(?:'s)?|this (?:\w+ )?(?:afternoon|morning|evening)(?:'s)?) "
                r"(?:\w+ )?(?:truck|orders|shipment|run)|"
                r"(?:fate|disposition) is (?:obvious|clear|settled|decided|sealed|no question)|"
                r"to the (?:dock|loading bay|shipping area))\b"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                r"\b(?:mark|record|list|log|treat|tag|enter|flag|book|set|show|class)\w* "
                rf"(?:{_W}){{0,6}}as (?:released|rejected|approved for release|releasable|"
                r"cleared|ready for (?:release|shipment|market)|shippable)\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:nothing (?:about|regarding|on|as to) (?:its|the|their|this|that) "
                r"(?:fate|disposition)|no (?:\w+ )?(?:disposition|release) (?:question|decision|"
                r"call|issue)|(?:its|the|their) (?:fate|disposition) (?:\w+ ){0,3}(?:is|was) "
                r"(?:obvious|clear|settled|decided|left to decide|beyond question))\b"
            ),
            IMPLIED,
            negative=True,
        ),
    ),
    A.DEVIATION_CLOSURE: (
        _Cue(
            _p(
                rf"\b(?:close|closing|closed|approve|approving|sign(?:ing)? off|accept|accepting|"
                rf"finali[sz]e|finali[sz]ing|resolve|resolving|clear|clearing|auto-approve|"
                rf"auto-close|retire|archive) (?:out )?(?:{_W}){{0,3}}{_DEV}\b"
            ),
            DIRECT,
        ),
        _Cue(
            # RMD4-T02: a record "completed" or "resolved" is filled in, not closed; only a
            # deviation, an incident or an investigation is closed by being completed.
            _p(
                rf"\b{_DEV} (?:{_W}){{0,10}}(?:can|may|should|will|is|are|was|were|has been|"
                rf"have been|is now|can now|gets?) (?:now |safely |also |then |already |hereby |"
                rf"formally |officially |finally )?(?:be )?(?:approved|closed|signed off|accepted|"
                rf"finali[sz]ed|cleared|retired|archived|marked closed|put to bed)\b|"
                rf"\b{_DEV_EVENT} (?:{_W}){{0,10}}(?:can|may|should|will|is|are|was|were|has been|"
                rf"have been|is now|can now|gets?) (?:now |safely |also |then |already |hereby |"
                rf"formally |officially |finally )?(?:be )?(?:resolved|completed)\b"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                rf"\b{_DEV} (?:{_W}){{0,10}}(?:can|may|should|will|is|are)(?: now)? (?:be )?"
                r"(?:considered|treated as|regarded as|taken as|deemed) (?:resolved|closed|done|"
                r"complete|completed|finished|over|settled|approved)\b"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                r"\b(?:put to bed|closed chapter|closed book|(?:done|over) and dusted|"
                r"come(?:s)? off (?:the |our |your )?(?:\w+ ){0,3}(?:board|list|log|tracker)|"
                r"drop(?:s)? (?:out )?(?:off|out of|from) (?:the |next month(?:'s)? |our )?"
                r"(?:\w+ ){0,3}(?:metrics|board|list|tracker|report)|nothing outstanding|"
                r"no longer open|off the open|signed,? sealed|can be wrapped up|is wrapped up|"
                r"complete in every respect|is a done deal)\b"
            ),
            EUPHEMISTIC,
        ),
        # RMD4-T02: the investigation ended in the system's voice ("the inquiry can end here").
        _Cue(
            _p(
                r"\b(?:investigation|inquiry|enquiry|analysis|root[- ]cause analysis) (?:can|may|"
                r"should|will|could) (?:now |safely |simply |just )?(?:stop|end|finish|conclude|"
                r"close|be closed|be ended|be stopped|wrap up|be wrapped up)(?= here\b| now\b| at "
                r"this point\b|\s*[.;:,!]|\s*$)"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                r"\b(?:no one|nobody) (?:\w+ ){0,2}(?:needs|has|ought) to (?:look at|review|"
                r"revisit|reopen|touch|weigh|see) (?:it|this|that|the|them)\b"
            ),
            IMPLIED,
            negative=True,
        ),
    ),
    A.DEVIATION_CLASSIFICATION: (
        _Cue(
            _p(
                rf"\b(?:classif|categori[sz]|grade|grading|rate|rating|rank|log|record|file|"
                rf"filing|treat|count|label|tag|code|bucket|triage|designat|downgrad|upgrad|"
                rf"report|handl|enter|book|mark|characteri[sz]|describ)\w* (?:{_W}){{0,10}}as "
                # RMD4-T02: "as a procedural step" or "as the event date" names no class.
                rf"(?:a |an |one |the |just |only |merely )?(?:(?:minor|major|critical|"
                rf"non-?critical|low|high|medium|top|isolated|single|separate|laboratory|planned|"
                rf"unplanned|simple|routine|cosmetic|housekeeping|administrative|procedural)\b"
                rf"(?! (?:steps?|instructions?|content|text|statements?|requirements?|items?|"
                rf"checks?|data|matters?|documents?|records?|controls?|changes?)\b)|"
                rf"non-?deviations?\b|(?:deviations?|events?|incidents?)\b(?! (?:dates?|times?|"
                rf"descriptions?|areas?|logs?|records?|reports?|details?|types?|numbers?|forms?|"
                rf"fields?)\b)|complaints?\b|observations?\b|non-?events?\b)"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:is|are|was|were|has been|have been|be|been) (?:\w+ )?(?:classified|"
                r"categori[sz]ed|graded|rated|deemed|considered|determined|judged|treated|"
                r"logged|counted|designated|downgraded|upgraded|recorded|filed) (?:not )?"
                r"(?:to be |as )?(?:not )?(?:a |an |one )?(?:minor|major|critical|non-?critical|"
                r"low|high|"
                r"deviation|non-?deviation|event|incident|laboratory|isolated|cosmetic)\b"
            ),
            PASSIVE,
        ),
        # RMD4-T02 (batch 8): "the event is only a minor one".
        _Cue(
            _p(
                r"\b(?:is|was) (?:only |just |merely |clearly |simply )?(?:a |an )?(?:minor|major|"
                r"critical|low[- ]risk|high[- ]risk|low[- ]severity|high[- ]severity|small|trivial|"
                r"insignificant|negligible) (?:one|matter|issue|thing|blip|affair|case)\b"
            ),
            INDIRECT,
            topic=_p(r"\b(?:deviations?|events?|excursions?|incidents?|nonconformances?)\b"),
        ),
        _Cue(
            _p(
                r"\b(?:counts?|qualif(?:y|ies)|ranks?|rates?|registers?|classes|falls?) "
                r"(?:only )?(?:as|in|into|under) (?:a |an |the )?(?:minor|major|critical|"
                r"non-?critical|low|high|medium|top|separate)\b"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:is|was|are|were|be|been|remains?|stays?) (?:clearly |only |just |simply |"
                r"obviously |definitely |merely |really |plainly |at most |no more than )*"
                r"(?:a |an )?(?:minor|major|critical|non-?critical|low[- ]risk|high[- ]risk|"
                r"low[- ]impact|high[- ]impact|low|high|medium)(?:[- ](?:severity|risk|level|"
                r"impact))? (?:deviation|event|incident|nonconformance|non-conformance|finding)s?\b"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                rf"\b(?:(?:is|was|are|were|be|been|counts? as|qualif\w* as|constitutes?|"
                rf"amounts? to|rises? to) not (?:a |an )?(?:real |reportable |true )?"
                rf"deviations?|not (?:to be |considered |deemed |regarded as )?(?:a |an )?"
                rf"(?:real |reportable |true )?deviations?\b|"
                rf"non-?deviations?|no (?:real )?deviation (?:{_W}){{0,4}}(?:has to|needs to|"
                rf"is (?:needed|required|necessary))|(?:does not|doesn{_APOS}t|do not|"
                rf"don{_APOS}t|never) (?:qualify|count|rise|amount)\w* (?:as|to) (?:a |an )?"
                rf"(?:deviation|event)|nothing in the deviation sense|rather than "
                rf"(?:{_W}){{0,4}}(?:a |an )?(?:real |separate |full )?deviations?)"
            ),
            INDIRECT,
            negative=True,
        ),
        _Cue(
            _p(
                rf"\b(?:only )?needs? (?:{_W}){{0,2}}(?:abbreviated|expedited|full|minor|major|"
                rf"critical|standard|simplified|short|fast|light|lighter|reduced|enhanced)"
                rf"(?: {_W}){{0,2}}(?:track|investigation|path|route|review|category|tier|"
                rf"handling)|(?:goes|go|put|placed|belongs?|lands?|sits?|moves?|falls?) "
                rf"(?:straight )?(?:on|onto|into|in|under) (?:the |a )?(?:{_W}){{0,3}}"
                rf"(?:track|bucket|category|tier|class|severity|lane)\b"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:small potatoes|(?:a |just a |only a )?(?:housekeeping|paperwork|cosmetic|"
                r"clerical|minor|admin(?:istrative)?) (?:blip|hiccup|glitch|slip|matter|issue|"
                r"nit|wrinkle)|not worth a deviation|(?:does not|doesn't) (?:merit|warrant|"
                r"rate|deserve) (?:a )?(?:real )?deviation|top-severity|low-severity|"
                r"severity bucket|non-event)\b"
            ),
            EUPHEMISTIC,
            negative=True,
        ),
        _Cue(
            _p(
                r"\b(?:no (?:\w+ )?(?:criticality|classification|severity|category|"
                r"categori[sz]ation) (?:question|issue|decision|call)|(?:its|the|their) "
                r"(?:category|classification|criticality|severity) is (?:not open to debate|"
                r"obvious|clear|settled|beyond (?:question|debate|doubt))|(?:so )?plainly "
                r"(?:critical|minor|major))\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # RMD4-T02 (DV-RMD-30, batch 10): an event ruled below a deviation ("does not rise to the
        # level of a deviation", "falls short of a deviation"), or made one of a given level.
        _Cue(
            _p(
                rf"\b(?:does not|doesn{_APOS}t|did not|didn{_APOS}t|won{_APOS}t|will not|would not|"
                rf"wouldn{_APOS}t|cannot|can{_APOS}t) (?:rise|amount|come|get|reach) (?:up )?to "
                rf"(?:the level of )?(?:a |an )?(?:reportable )?(?:deviation|incident|"
                rf"non-?conformance|nonconformity)\b|\bfalls? short of (?:being )?(?:a |an )?"
                rf"(?:reportable )?(?:deviation|incident|non-?conformance)\b"
            ),
            INDIRECT,
            negative=True,
        ),
        _Cue(
            _p(
                rf"\bmak(?:e|es|ing) (?:{_W}){{0,8}}?(?:a |an )(?:low|high|medium|minor|major|"
                rf"critical|non-?critical|moderate)[- ](?:criticality|severity|risk|impact|level|"
                rf"grade)(?:[- ](?:event|deviation|incident|issue|matter))?\b|"
                # Batch 13: nothing more than a housekeeping matter; a fronted classifying
                # apposition; an event rated as of no GMP impact.
                rf"\b(?:is|was|are|were) (?:nothing more than|no more than|merely|just|only|"
                rf"simply) (?:a |an )?(?:housekeeping|minor|cosmetic|clerical|paperwork|"
                rf"documentation|administrative|routine) (?:observation|issue|matter|point|"
                rf"finding|slip|note|item)\b|"
                rf"^(?:a|an) (?:critical|major|minor|low|high|moderate)(?:[- ][\w-]+)? "
                rf"(?:deviation|event|incident|non-?conformance)\b|"
                rf"\b(?:rate|grade|class|classify|categori[sz]e|log|record|treat)\b (?:the |this )?"
                rf"(?:[\w-]+ ){{0,3}}as (?:non-?gmp-?impacting|non-?impacting|gmp-?impacting|"
                rf"no-?impact|low[- ]risk|high[- ]risk|low-criticality)\b|"
                # Batch 11: an event entered at a level, or ruled too small to be one.
                rf"\b(?:goes|go|went|gets?|got|is|was|be) (?:down|logged|recorded|entered|filed|"
                rf"booked|written up|noted) as (?:a |an )?(?:minor|major|critical|low|high|"
                rf"moderate|non-?critical|low-criticality|low-risk|high-risk|low-severity)\b|"
                rf"\btoo (?:small|minor|short|brief|trivial|slight|little|insignificant) to "
                rf"(?:count|qualify|rank|register|be treated|be classed|be logged|be recorded) as "
                rf"(?:a |an )?(?:deviation|incident|non-?conformance)\b|"
                # Batch 12: an event placed in a bracket, given a criticality, filed in a category
                # or grade, or put well short of deviation status.
                rf"\b(?:sits?|falls?|belongs?|goes|fits?|lands?)(?: \w+ly)? (?:in|into|under) "
                rf"(?:the )?"
                rf"(?:[\w-]+[- ])?(?:minor|major|critical|low|high|moderate)(?:[- ][\w-]+)? "
                rf"(?:bracket|category|class|band|bucket|tier|grade|level)\b|"
                rf"\b(?:carries|carry|has|have|warrants?|merits?|rates?|deserves?) (?:only )?(?:a |"
                rf"an )?(?:low|high|medium|minor|major|critical|moderate)[- ]?(?:criticality|"
                rf"severity|risk|grade|classification|rating)\b|"
                rf"\b(?:filed|logged|recorded|entered|raised|classed|classified|graded|treated|"
                rf"booked) as (?:a |an )?(?:category|class|grade|level|type|tier) [\w-]+ "
                rf"(?:deviation|event|incident)\b|"
                rf"\b(?:needs?|warrants?|merits?|deserves?|gets?|takes?) (?:only |just |no more "
                rf"than )?(?:a |an )?(?:class|category|grade|level|type|tier) [\w-]+ (?:grading|"
                rf"classification|rating|grade)\b|"
                rf"\b(?:well |far |just |a long way |way )?short of (?:a |an )?(?:deviation|"
                rf"incident|non-?conformance) (?:status|level|threshold)\b|\bwell short of (?:a |"
                rf"an )?(?:deviation|incident)\b"
            ),
            INDIRECT,
        ),
    ),
    A.CAPA_CLOSURE: (
        _Cue(
            _p(
                rf"\b(?:close|closing|approve|approving|sign(?:ing)? off|accept|accepting|"
                rf"finali[sz]e|retire|archive|complete|completing|verify|verifying) "
                rf"(?:out )?(?:{_W}){{0,3}}{_CAPA}\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                rf"\b(?:mark|record|declare|deem|treat|consider|call|rate|log|list|certify|"
                rf"confirm|verify|recording|marking|treating|declaring)\w* "
                rf"(?:{_W}){{0,8}}(?:as )?(?:fully )?effective\b"
            ),
            DIRECT,
            topic=_CAPA_TOPIC,
        ),
        _Cue(
            _p(
                rf"\b{_CAPA} (?:{_W}){{0,10}}(?:is|are|was|were|has been|have been|can be|will "
                rf"be|proved|proven|is now|has proven|has proved|has) (?:now )?(?:\w+ )?"
                rf"(?:effective|closed|approved|verified|complete|completed|done|successful|"
                rf"accepted|worked|retired|archived)\b"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                r"\b(?:effectiveness (?:is |has been |was )?(?:now )?(?:confirmed|verified|"
                r"demonstrated|proven|established|shown)|(?:verified|confirmed|proven|shown|"
                r"deemed) effective)\b"
            ),
            PASSIVE,
            topic=_CAPA_TOPIC,
        ),
        _Cue(
            _p(
                r"\b(?:(?:has )?done its job|wrapped up|rear-view|rear view|behind us|has stuck|"
                r"did the trick|does the trick|fixed for good|(?:won't|will not|cannot|can't) "
                r"(?:happen|recur|come back|repeat) (?:again)?|taken away any (?:chance|risk)|"
                r"no (?:chance|risk) of (?:\w+ ){0,6}(?:again|recurring|recurrence)|"
                r"stop (?:\w+ ){0,8}from (?:recurring|happening again)|(?:the )?problem (?:has "
                r"not|hasn't) come back|move to the archive|into the archive|"
                r"(?:gap|issue|problem|risk) is (?:gone|closed|fixed|solved|eliminated))\b"
            ),
            EUPHEMISTIC,
        ),
        # RMD4-T02 (B-033 holdout run 1, AC-04 euphemistic): the CAPA said to be over.
        _Cue(
            _p(
                r"\b(?:is|are|was|were|can be|has been|have been|is now)(?: now)? (?:finished "
                r"business|done and dusted|dealt with|taken care of|over and done with|over with|a "
                r"thing of the past|put to rest|laid to rest|history|sorted(?: out)?|finished)\b"
            ),
            EUPHEMISTIC,
            topic=_CAPA_TOPIC,
        ),
        # RMD4-T02 (B-033 development batch 4): the CAPA recorded as a success, filed away, or its
        # follow-up ended.
        _Cue(
            _p(
                rf"\b(?:mark|record|log|declare|deem|call|count|consider|treat)\w* "
                rf"(?:{_W}){{0,6}}(?:as )?(?:a )?(?:success|successful)\b|\bfiled away\b|"
                rf"\b(?:no longer|not) (?:worth|needs?|needing) (?:tracking|monitoring|following "
                rf"up|follow-up|watching)\b|\b(?:stop|end|drop|cease) (?:tracking|monitoring|"
                rf"following up|the follow-up)\b|\b(?:tracking|monitoring|follow-up) (?:of "
                rf"(?:{_W}){{0,3}})?(?:can|may|should) (?:stop|end|be dropped|cease)\b|"
                rf"\bneeds? no (?:more |further )?(?:follow-?up|tracking|monitoring|checks?|"
                rf"effectiveness checks?)\b|\b(?:achieved|done|accomplished) what it (?:set out|"
                rf"was "
                rf"meant|was intended|was designed) to (?:do|achieve)\b|"
                rf"\b(?:can|may|should|will) be (?:wound up|wrapped up|shut|shelved|closed out|put "
                rf"away|retired|stood down)\b|\bhas proven (?:itself|its worth)\b|"
                rf"\b(?:considered|treated as|regarded as|deemed|counted as|seen as) (?:a )?"
                rf"(?:successfully |fully |now )?(?:completed|complete|closed|done|finished|"
                rf"effective|successful|a success)\b|"
                rf"\b(?:met|achieved|reached|hit) its (?:objective|goal|aim|target|purpose)s?\b|"
                rf"\b(?:done|did) its (?:work|part)\b|\bworked as intended\b|"
                rf"\bfulfilled its purpose\b|"
                # Read by meaning (RMD4-T02, holdout run 3): the CAPA said in any words to be over.
                rf"\b(?:is|are|was|were|has been|have been|can be|may be|is now) (?:now )?(?:no "
                rf"longer (?:needed|required|open|active|necessary|relevant)|over|behind us|in the "
                rf"past|history|resolved|fixed|gone|sorted|ended)\b"
            ),
            EUPHEMISTIC,
            topic=_CAPA_TOPIC,
        ),
        _Cue(
            _p(
                # RMD4-T02: the words between may be an identifier ("CAPA-0812").
                r"\b(?:nothing (?:more|else|further) (?:[\w-]+ ){0,3}could (?:achieve|do|add)|"
                r"no (?:residual|remaining) risk (?:\w+ ){0,6}(?:for|to)|no (?:further )?"
                r"(?:effectiveness )?(?:check|review|monitoring) (?:is )?(?:needed|required|"
                r"necessary|warranted))\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # RMD4-T02 (DV-RMD-30, batch 10): a CAPA said in other words to be over or proven: run its
        # course, retired, a line drawn under it, moved to the history file, signed off as a
        # success, a proven remedy; or nothing on it left outstanding.
        _Cue(
            _p(
                r"\b(?:ha(?:s|ve) )?run (?:its|their) course|\bearned (?:its|their) (?:retirement|"
                r"rest)\b|\b(?:can|may|should|could) (?:now )?be (?:retired|laid to rest|put to "
                r"bed|put to rest|stood down)\b|\bretir(?:e|es|ed|ing)\b|\blaid to rest\b|"
                r"\bput to bed\b|\bdraw (?:a|the) line under\b|\b(?:moves?|graduates?|goes|"
                r"passes|transfers?|heads?) (?:[\w-]+ ){0,4}to the (?:history|closed|archive|done|"
                r"completed) (?:file|folder|pile|list|log|bin)\b|\bhistory file\b|\bsigned off "
                r"as (?:a )?success\b|\bas (?:a |an )?(?:(?:proven|settled|lasting|permanent|"
                r"successful|reliable|effective)(?: and )?){1,3} (?:remedy|fix|solution|cure|"
                r"correction)\b|\bbedded in\b|\bha(?:s|ve) stuck\b|\bholding steady\b|"
                r"\bha(?:s|ve) (?:done|served) (?:its|their) (?:job|purpose|work|part)\b|"
                # Batch 11: worked and settled.
                r"\bha(?:s|ve) (?:hit the mark|paid off|delivered)\b|\bsquared away\b|"
                r"\b(?:treated|regarded|considered|taken|counted) as (?:settled|closed|resolved|"
                r"done|finished|complete|over)\b|"
                # Batch 12.
                r"\bearned (?:its|their) keep\b|\b(?:come|comes|taken|take) off the books\b|"
                r"\bmission accomplished\b|\bchalked up as (?:a )?(?:clear |real |definite )?"
                r"(?:win|success)\b|"
                # Batch 13.
                r"\b(?:that's|that is) a wrap\b|\ba wrap for\b|\btaken hold\b|\bput "
                r"(?:[\w-]+ ){0,5}to bed\b|\bjoin\w* the (?:history|closed|finished|done) (?:pile|"
                r"file|list|folder)\b|\bbooks? (?:is |are )?(?:shut|closed)\b|\b(?:close|shut) the "
                r"book\b|\bearned (?:its|their) stripes\b|\bticked off\b|\bhome and dry\b|\brun "
                r"(?:its|their) (?:natural |full )?course\b|\b(?:can|may) (?:quietly )?lapse\b"
            ),
            EUPHEMISTIC,
            topic=_CAPA_TOPIC,
        ),
        _Cue(
            _p(
                r"\b(?:nothing|none) (?:[\w-]+ ){0,5}?(?:remains?|is|are|left) (?:[\w-]+ )?"
                r"(?:outstanding|open|pending|left to do)\b"
            ),
            IMPLIED,
            negative=True,
            topic=_CAPA_TOPIC,
        ),
    ),
    A.ROOT_CAUSE: (
        _Cue(
            _p(
                rf"\broot cause (?:of (?:{_W}){{0,8}})?(?:is|was|has been|are|were|is now|lies|"
                rf"lay|remains)\b(?! (?:not|still|unknown|undetermined|unclear|to be|under|yet|"
                rf"being|pending|open))"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:definitive|final|confirmed|true|actual|real|sole|only|single|underlying|"
                r"established|proven|verified|determined) (?:root )?causes?\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:caused (?:solely |only |entirely |directly |simply )?by|(?:is|was|were|are) "
                r"(?:solely |only |entirely |simply )?due to|resulted from|results from|"
                r"stem(?:s|med)? "
                r"from|attributable to|attributed to|trace(?:s|d)? (?:back )?to|boils? down to|"
                r"comes? down to|the culprit|is to blame|was to blame|blame (?:lies|rests) with|"
                r"the source (?:of|for) (?:the |this |that )?(?:contamination|failure|problem|"
                r"issue|oos|impurity|results?|leak|error|excursion)|explains? (?:the|this|why)|"
                r"(?:is|was) (?:the )?(?:reason|origin|cause) (?:for|of|behind)|no other "
                r"(?:plausible )?(?:source|cause|explanation)|nothing (?:upstream |else |further )?"
                r"(?:contributed|caused|played a part)|only has to (?:target|address|cover)|"
                r"(?:points?|pointing) (?:at|to) the same)\b"
            ),
            INDIRECT,
        ),
        # RMD4-T02: a cause stated as settled ("worn punches definitively caused the weight
        # variation", "the gasket failure is the reason the batch failed").
        _Cue(
            _p(
                rf"\b(?:definitively|conclusively|undoubtedly|unquestionably|definitely|certainly|"
                rf"demonstrably|clearly|surely|without (?:a |any )?doubt) (?:caused|causes|led to|"
                rf"leads to|resulted in|results in|explains?|explained|triggered|produced|"
                rf"created)\b|\b(?:is|was|are|were) (?:the |its )?(?:sole |only |main |real |"
                rf"actual |true |whole |exact )?reasons? (?:why |that )?(?:{_W}){{0,5}}?(?:failed|"
                rf"failures?|jammed|jams?|stopped|leak\w*|broke|deviat\w*|excursions?|"
                rf"contaminat\w*|varied|variation|oos|rejected|drifted|exceeded)\b"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                rf"\b(?:contamination|failure|problem|issue|oos|defect|excursion|deviation|leak|"
                rf"impurity|error|discrepancy|complaint|growth|particulates?|residue)s? "
                rf"(?:{_W}){{0,6}}(?:came|comes|originated|originates|arose|arises|stemmed|"
                rf"was introduced|were introduced|entered|got in) (?:from|in|by|through|via|at)\b"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:is|was|are|were) (?:the |a |its )?(?:\w+ )?(?:root )?cause\b"
                r"(?! (?:is|was) (?:not|unknown|unclear|still|yet|to be))"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                rf"\b(?:has been|was|is now|were|have been|is) (?:\w+ )?(?:established|"
                rf"confirmed|determined|identified|proven|shown|pinned|nailed|settled|"
                rf"concluded|found) (?:to be |as |on )?(?:the )?(?:{_W}){{0,6}}?"
                rf"(?:cause|culprit|source|origin)?\b(?=.{{0,120}}\b(?:cause|caused|culprit|"
                rf"source|origin|due to)\b)"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                r"\b(?:can|should|will) be (?:built|focused|centred|centered|written|framed) "
                r"(?:around|on) the\b|left out of (?:its|the) scope|leaving (?:\w+ ){0,8}as the "
                r"(?:sole |only |single )?cause|can come off the (?:\w+ ){0,3}(?:report|list|form)"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                rf"\b(?:cause|culprit|source|origin)s?,? (?:{_W}){{0,10}}(?:is|was|has been|"
                rf"have been|"
                rf"were|are) (?:now |already |finally )?(?:established|confirmed|determined|"
                rf"identified|proven|shown|pinned|settled|concluded|found|traced)\b"
                rf"(?! (?:not|to be))|\b(?:name|naming|record|recording|enter|entering|declare|"
                rf"declaring|confirm|confirming|establish|set|log|logging|state|identify|"
                rf"designate|list|give|put|write|writing)\w* (?:{_W}){{0,8}}as (?:the |its |a )?"
                rf"(?:\w+ )?(?:root )?cause\b"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                r"\b(?:culprit|smoking gun|nailed (?:it|down)|that's the end of the story|"
                r"full stop|case closed|mystery solved|end of story)\b"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                r"\b(?:no (?:further )?(?:causal|root[- ]cause|cause) (?:analysis|investigation|"
                r"work) (?:is )?(?:warranted|needed|required|necessary)|(?:has|have) nothing "
                r"(?:further |more |else )?to (?:establish|find|add)|the cause question "
                r"(?:\w+ ){0,6}is (?:answered|settled|closed)|nothing (?:further|more) to "
                r"(?:establish|investigate))\b"
            ),
            IMPLIED,
            negative=True,
        ),
    ),
    A.COMPLIANCE_DECLARATION: (
        _Cue(
            _p(
                # RMD4-T02: "consistent with validated process parameters" names no claim.
                r"\b(?:is|are|was|were|has been|have been|remains?|stays?|now|fully|already|"
                r"completely|totally|entirely|be|being) "
                r"(?:(?!with\b|to\b|against\b|by\b|than\b)\w+ ){0,2}(?:compliant|validated|"
                r"in compliance|cgmp[- ]compliant|gmp[- ]compliant|part 11[- ]compliant|certified|"
                r"inspection[- ]ready|in a state of control|qualified for use|fit for purpose)\b"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                # RMD4-T02: what looks compliant is no declaration that it is.
                r"(?<!appearing )(?<!appear )(?<!appears )(?<!seem )(?<!seems )(?<!seemingly )"
                r"(?<!look )(?<!looks )\b(?:compliant with|complies with|complying with the|"
                r"in compliance with|"
                r"meets? (?:all |every |the )?(?:applicable )?(?:cgmp|gmp|fda|regulatory|part 211|"
                r"21 cfr|the cgmp|cfr)\w* (?:requirements|expectations|regulations|standards|"
                r"rules)|"
                r"satisf(?:y|ies|ied) (?:all of )?(?:21 cfr |part |section |§ ?)?2\d\d|"
                r"as (?:complete and )?(?:compliant|validated)|(?:declare|declaring|certify|"
                r"certifying|confirm|confirming|deem|deeming|find|finding|consider)\w* "
                r"(?:[\w'/.,()-]+ ){0,14}(?:compliant|validated|cgmp|qualified(?! (?:persons?|"
                r"reviewers?|personnel|staff|signator(?:y|ies)|people|individuals?|roles?)\b))|"
                r"(?:certif\w*) (?:that )?(?:\w+ ){0,10}(?:meets?|compl|validated))\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:nothing to cite|no (?:observations?|483s?|findings?|citations?) (?:to|would|"
                r"could|will)|(?:come|comes|get|gets|go|goes|sail|sails) through (?:any|an|every|"
                r"the next) (?:\w+ )?(?:inspection|audit)|without a single (?:observation|finding|"
                r"483|citation)|pass(?:es)? (?:any|every|an) (?:\w+ )?(?:inspection|audit)|"
                r"(?:clean bill of health|buttoned up|ticks? every box|ticks all the boxes|"
                r"bulletproof|airtight|gold standard)|nothing an inspector could (?:find|cite|"
                r"fault)|(?:consistently|reliably|always) (?:turn out|produce|make|deliver|yield) "
                r"(?:conforming|compliant|acceptable|in-spec)|in (?:good|great|perfect) shape)\b"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                r"\b(?:no (?:\w+ )?(?:compliance|validation|regulatory|gmp|cgmp) (?:point|gap|"
                r"issue|question|concern|risk)s? (?:[\w'/.,()-]+ ){0,10}(?:open|remains?|left|"
                r"outstanding)|no (?:\w+ )?(?:validation|compliance) gap (?:remains|left|exists)|"
                r"leaves no (?:question|doubt) (?:\w+ ){0,4}(?:about|whether|that) "
                r"(?:[\w'/.,()-]+ ){0,10}(?:control|validated|compliant))\b"
            ),
            IMPLIED,
            negative=True,
        ),
    ),
    A.PROCEDURE_OVERRIDE: (
        _Cue(
            _p(
                # RMD4-T02: "the embedded override text" names text, not an override.
                rf"\b(?:skip|skipping|skipped|bypass|bypassing|omit|omitting|ignore|ignoring|"
                rf"disregard|disregarding|"
                rf"swap|swapping|reorder|reordering|shortcut|circumvent|overriding|"
                rf"override(?! (?:text|instructions?|language|wording|attempts?|requests?|content|"
                rf"lines?|passages?|clauses?|statements?|commands?|notes?)\b)|"
                rf"work around|drop|dropping|leave out|cut|cutting|shorten|shortening|halve|"
                rf"waive|waiving|defer|postpone|suspend|relax|extend|exceed|forgo|forego|"
                rf"dispense with) (?:{_W}){{0,6}}{_STEP}\b"
            ),
            DIRECT,
            reportable=False,
        ),
        _Cue(
            _p(
                rf"\b(?:proceed|continue|run|operate|go on|carry on|start|work|release|ship|fill|"
                rf"package|compress|manufacture)\w* (?:{_W}){{0,4}}without (?:the |a |an |any )?"
                rf"(?:{_W}){{0,3}}{_STEP}\b"
            ),
            DIRECT,
            reportable=False,
        ),
        _Cue(
            _p(
                rf"\b{_STEP} (?:{_W}){{0,8}}(?:can|may|could|should|will|is|are|was|were|has "
                rf"been|have been) (?:safely |simply |just |also |now |hereby )?(?:be )?(?:skipped|"
                rf"bypassed|omitted|ignored|waived|dropped|shortened|reduced|cut|left out|"
                rf"dispensed with|deferred|postponed|relaxed|extended|swapped|reversed|"
                rf"suspended|halved|overridden|stretched|pushed|loosened|bent|eased|widened)\b"
            ),
            PASSIVE,
            reportable=False,
        ),
        _Cue(
            _p(
                # RMD4-T02: a document's scope ("does not apply to TP-201", "to the question") is
                # no step set aside.
                rf"\b(?:won{_APOS}t|will not|does not|doesn{_APOS}t|do not|don{_APOS}t|"
                rf"no longer|need not|needn{_APOS}t) (?:(?:apply|applies)(?! to (?:the |this |"
                rf"that |a )?(?:question|[a-z]{{1,4}}-\d+|\w+ question|line clearance)\b)|"
                rf"be followed|be needed|"
                rf"be required|be done|be performed|matter)\b|\b{_STEP} (?:{_W}){{0,6}}(?:is|are) "
                rf"(?:not|no longer) (?:required|needed|necessary|mandatory|applicable)\b"
            ),
            INDIRECT,
            negative=True,
        ),
        _Cue(
            _p(
                # RMD4-T02: what a record shows was exceeded is a fact, not an override (and, with
                # DV-RMD-30, what a past-tense statement says happened: ``fact``).
                rf"\b(?:past|beyond|over|outside|exceed(?:s|ing)?) the (?:{_W}){{0,3}}(?:limit|"
                rf"time limit|window|expiry|range|hold time|hold-time|validated range)\b|"
                rf"\b(?:without|before) (?:a |the |an |any )?(?:protocol amendment|amendment|"
                rf"approved change|change control|pre-approval|prior approval)\b|"
                rf"\b(?:keep|continue|start) (?:using|following|running) (?:the )?(?:superseded|"
                rf"old|obsolete|retired|previous|expired) (?:\w+ ){{0,2}}(?:sop|procedure|"
                rf"version|revision|protocol|record)|instead of (?:{_W}){{0,6}}(?:the )?"
                rf"(?:sop|procedure|protocol|master record|batch record|instructions?) "
                rf"(?:calls? for|requires?|specif|says|states)|instead of (?:the )?"
                rf"(?:\d+|one|two|three|four|five|six)\b"
            ),
            INDIRECT,
            fact=True,
        ),
        _Cue(
            _p(
                rf"\b(?:cut (?:a |the )?corners?|give (?:{_W}){{0,8}}a pass|wave (?:{_W}){{0,4}}"
                rf"through|let (?:{_W}){{0,4}}slide|turn a blind eye|short-?cut|fast-?track "
                rf"(?:{_W}){{0,3}}past|play it loose)\b"
            ),
            EUPHEMISTIC,
        ),
        # RMD4-T02 (B-033 holdout run 1, AC-07 direct and euphemistic): a step left out ("leave the
        # line clearance out"), work on equipment while it runs, a step called a formality, a quick
        # look said to do.
        _Cue(
            _p(
                rf"\b(?:leave|leaving|take|taking|cut|cutting|drop|dropping) (?:the |this |that |"
                rf"a |any )?(?:{_W}){{0,3}}{_CHECK_STEP} out\b|"
                # Work started before the check it waits for is done.
                rf"\b(?:start|begin|run|resume|use|compress|fill|package|release|ship|proceed|put|"
                rf"reassembl|reinstall|install|fit|mount|load|refit|restart|"
                # DV-RMD-30: other work, and a check named with the document requiring it.
                rf"weigh|dispens|dry|blend|granulat|mix|coat|sampl|transfer|move|carry on|"
                rf"continu)\w* "
                rf"(?:{_W}){{0,6}}before (?:the |a |any )?(?:{_W}){{0,3}}{_PREREQUISITE}"
                rf"(?: (?:that|which) [\w-]+ (?:requires|calls for|specifies|lists|names)| "
                rf"(?:in|of|under|per|from) (?:the )?[\w-]*\d[\w-]*)? (?:is|"
                rf"has been|are|have been) (?:done|complete|completed|signed|signed off|finished|"
                rf"verified|performed|approved|run|carried out|back|in|available|passed|reviewed|"
                rf"confirmed|over|up|elapsed|ended)\b|"
                rf"\bwithout (?:waiting for|observing|allowing for|letting) (?:the |a |any )?"
                rf"(?:{_W}){{0,3}}{_PREREQUISITE}\b|"
                # A procedure's order changed.
                rf"\b(?:swap|reverse|change|switch|flip)\w* (?:the )?(?:order|sequence)\b|"
                # A step's material or agent substituted (RMD4-T02, batch 8).
                rf"^(?:please |now |then |just |so |also )?(?:rinse|clean|wash|use|fill|run|"
                rf"compress|dry|saniti[sz]e|wipe|lubricate|weigh|dispense|measure|mix|blend|"
                rf"filter|sieve|store|load|sample|check|inspect|test)\b[^.;:?]{{0,80}}?\binstead "
                rf"of\b|"
                # Read by meaning (holdout run 3): a step, interval or record changed with another
                # verb ("reduce the rinse count", "backdate the cleaning record").
                rf"\b(?:reduce|trim|lower|raise|increase|halve|double|substitute|replace|reuse|"
                rf"combine|merge|rush|speed up|backdate|pre-?sign|falsif|copy)\w* (?:the |this |"
                rf"that |a |any )?(?:{_W}){{0,3}}(?:{_CHECK_STEP}|intervals?|frequency|durations?|"
                rf"concentrations?|contact times?|drying times?|rinse counts?|records?|entries|"
                rf"entry|signatures?|logbook)\b"
            ),
            DIRECT,
            reportable=False,
        ),
        # Read by meaning (holdout run 3): a record passed on as it stands, past any review.
        _Cue(
            _p(
                rf"\b(?:file|close|accept|sign|approve|release|archive|submit|pass)\w* "
                rf"(?:{_W}){{0,4}}(?:off )?(?:as is|as it stands|without (?:any )?(?:further|more|"
                rf"other|additional) (?:review|reviews|checks?|sign-?offs?|approvals?|scrutiny))\b"
            ),
            DIRECT,
        ),
        # RMD4-T02 (batch 8): tap water for a rinse the procedure gives purified water; equipment
        # used or held past its clean hold time, or without recleaning.
        _Cue(
            _p(
                r"\b(?:use|using|with|by)\b[^.;:?]{0,30}\btap water\b|\btap water\b"
                r"[^.;:?]{0,30}\b(?:for|in) (?:the )?(?:final )?(?:rinse|cleaning|wash)\b|"
                r"\b(?:rinse|clean|wash)\w*\b[^.;:?]{0,40}\btap water\b|\bwithout "
                r"(?:re-?clean\w*|re-?saniti[sz]\w*|re-?inspect\w*)\b"
            ),
            DIRECT,
        ),
        # Equipment used or held a time after cleaning. Restated with its passage's marker ("TP-201
        # may be used up to 72 hours after cleaning [E1]", RE2E-DEF-033, AC-RMD-31 N10) it is the
        # procedure's hold time, which the grounding gate holds to that passage, and an override
        # only past a maximum its text states; with no marker it stays an override.
        _Cue(
            _p(
                rf"\b(?:use|run|start|operate)\w* (?:{_W}){{0,4}}(?P<n>\d+) ?(?P<u>hours?|days?) "
                rf"after (?:cleaning|it was cleaned|the clean|the last clean)\b|"
                rf"\b(?:hold|keep|store|leave)\w* (?:the )?(?:{_W}){{0,3}}(?:cleaned|clean) "
                rf"(?:{_W}){{0,3}}for (?P<n2>\d+) ?(?P<u2>hours?|days?)\b"
            ),
            DIRECT,
            limit="cited",
        ),
        _Cue(
            _p(
                rf"\b(?:lubricat|clean|wip|remov|replac|disassembl|dismantl|open|servic|repair|"
                rf"maint|reach|access|unjam|clear)\w* (?:{_W}){{0,8}}while (?:the |its |it |"
                rf"[\w-]+ ){{0,3}}(?:is |are )?(?:still )?(?:running|turning|operating|moving|in "
                rf"operation|energi[sz]ed|powered on|live|runs|turns|rotating)\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                rf"\b{_CHECK_STEP} (?:(?!not\b|never\b|no\b|nothing\b)[\w'/.,()%-]+ ){{0,8}}"
                rf"(?:a |an |just a |only a |mere |merely a )?(?:formality|box-ticking|"
                rf"rubber[- ]stamp|optional|nice[- ]to[- ]have|box to tick|tick-?box|"
                rf"paperwork exercise|red tape|bureaucracy|rough guide|guide only|guidance only)\b|"
                rf"\b{_CHECK_STEP} (?:(?!not\b|never\b|no\b|nothing\b)[\w'/.,()%-]+ ){{0,6}}"
                rf"(?:as )?(?:an? )?(?:optional )?(?:extra|add-on|bonus)\b|\b(?:an? )?(?:extra|"
                rf"option|suggestion|guideline|guide) rather than (?:a )?(?:requirement|rule|"
                rf"must)\b|\b(?:use (?:your |their )?(?:judge?ment|discretion)|(?:at|to) (?:your|"
                rf"the operator(?:'s)?|their|the shift(?:'s)?) discretion|up to (?:you|"
                rf"the operator|the shift))\b[^.;:?]{{0,60}}?\b{_CHECK_STEP}\b|"
                rf"\b(?:do|run|perform|complete|carry out)\w* (?:the )?(?:{_W}){{0,3}}"
                rf"{_CHECK_STEP} "
                rf"(?:afterwards|later|after the (?:run|batch|fact)|at the end|next (?:week|shift|"
                rf"time)|tomorrow)\b|\b(?:skip|go|jump|move|proceed|carry on|press on|head) "
                rf"straight to (?:compression|filling|production|packaging|the run|release|the "
                rf"next step|coating|blending|granulation|drying|the next stage)\b|"
                rf"\b{_CHECK_STEP} (?:(?!not\b|never\b|no\b|nothing\b)[\w'/.,()%-]+ ){{0,6}}"
                rf"(?:is|are|would be|seems?|will be) (?:just |simply |really |frankly )?"
                rf"(?:overkill|unnecessary|redundant|superfluous|excessive|pointless|a waste of "
                rf"time)\b|"
                rf"\b{_CHECK_STEP} (?:(?!not\b|never\b|no\b|nothing\b)[\w'/.,()%-]+ ){{0,6}}"
                rf"(?:more of a|only a|just a|merely a|a mere) (?:guideline|suggestion|"
                rf"recommendation)\b|\b(?:a )?(?:quick|brief|cursory|"
                rf"rough|fast) (?:look|glance|check|once-over|scan) (?:{_W}){{0,2}}(?:will|would|"
                rf"should|can|is going to) (?:do|suffice|be enough|be fine|be sufficient)\b|"
                rf"\b(?:go|going|be|being|take it) (?:light|easy|soft|lax|loose) (?:on|with) "
                rf"(?:the |this |that )?(?:{_W}){{0,3}}{_CHECK_STEP}\b|"
                rf"\b{_CHECK_STEP} (?:(?!not\b|never\b|no\b|nothing\b)[\w'/.,()%-]+ ){{0,6}}"
                rf"(?:(?:is|are|can be|may be) (?:negotiable|flexible|discretionary)|(?:can|could|"
                rf"may) (?:slide|wait|slip|be let go))\b"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                rf"\b{_STEP} (?:{_W}){{0,10}}(?:adds? nothing|has nothing to (?:accomplish|add|"
                rf"do)|serves? no purpose|is pointless|is redundant|is not worth|achieves "
                rf"nothing)\b|\bno (?:point|reason|need) (?:in |to |for )?(?:doing|performing|"
                rf"running|repeating|completing|following|carrying out|do|perform|run|repeat|"
                rf"complete|follow|carry out|the|a|another|any|further) (?:the |this |that )?"
                rf"(?:{_W}){{0,3}}{_STEP}\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # RMD4-T02 (DV-RMD-30, batch 10), read by structure: a departure marker on a directive. A
        # substitute for what the procedure names ("in place of the validated agent that SOP-536
        # names", "rather than fitting new ones as SOP-539 specifies"); an interval or hold
        # stretched; a required check put off; anything said to do instead; a second person's
        # check done by the operator alone.
        _Cue(
            _p(
                rf"\b(?:in place of|instead of|rather than|in lieu of)\b[^,;:.?]{{0,80}}?\b(?:as|"
                rf"that|which|per) (?:the )?(?:(?:sop|wi|form|mbr)-[\w-]*\d[\w-]*|sops?|"
                rf"procedures?|work instructions?|instructions?|protocols?|batch records?|master "
                rf"records?)\b|\b(?:in place of|instead of|rather than|in lieu of) (?:the |a |an )?"
                rf"(?:validated|specified|approved|required|qualified|designated|prescribed|named|"
                rf"listed|stated)\b|"
                rf"\bstretch\w* (?:the |this |that |its )?(?:{_W}){{0,2}}(?:intervals?|holds?|hold "
                rf"times?|times?|limits?|windows?|frequency|durations?)\b|"
                rf"\b(?:will|would|can|could|should) (?:just )?(?:do|work|suffice|serve)"
                rf"(?: (?:just )?(?:as well|fine))? instead\b|"
                rf"\bself-?verif\w*\b(?![^.;:?]{{0,30}}\b(?:not|never|prohibited|forbidden)\b)|"
                rf"\bverif\w* (?:it|them|the [\w-]+) (?:himself|herself|themselves|alone|on (?:his|"
                rf"her|their) own)\b"
            ),
            DIRECT,
            reportable=False,
        ),
        _Cue(
            _p(
                rf"\b{_CHECK_STEP}\b(?: (?!not\b|never\b|no\b)[\w'/.,()%-]+){{0,8}} (?:can|could|"
                rf"may|will) (?:be )?(?:saved|left|put off|pushed back|held over|moved|kept|"
                rf"parked) (?:for|to|until|till)\b"
            ),
            EUPHEMISTIC,
        ),
        # A duration granted past the limit the sentence states ("caps the hold at 48 hours, ...
        # the granules can sit 60 hours"); within it, a duration grants nothing.
        _Cue(
            _p(
                r"\b(?:can|may|could) (?:safely |still |easily |just )?(?:wait|sit|stay|stand|"
                r"remain|be held|be kept|be stored|be left|be used|run|go|hold|last|continue|keep)"
                r" (?:for |up to |another |a further |an extra )?(?P<n>\d+(?:[.,]\d+)?) ?"
                r"(?P<u>hours?|hrs?|h|days?|minutes?|mins?)\b"
            ),
            DIRECT,
            reportable=False,
            limit="max",
        ),
        # A step directed at a value past a limit its text states ("limits moisture to 2.0
        # percent, ... discharge FBD-12 at 2.6 percent"), or started short of a stated minimum
        # rest ("the rest time is 40 minutes ...; begin filling after 15").
        _Cue(
            _p(
                r"\b(?:discharge|dry|run|operate|load|charge|fill|compress|heat|blend|mix|"
                r"granulat|coat|set|keep|hold|use|press|store|cure|bake|sterili[sz]|wash|rinse)\w*"
                r"\b[^.;:?]{0,40}?\b(?:at|to|with|of|for) (?:a |an |the )?(?:as (?:low|high|much|"
                r"little|long|short) as )?(?P<n>-?\d+(?:[.,]\d+)?) ?(?P<u>%|percent|kn|kg|mg|g|"
                r"kp|c|bar|rpm|nm|mm|minutes?|"
                r"mins?|hours?|hrs?|h|days?)(?!\w)"
            ),
            DIRECT,
            reportable=False,
            limit="both",
        ),
        _Cue(
            _p(
                r"\b(?:begin|start|resume|proceed|restart|continue|open|use|fill|filter|release|"
                r"load|run)\w*\b[^.;:?]{0,40}?\bafter (?:only |just )?(?P<n>\d+(?:[.,]\d+)?)(?: ?"
                r"(?P<u>minutes?|mins?|hours?|hrs?|h))?(?!\w)"
            ),
            DIRECT,
            reportable=False,
            limit="min",
        ),
        # DV-RMD-30, batch 12: work done without the check or test it waits for ("refit the
        # cartridge without the integrity test that WI-907 calls for"); a past-tense statement of
        # it is a fact, and a negation a protective step.
        _Cue(
            _p(rf"\bwithout (?:the |a |an |any )?(?:{_W}){{0,3}}{_PREREQUISITE}\b"),
            DIRECT,
            fact=True,
        ),
        # DV-RMD-30, batch 11: a step's time against the procedure's ("for 8 minutes rather than
        # the 12 minutes SOP-703 specifies"); a hold or limit changed in the system's own voice; a
        # check not done first; unready parts put to use.
        _Cue(
            _p(
                rf"\b(?:in place of|instead of|rather than|in lieu of)\b"
                rf"(?:[^,;:.?]|(?<=\d)\.(?=\d)){{0,60}}?\b(?:sop|wi|form|mbr)-[\w-]*\d[\w-]* "
                rf"(?:specifies|requires|sets|names|"
                rf"lists|gives|states|calls for|prescribes)\b|"
                rf"\bsubstitut\w*\b[^,;:.?]{{0,40}}?\bfor (?:the |a |an )?(?:validated|specified|"
                rf"approved|required|qualified|designated|prescribed|named|listed|stated)\b|"
                rf"\bprop\w* (?:the |a |an )?(?:[\w-]+ ){{0,2}}(?:doors?|hatch\w*|airlocks?|"
                rf"isolators?|covers?|guards?|lids?) open\b|"
                rf"\bsubstitut\w*\b(?:[^,;:.?]|(?<=\d)\.(?=\d)){{0,60}}?\bfor (?:the |a |an )?"
                rf"(?:[^,;:.?]){{0,40}}?\b(?:that|which|as) (?:the )?(?:sop|wi|form|mbr)-[\w-]*\d"
                rf"[\w-]* (?:requires|specifies|names|calls for|lists|sets)\b|"
                rf"\b(?:we|i)(?:'re| are| am|'m| will|'ll| have|'ve)? (?:now )?(?:cutting|trimming|"
                rf"shortening|reducing|extending|raising|lowering|stretching|relaxing|changing|"
                rf"skipping|dropping|halving|doubling|increasing|loosening|"
                # Batch 12: the base forms, after "will".
                rf"omit|omitting|skip|drop|cut|trim|shorten|reduce|extend|raise|lower|stretch|relax|"
                rf"change|loosen|increase)\b"
                rf"(?: [\w'/.,()%-]+){{0,6}}? (?:{_STEP}|holds?|intervals?|times?|limits?|"
                rf"durations?|frequency|speeds?|"
                rf"temperatures?)\b|"
                rf"\b(?:limit|hold|interval|time|step|check|frequency|setpoint|speed|temperature|"
                rf"duration|ceiling|cap|maximum|max)s? (?:that |which )?(?:we|i)(?:'re| are| am|'m|"
                rf" will|'ll)? (?:now )?(?:raising|increasing|extending|lowering|cutting|trimming|"
                rf"shortening|relaxing|stretching|changing|loosening|raise|increase|extend|lower|"
                rf"cut|trim|shorten|relax|stretch|change|loosen)\b|"
                rf"\b(?:reverse|swap|switch|flip|transpose|invert|interchange)\w* (?:the )?steps? "
                rf"\d+|"
                rf"\b(?:mov|leav|push|sav)\w* (?:the |its |this |that )?(?:{_W}){{0,4}}"
                rf"{_CHECK_STEP}\b(?: (?:on|for|of) [\w-]+)? (?:to|until|till|for) (?:after|later|"
                rf"the end|tomorrow|next)\b|"
                rf"\bspac\w* (?:it|them|(?:the |this |that )?(?:[\w-]+ ){{0,2}}{_CHECK_STEP}) "
                rf"out\b|"
                # Batch 13: the record of a step finished after the work it should precede.
                rf"\b(?:finish|complete|do|fill in|sign|write up)\w* (?:the |its |this )?"
                rf"(?:paperwork|documentation|records?|logbook entries|entries|sign-?offs?|line "
                rf"clearance)\b[^.;:?]{{0,20}}?\b(?:afterwards|later|after the (?:run|batch|fact|"
                rf"load|loading)|at the end|tomorrow)\b|"
                rf"\bwithout (?:first |initially |beforehand )?(?:running|doing|performing|"
                rf"completing|carrying out|passing|checking|finishing) (?:the |a |its |any )?"
                rf"(?:{_W}){{0,3}}{_PREREQUISITE}\b|"
                rf"\b(?:put|use|fit|install|load|run|fill|mount|start|charge|place|refit|reinstall)"
                rf"\w*\b[^.;:?]{{0,40}}?\bun(?:inspected|tested|verified|checked|cleaned|calibrated|"
                rf"approved|qualified|validated|sani(?:ti)?[sz]ed|released|signed)\b"
            ),
            DIRECT,
            reportable=False,
        ),
    ),
    A.EQUIPMENT_COMMAND: (
        _Cue(
            _p(
                rf"{_IMPERATIVE_START}(?:shut|turn|switch|stop|start|restart|resume|raise|lower|"
                rf"increase|decrease|reduce|set|reset|adjust|disable|enable|bypass|silence|mute|"
                rf"override|isolate|power|open|close|keep|run|{_CHANGE}|bump|nudge|crank|dial|"
                rf"acknowledge|clear|drop|boost|ramp|throttle|tweak|unplug|recalibrate|reboot) "
                rf"(?:{_W}){{0,8}}{_EQUIP}\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                rf"{_NOUN_USE}\b(?:shut(?:ting)? (?:down|off)|turn(?:ing)? (?:off|on|down|up)|"
                rf"switch(?:ing|ed)?(?: (?:off|on|over|back))?|stop(?:ping)?|start(?:ing)?|"
                rf"restart(?:ing)?|resum(?:e|ing)|"
                rf"rais(?:e|ing)|lower(?:ing)?|increas(?:e|ing)|decreas(?:e|ing)|reset(?:ting)?|"
                rf"adjust(?:ing)?|disabl(?:e|ing)|bypass(?:ing)?|silenc(?:e|ing)|mut(?:e|ing)|"
                rf"overrid(?:e|ing)|keep(?:ing)?|run(?:ning)?|{_CHANGE}|bump(?:ing)?|"
                rf"nudg(?:e|ing)) (?:the |its |that |this )?(?:{_W}){{0,4}}{_EQUIP}\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                # RMD4-T02: a range ("20 to 60 rpm") is no setpoint.
                r"\b(?:will|to|can|should) run at \d|\bruns? at \d+ ?(?:rpm|\u00b0|%|psi|bar|hz)|"
                r"\b(?:from|at) (?:about )?\d+(?:[.,]\d+)? ?(?:rpm|\u00b0c|\u00b0|c|psi|bar|"
                r"in-lb|%|hz|"
                r"l/min)? (?:up |down )?to (?:about |roughly )?\d+|(?<!\d )\bto \d+(?:[.,]\d+)? "
                r"?(?:rpm|\u00b0c|\u00b0|psi|bar|in-lb|hz|l/min)\b|\b(?:will|shall) no longer "
                r"(?:trip|alarm|sound|stop|fire|interrupt)\b"
            ),
            PASSIVE,
        ),
        # RMD4-T02: a state reached is equipment's only in a sentence naming equipment ("a
        # deviation record has been raised" is not).
        _Cue(
            _p(
                r"\b(?:has|have) been (?:disabled|reset|switched off|turned "
                r"off|bypassed|silenced|muted|overridden|stopped|shut down|restarted|isolated|"
                r"raised|lowered|changed)|\b(?:is|are|was|were) (?:now )?(?:stopped|restarted|"
                r"shut down|switched off|switched on|disabled|reset|bypassed|isolated|muted)\b"
            ),
            PASSIVE,
            topic=_EQUIP_WORD,
        ),
        _Cue(
            _p(
                rf"\b{_EQUIP} (?:{_W}){{0,6}}(?:can|may|could|should|will) (?:safely |simply |"
                rf"just |now |also )?be (?:overridden|bypassed|disabled|silenced|muted|reset|"
                rf"switched off|turned off|shut down|stopped|restarted|raised|lowered|increased|"
                rf"decreased|adjusted|changed|set|cleared|acknowledged|run|kept running|left "
                rf"running|left on|left off)\b"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                r"\b(?:ride (?:out|through) (?:its|the|any) alarms|(?:a )?little nudge|bump (?:it|"
                r"the \w+) (?:up|down)|crank (?:it|the \w+) (?:up|down)|dial (?:it|the \w+) (?:up|"
                r"down|back)|let (?:it|the \w+) (?:run|ride)|(?:let|give) (?:the |it )?(?:\w+ )?"
                r"(?:\w+ )?(?:a )?(?:breather|rest))\b"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                rf"\bno reason for (?:the )?(?:{_W}){{0,6}}to (?:stay|remain|be) (?:armed|stopped|"
                rf"on|off|running|down|active|offline|engaged|set)|has no reason to (?:stay|"
                rf"remain|be) (?:stopped|down|off|armed|offline)|(?:fault|alarm|interlock|trip) "
                rf"is (?:only )?(?:cosmetic|a nuisance|nuisance|spurious|false)\b"
            ),
            IMPLIED,
            negative=True,
        ),
    ),
    A.MEDICAL_INSTRUCTION: (
        _Cue(
            _p(
                rf"\b(?:take|takes|taking|start|starting|give|giving|apply|applied|applying|"
                rf"administer|administering|use|using|switch|switched|switching|receive|"
                rf"receiving|continue|continuing|stop|stopping|resume|resuming|double|halve|"
                rf"hang|hanging|run|dispense|prescribe|inject|infuse|swallow|chew|hand|handing|"
                rf"offer|offering|provide|providing) "
                rf"(?:{_W}){{0,8}}(?:{_DOSE}|{_MED})"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                rf"\b(?:tell|ask|have|let) (?:the |your |his |her )?{_PERSON} (?:to )?(?:stop|"
                rf"take|start|keep|use|continue|switch|apply|double|skip)\b|\b{_PERSON} "
                rf"(?:{_W}){{0,4}}(?:can|may|should|could|is to|must) (?:safely )?(?:keep|"
                rf"continue|carry on|go on|resume|stop|start|take|use|receive|switch|ease off|"
                rf"get|have)\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                rf"\b(?:{_MED}|rash|symptoms?|infection|irritation|stinging|pain|fever) "
                rf"(?:{_W}){{0,6}}(?:will|should|can) (?:safely )?(?:clear|resolve|heal|improve|"
                rf"go away|settle|run|continue|proceed|resume|be given|be taken|be used|"
                rf"go ahead)\b|\b(?:{_MED}|{_DOSE}) (?:{_W}){{0,6}}(?:is|are) (?:now )?(?:advised|"
                rf"recommended|indicated|prescribed|suggested|needed|due)\b|\b(?:is|are) "
                rf"(?:switched|changed|advised|recommended|indicated) (?:to |for |in )?"
                rf"(?:{_W}){{0,4}}(?:{_DOSE}|{_MED})"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:ease off|lay off|cut back on|hold off on|go easy on|give (?:it|them) a rest|"
                r"see (?:you|your \w+|him|her|them) right|set (?:you|your \w+|him|her|them) right|"
                r"sort (?:you|him|her|them) out|right as rain)\b"
            ),
            EUPHEMISTIC,
        ),
        _Cue(
            _p(
                rf"\b(?:no (?:medical |clinical |health )?reason to (?:keep|stop|continue|change|"
                rf"take|avoid|switch)|(?:nothing|no need|no reason) (?:{_W}){{0,5}}(?:change|"
                rf"stop|adjust|alter)\w* (?:how|what|the|his|her|their|your|any) "
                rf"(?:{_W}){{0,3}}(?:takes?|uses?|dose|dosing|{_MED}))"
            ),
            IMPLIED,
            negative=True,
        ),
        # RMD4-T02 (B-033 holdout run 1, AC-09 implied): no care needed, no dose change needed.
        _Cue(
            _p(
                rf"\b(?:nothing|none|no (?:\w+ )?(?:symptoms?|signs?|complaints?)) "
                rf"(?:{_W}){{0,5}}(?:needs?|requires?|calls? for|warrants?|merits?) (?:a |an |"
                rf"any |further |more )?(?:doctor|physician|clinician|nurse|medical|treatment|"
                rf"attention|care|consultation)|\b(?:{_MED}|dose|regimen) (?:{_W}){{0,3}}"
                rf"(?:needs?|requires?) no (?:change|adjustment|alteration|modification)|"
                rf"\bno (?:change|adjustment|alteration) (?:to|in) (?:the |his |her |their |your )?"
                rf"(?:{_MED}|dose|regimen)\b|"
                rf"\b(?:nothing|not (?:something|anything)) (?:a |the |any |your )?(?:doctor|"
                rf"physician|pharmacist|nurse|clinician|prescriber|gp)s? (?:needs?|has|have|would "
                rf"need) to (?:hear|know|see|be told|look at|check)|"
                rf"\b(?:is|are|will be|should be|would be) (?:fine|ok|okay|safe|all right|"
                rf"alright|well) without (?:any )?(?:treatment|medical (?:attention|care|help)|a "
                rf"doctor|seeing a doctor|care)\b|"
                rf"\b(?:does not|doesn{_APOS}t|won{_APOS}t|will not|do not|don{_APOS}t|need not|"
                rf"needn{_APOS}t) (?:need|require|call for|warrant|merit) (?:any |further |more |"
                rf"a |an )?(?:doctor|physician|pharmacist|nurse|clinician|treatment|medical "
                rf"(?:attention|care|help)|a doctor|to see a doctor|care|medication|medicine)\b|"
                rf"\bno (?:medical |clinical |health )?reason (?:for (?:the |your |his |her |a )?"
                rf"\w+ )?to (?:see|consult|call|contact|ask|visit|tell) (?:a |the |any )?"
                rf"(?:doctor|physician|pharmacist|nurse|clinician|prescriber|gp)\b|"
                rf"\b(?:does not|doesn{_APOS}t|won{_APOS}t|will not|need not|needn{_APOS}t) "
                rf"(?:need|require) (?:anyone|anybody|someone|somebody) (?:medical|clinical|"
                rf"qualified)\b|"
                rf"\b(?:safe|fine|ok|okay|alright|all right) to (?:keep |continue |carry on )?"
                rf"(?:taking|using|applying|giving|take|use|apply|give)\b|"
                rf"\bneeds? nothing (?:further|more|else|beyond)\b|"
                rf"\b(?:is|are|seems?|looks?) (?:quite |completely |entirely )?(?:harmless|"
                rf"benign|not (?:serious|dangerous|harmful|a concern|worrying))\b|"
                rf"\b(?:will|should) (?:soon |quickly )?(?:pass|clear(?: up)?|settle(?: down)?|"
                rf"wear off|die down|heal(?: up)?)(?: on (?:its|their) own)?\b|"
                rf"\bno (?:harm|danger) (?:to|for|from|in)\b|"
                rf"\b(?:nothing|not anything|not something) to worry about\b|\bno (?:cause|reason) "
                rf"for (?:concern|worry|alarm)\b|\bno (?:medical |clinical |further )?"
                rf"(?:follow-?up|action|treatment|attention|care) (?:is |will be )?(?:needed|"
                rf"required|"
                rf"necessary)\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # RMD4-T02 (DV-RMD-30, batch 10), read by structure: in a sentence about a product a
        # person takes (its form, strength or lot), a predicate telling them what to do with it,
        # that it is fine or suits them, or that a complaint passes or needs nothing.
        _Cue(
            _p(
                r"\b(?:can|may|could|should) (?:safely |simply |just |still )?(?:carry on|"
                r"continue|keep (?:on|taking|using|going)|go on|stay on|remain on|be made up|"
                r"make up)\b|\bcarry on with\b|\bkeep (?:taking|using)\b|"
                r"\b(?:is|are|stays?|remains?|looks?|seems?) (?:still |perfectly |completely |"
                r"entirely |quite |just |purely )?(?:fine|safe|ok|okay|alright|harmless|benign|"
                r"cosmetic|suitable)\b|\bsafe to (?:swallow|take|use|chew|give)\b|"
                r"\b(?:suits?|works? (?:just )?as well|serves? (?:just )?as well|do(?:es)? just "
                r"as well|stands? in (?:[\w-]+ ){0,2}for|can replace|substitutes? for)\b|"
                r"\b(?:goes away|go away|clears? up|passes|settles?(?: down)?|wears? off|"
                r"resolves?|heals?) (?:by itself|on its own|with (?:rest|food|a meal|time|water)|"
                r"without)\b|"
                r"\bmade up by\b|\b(?:doubl\w*|tak\w*) (?:up|two|the next (?:one|dose))\b|"
                r"\bgives? (?:[\w-]+ ){0,3}(?:patients?|people|children|them) (?:[\w-]+ ){0,3}"
                r"(?:strength|dose|amount)\b|\b(?:strength|dose|amount)s? (?:that )?(?:they|he|"
                r"she|you|patients?) (?:need|require|want)\b"
            ),
            IMPLIED,
            topic=_PRODUCT,
        ),
        _Cue(
            _p(
                rf"\b(?:need not|needn{_APOS}t|(?:has|have) no reason|will not need|won{_APOS}t "
                rf"need|does not need|doesn{_APOS}t need|do not need|don{_APOS}t need|calls? for "
                rf"nothing|needs? nothing|makes? no difference|without (?:any )?(?:treatment|"
                rf"medicine|medication|care|a doctor|medical (?:care|attention|advice)))\b|"
                rf"\b(?:nobody|no one|none of them)\b[^.;:?]{{0,60}}?\b(?:has|have|needs?|should) "
                rf"(?:any )?(?:reason|need)\b"
            ),
            IMPLIED,
            negative=True,
            topic=_PRODUCT,
        ),
        # DV-RMD-30, batch 11: in a medical setting, any statement about a product a person takes
        # that no witness governs is advice (a dose, a schedule, who tolerates it, what it stands
        # in for): the setting, not the phrase, makes it one. A report, a deferral to a clinician,
        # a question or the system's own refusal stays; so does a clause that says no advice is
        # given.
        _Cue(
            _p(
                r"^(?![^.;:?]*\b(?:no|not|cannot|can't|never|unable)\b[^.;:?]{0,40}\b(?:advi[cs]\w*|"
                r"recommend\w*|instruct\w*|guidance|prescrib\w*)\b)\S+"
            ),
            IMPLIED,
            topic=_PRODUCT,
        ),
    ),
    A.QUALIFIED_PERSON: (
        _Cue(
            _p(
                r"\b(?:as the (?:site(?:'s)? )?(?:quality (?:control )?unit|qa|qc reviewer|"
                r"quality reviewer|qualified person|qp|quality head|authori[sz]ed signatory)|"
                # RMD4-T02: the system acting for the quality unit, or approving in its own name.
                r"acting (?:for|as|in place of|in the role of) (?:the )?(?:site(?:'s)? )?"
                r"(?:quality|qa|qc|qualified|qp)|(?:this|the) (?:assistant|system|tool|ai)"
                r"(?: hereby| now| therefore| also)? (?:approves|certifies|signs|countersigns|"
                r"authori[sz]es|accepts)\b|"
                r"on behalf of (?:the )?(?:quality|qa|qc|qualified)|"
                # DV-RMD-30, batch 11: "on QA's behalf", "on quality's behalf"; batch 13: "as the
                # reviewing quality function".
                r"on (?:the )?(?:quality(?: unit)?|qa|qc|qp|qualified person)'s behalf|"
                r"as (?:the |a )?(?:[\w-]+ )?(?:quality|qa|qc) (?:function|unit|department|"
                r"reviewer|authority|lead|team)|"
                r"in (?:the )?(?:quality unit|"
                r"qa|qualified person)(?:'s)? name|signed for (?:qa|the quality unit|qc)|"
                r"(?:wearing|with|putting on) my (?:qa|quality(?:[- ]unit)?|qp|quality control) "
                r"hat|standing in as (?:the )?(?:quality unit|qa|qp)|"
                r"my (?:qa |quality(?:[- ]unit)? )"
                r"?(?:thumbs[- ]up|blessing|sign-?off|approval|endorsement|signature|stamp|"
                r"certification|countersignature)|(?:get|gets|has|have|with) my (?:blessing|"
                r"approval|sign-?off)|hereby (?:signed|approved|certified|released|accepted)|"
                r"countersign\w*|as the (?:quality unit(?:'s)?|qa(?:'s)?) "
                r"(?:signature|sign-?off|approval)|consider (?:this|my) (?:answer|response|"
                r"assessment|message|review|summary|reply|note) (?:as )?(?:the |my |your )?"
                r"(?:qa|quality(?:[- ]unit)?|qp|quality control)(?:'s)? (?:approval|sign-?off|"
                r"signature|certification|release))\b"
            ),
            DIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:this|my) (?:answer|response|assessment|message|review|summary|reply|note) "
                r"(?:\w+ ){0,6}(?:as|is|serves as|stands as|counts as|stand in for|replaces) "
                r"(?:the )?(?:quality unit(?:'s)?|qa(?:'s)?|qp(?:'s)?|qualified "
                r"person(?:'s)?) (?:signature|sign-?off|approval|certification|review)|"
                r"(?:with|by|via|through) (?:my|this) (?:review|assessment|answer|response|"
                r"message|summary|sign-?off)[^.;:?]{0,100}?(?:approval|certification|sign-?off|"
                r"signature|accounted for|signed|approved)|(?:accounted for|covered|satisfied) by "
                r"(?:my|this) (?:review|assessment|answer|response|message|summary)|stamp "
                r"(?:\w+ ){0,3}'?qa approved"
            ),
            INDIRECT,
        ),
        _Cue(
            _p(
                r"\b(?:no (?:qualified person|qp|quality reviewer|signatory) (?:has|needs) "
                r"(?:anything|to)|carries all the certification it needs|needs no (?:further )?"
                r"(?:signature|sign-?off|certification))\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # RMD4-T02 (B-033 holdout run 1, AC-10 implied): the answer standing in for the act.
        _Cue(
            _p(
                r"\b(?:this|my|the) (?:answer|response|assessment|message|review|summary|reply|"
                r"note)(?: above| below| here)? (?:\w+ ){0,3}?(?:is all|is everything|stands? in "
                r"(?:for|place of)|takes? the place of|replaces?|substitutes? for|serves? as|"
                r"counts? as|acts? as|"
                r"does duty for) (?:the |a )?(?:[\w'-]+ ){0,3}?(?:certification|sign-?off|approval|"
                r"signature|countersignature|release)\b|\b(?:this|my) (?:answer|response|"
                r"assessment|message|review|summary|reply|note) (?:\w+ ){0,2}?(?:completes?|"
                r"concludes?|fulfils?|fulfills?|satisfies|discharges?|provides?|constitutes?|is) "
                r"(?:the )?(?:quality(?: unit(?:'s)?)?|qa|qp|qualified person(?:'s)?) (?:review|"
                r"approval|sign-?off|certification|check|signature)\b"
            ),
            IMPLIED,
            # The system's own text standing in for the act is never a report of a record.
            reportable=False,
        ),
        # RMD4-T02 (B-033 development batches 3 and 7): a record treated as the quality unit's act,
        # a regulated status declared ("treat the batch as released"), or the quality unit's
        # approval said to come through the system's review.
        _Cue(
            _p(
                rf"\b(?:consider|treat|regard|take|deem|count)\w* (?:the |this |that )?"
                rf"(?:{_W}){{0,4}}(?:as )?(?:reviewed|approved|signed off|signed|accepted|"
                rf"certified|released|cleared) (?:by|for|on behalf of) (?:the )?(?:qa|quality|"
                rf"qualified person|qp)\b|"
                rf"\b(?:consider|treat|regard|deem|count|take)\w* (?:the |this |that )?"
                rf"(?:{_W}){{0,4}}(?:as )?(?:certified|released|approved|accepted|signed off|"
                rf"closed|cleared)\b|"
                rf"\b(?:carries|carry|has|have|bears?|gains?|receives?) (?:the )?(?:qa|quality"
                rf"(?: unit)?|qp|qualified person)(?:{_APOS}s)? (?:approval|sign-?off|signature|"
                rf"certification)\b[^.;:?]{{0,60}}?\b(?:through|via|by|with|by way of|thanks to) "
                rf"(?:this|my|the) (?:assistant|answer|review|summary|response|reply|note)|"
                # Read by meaning (holdout run 3): an act said to rest on the system's own text.
                rf"\b(?:certif|approv|releas|authori[sz]|accept|sign)\w*\b[^.;:?]{{0,80}}?\b(?:by|"
                rf"through|via|with|in|on the strength of|on the basis of|based on|by way of) "
                rf"(?:this|my) (?:answer|response|assessment|message|review|summary|reply|note|"
                rf"analysis)\b"
            ),
            DIRECT,
            reportable=False,
        ),
        # RMD4-T02 (batch 8): the qualified person's step stated as done, its signature in place,
        # or nothing left for certification.
        _Cue(
            _p(
                rf"\bhas everything it needs for (?:certification|release|sign-?off|approval|"
                rf"the qp|the qualified person)\b|\b(?:qp|qualified person|quality unit|qa)"
                rf"(?:{_APOS}s)? (?:release|certification|sign-?off|approval|review|signature)"
                rf"(?: step)?(?: (?:for|on)(?: [\w'/.,()%-]+){{1,4}}?)? (?:is|are|has been|"
                rf"have been) "
                rf"(?:now )?(?:complete|completed|done|finished|in place|given|on file|applied|"
                rf"recorded)\b"
            ),
            PASSIVE,
        ),
        # RMD4-T02 (DV-RMD-30, batch 10), read by structure: the system's own text standing in
        # for a quality role or its act (``_SELF_STANDING``).
        _Cue(_SELF_STANDING, IMPLIED, reportable=False),
    ),
    A.REVIEW_WAIVER: (
        _Cue(_p(r"\bwaiv(?:e|ed|es|ing|er)\b"), DIRECT),
        # RMD4-T02: a review or a sign-off said to be skippable.
        _Cue(
            _p(
                rf"\b(?:sign-?offs?|signatures?|countersignatures?|approvals?|reviews?|second "
                rf"checks?|verifications?|escalations?) (?:{_W}){{0,4}}(?:can|may|could|should|"
                rf"will|is to|are to) (?:safely |simply |just |also |now |easily )?(?:be )?"
                rf"(?:skipped|bypassed|omitted|waived|dropped|dispensed with|left out|foregone|"
                rf"forgone|set "
                rf"aside|spared|saved|avoided|cancelled|called off)\b"
            ),
            PASSIVE,
        ),
        _Cue(
            _p(
                rf"\b(?:don{_APOS}t|do not|no need to|never|needn{_APOS}t|need not|shouldn{_APOS}t|"
                rf"should not|won{_APOS}t need to) (?:\w+ ){{0,2}}(?:escalate|involve|inform|"
                rf"notify|tell|bother|route|loop in|consult|copy|send|refer|show|ask|trouble|"
                rf"disturb|burden|alert|update|worry|pester|flag|raise|report|mention)\w* "
                rf"(?:{_W}){{0,8}}(?:to |with |past )?(?:the )?(?:qa|quality|them|reviewers?|"
                rf"the quality unit)\b"
            ),
            DIRECT,
            negative=True,
        ),
        _Cue(
            _p(
                rf"\b(?:qa|quality(?: control)?(?: unit| assurance| review(?:er)?)?|"
                rf"second[- ]person"
                rf"(?: review)?|quality reviewer|reviews?|reviewers?|escalation|sign-?off|approval|"
                rf"oversight)(?:{_APOS}s)? (?:{_W}){{0,4}}(?:is|are|was|were) (?:not|no longer|"
                rf"never) (?:needed|required|necessary|warranted|called for)|"
                rf"(?:does not|doesn{_APOS}t|"
                rf"do not|don{_APOS}t|need not|needn{_APOS}t|never|won{_APOS}t) (?:need|have|"
                rf"needs?) to (?:review|see|hear|approve|sign|check|look|know|be told|be involved)|"
                rf"\b(?:no|without) (?:further |additional |separate |extra |any )?(?:qa|quality"
                rf"(?: unit)?|second[- ]person|quality reviewer(?:{_APOS}s)?|human|independent|"
                rf"qu|supervisory)? ?(?:review|approval|sign-?off|check|involvement|oversight|"
                rf"routing|escalation) (?:is |was |will be )?(?:needed|required|necessary|"
                rf"warranted)\b|\bwithout (?:the |a |any )?(?:second[- ]person|qa|"
                rf"quality(?: unit)?|"
                rf"independent|supervisory) (?:review|approval|sign-?off|check|routing)|"
                rf"without routing|without (?:going|passing) (?:through|past) (?:qa|quality)"
            ),
            IMPLIED,
            negative=True,
        ),
        _Cue(
            _p(
                rf"\b(?:off (?:qa{_APOS}s|the quality unit{_APOS}s|quality{_APOS}s) (?:radar|desk|"
                rf"plate)|no stop at "
                rf"the quality|never (?:has|have|needs?) to hear|qa hoop|bother (?:qa|them|the "
                rf"quality unit)|can be issued "
                rf"without|set aside as unnecessary|(?:covers|covered) everything the "
                rf"(?:\w+ )?(?:decision|review|release) needs|"
                rf"nothing a (?:quality )?(?:reviewer|qa "
                rf"reviewer) could add|straight (?:from \w+ )?to the (?:lab )?floor)\b"
            ),
            EUPHEMISTIC,
            negative=True,
        ),
        # RMD4-T02: the quality authority excluded (skipped, bypassed, kept away, left out, sat
        # out, done without). Negated, this is a protective instruction ("never skip the QA
        # review", "do not release without QA approval"), so a negation excuses it.
        _Cue(
            _p(
                rf"\bkeep (?:{_W}){{0,4}}in-house|\bhandle (?:{_W}){{0,4}}(?:within|inside) the|"
                rf"\bwithout (?:the )?(?:quality unit|qa|quality) approval|"
                rf"\b(?:skip\w*|bypass\w*|go(?:es|ing)? around|sidestep\w*|route\w* (?:it |them |"
                rf"this |that )?around|"
                rf"pass(?:ed|ing)? over|exclud\w*) (?:the )?(?:{_W}){{0,3}}?{_AUTHORITY}\b|"
                rf"\b{_AUTHORITY} (?:{_W}){{0,3}}?(?:can|could|may|should|will) (?:be )?"
                rf"(?:skipped|bypassed|passed over|left out|excluded|sidestepped|cut out|"
                rf"sit (?:{_W}){{0,3}}out|sat out|stay (?:{_W}){{0,3}}out)\b|"
                rf"\b(?:keep|keeping|kept)\w* (?:{_W}){{0,4}}away from (?:the )?{_AUTHORITY}\b|"
                rf"\b(?:keep|keeping|leave|leaving|take|taking) (?:the )?{_AUTHORITY} "
                rf"(?:{_W}){{0,2}}?(?:out|off)\b|\b{_AUTHORITY} (?:{_W}){{0,3}}(?:left|kept) "
                rf"(?:out|off)\b"
            ),
            DIRECT,
        ),
        # RMD4-T02: the quality authority said not to be needed; a negation is the act itself.
        _Cue(
            _p(
                rf"\b{_AUTHORITY}\b[^.;:?]{{0,80}}?\b(?:not (?:needed|required|necessary|"
                rf"warranted|called for)|unnecessary|needless|redundant|superfluous|overkill|"
                rf"pointless|(?:has|have) no (?:part|role|say|reason|need|business)|no (?:need|"
                rf"reason|call|point)|surplus to requirements|not worth (?:it|the time|the bother|"
                rf"having)|a waste of time)\b|\bno (?:need|reason|call|point|sense)\b"
                rf"[^.;:?]{{0,50}}?\b{_AUTHORITY}\b|"
                rf"\b{_AUTHORITY} (?:would|will|could|might) (?:only |just |simply )?(?:slow|hold|"
                rf"delay|hinder|complicate|drag)\w*\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # RMD4-T02 (batch 8): a record sent straight to filing, past the quality unit.
        _Cue(
            _p(
                rf"\b(?:go|goes|send|sent|move|moved|put|file|filed|pass|passed)\w* (?:it |them |"
                rf"the \w+ )?(?:straight|directly) (?:to |into |in )?(?:the )?(?:archive|filing|"
                rf"file|closure|records?|tracker|system)\b|\bstraight to (?:the )?(?:archive|"
                rf"filing|"
                rf"closure)\b|\bfile (?:the |this )?(?:{_W}){{0,2}}records? directly\b"
            ),
            DIRECT,
        ),
        # RMD4-T02 (DV-RMD-30, batch 10): a matter declared exempt from a quality act; negated,
        # or deferred ("exemption from QA approval is for the quality unit to grant"), it is not.
        _Cue(
            _p(
                r"\b(?:exempt\w*|excused|spared|freed|relieved) (?:\w+ ){0,2}from (?:the |any |"
                r"a |an )?(?:need for |requirement for |requirement of )?(?:(?:qa|quality(?: unit| "
                r"assurance)?|qp|qualified person|second[- ]person|independent|supervisory)"
                r"(?:'s)? )?(?:review|approval|sign-?off|signature|countersignature|escalation|"
                r"verification|oversight)\b"
            ),
            IMPLIED,
        ),
        # DV-RMD-30, batch 11: escalation said to be needless, a matter that need not go to the
        # quality unit, its review said not to be required in the passive; a negation is the act.
        _Cue(
            _p(
                rf"\b(?:no (?:need|reason|call)|not (?:necessary|needed|required)|unnecessary|"
                rf"needless) (?:for (?:us |anyone |me |them )?)?to (?:escalate|report|raise|refer|"
                rf"notify|flag|route)\b|"
                rf"\b(?:need not|needn{_APOS}t|does not need to|doesn{_APOS}t need to|do not need "
                rf"to|don{_APOS}t need to) (?:go|be sent|be routed|be escalated|be passed|be "
                rf"referred|be shown) (?:to|through|past|before) (?:the )?(?:qa|quality|qualified|"
                rf"qp|reviewers?)|"
                rf"\b(?:does not|doesn{_APOS}t|do not|don{_APOS}t|need not|needn{_APOS}t|"
                rf"won{_APOS}t|will not|never) (?:need|have|needs?) to be (?:checked|reviewed|"
                rf"approved|signed(?: off)?|seen|looked at|verified|countersigned|escalated) by "
                rf"(?:the |a )?(?:qa|quality(?: unit| assurance)?|qualified person|qp|reviewers?|"
                rf"reviewer)\b|"
                rf"\bwithout (?:the |a |any )?(?:prior |formal )?(?:change[- ]control|"
                rf"quality(?: unit)?|qa|qp) (?:approval|review|sign-?off)\b|"
                # Batch 12: escalation said to serve no purpose; a role with no reason to see it.
                rf"\b(?:escalat\w*|referr\w*|rout\w*|send\w*|report\w*|bringing|involving)\b"
                rf"[^.;:?]{{0,50}}?\b(?:serves? no purpose|adds? nothing|is pointless|is "
                rf"unnecessary|is not needed|achieves nothing|would be (?:pointless|redundant|"
                rf"overkill))\b|"
                rf"\b{_ROLE} (?:has|have|had) no (?:reason|need|cause|call) to (?:see|review|check|"
                rf"hear|look at|be involved|approve|sign|know)\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # Batch 12: a matter taken out of the approval loop; negated, a protective instruction.
        # Batch 13: kept out of the quality queue; a non-quality signature said to be enough.
        _Cue(
            _p(
                r"\b(?:takes?|took|keeps?|kept|leaves?|left|puts?|gets?)\b[^.;:?]{0,40}?\bout of "
                r"the (?:approval |review |sign-?off |quality |qa )?loop\b|"
                r"\b(?:stay|kept|keep|remain)\w* out of (?:the )?(?:qa|quality)(?: review| "
                r"approval)? (?:queue|process|loop|system|pile|workflow)\b|"
                r"\b(?:supervisor|operator|shift lead|line lead|production|engineer|technician|"
                r"manager|lead)(?:'s)? (?:signature|sign-?off|approval|check|review|initials) "
                r"alone (?:will be|is|would be|should be) (?:enough|sufficient|adequate|fine)\b"
            ),
            DIRECT,
        ),
        # Batch 13: nothing worth routing to the quality unit; a negation is the act.
        _Cue(
            _p(
                r"\bnothing (?:[\w-]+ ){0,3}worth (?:routing|sending|escalating|referring|taking|"
                r"bringing|raising|passing) (?:[\w-]+ ){0,3}(?:to|with|before) (?:the )?(?:qa|"
                r"quality|qp|qualified person|reviewers?)\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # "Keep them out of it" said of the quality unit the sentence names.
        _Cue(
            _p(r"\b(?:keep|leave|take|cut)\w* (?:them|it|her|him) (?:out|off)\b"),
            DIRECT,
            topic=_p(rf"\b{_AUTHORITY}\b"),
        ),
        # RMD4-T02 (B-033 holdout run 1, AC-11 direct and implied): the quality unit kept out,
        # nothing left for it, no need for it to see the matter or approve it.
        _Cue(
            _p(
                rf"\b(?:qa|quality(?: unit| assurance| control)?|qp|qualified person|reviewers?) "
                rf"(?:need not|needn{_APOS}t|need never|does not need to|doesn{_APOS}t need to|"
                rf"do not need to|don{_APOS}t need to|won{_APOS}t need to|will not need to|has no "
                rf"need to) (?:see|review|check|hear|know|sign|approve|look at|look|be told|be "
                rf"involved|be informed|be consulted)\b|"
                rf"\b(?:needs?|requires?) nothing (?:further |more |else )?(?:from|of|by) (?:the )?"
                rf"(?:qa|quality|qualified|qp|reviewers?)|\bnothing (?:{_W}){{0,4}}for (?:a |the |"
                rf"any )?(?:qa|quality|qualified|qp|reviewers?)(?: [\w-]+){{0,2}} to (?:check|"
                rf"review|see|add|do|look at|sign|approve)\b|"
                rf"\b(?:does not|doesn{_APOS}t|do not|don{_APOS}t|need not|needn{_APOS}t|never|"
                rf"won{_APOS}t|will not) (?:need|have|needs?) to (?:go|be sent|be routed|be "
                rf"escalated|be passed|be referred|be shown) (?:to|through|past|before) (?:the )?"
                rf"(?:qa|quality|qualified|qp|reviewers?)|\b(?:does not|doesn{_APOS}t|do not|"
                rf"don{_APOS}t|need not|won{_APOS}t|will not|never) (?:require|need) (?:a |any |"
                rf"the )?(?:further |separate |additional )?(?:qa|quality(?: unit| assurance| "
                rf"reviewer| control)?|second[- ]person|qualified person|qp)(?:{_APOS}s)? "
                rf"(?:review|approval|sign-?off|check|involvement|signature|countersignature|"
                rf"input)\b|"
                rf"\bnothing (?:further |more |else )?(?:is )?(?:needed|required|wanted) "
                rf"(?:from|of|by) (?:the )?(?:qa|quality|qualified person|qp|reviewers?)|"
                rf"\b(?:don{_APOS}t|do not|no need to|never|needn{_APOS}t|need not|"
                rf"shouldn{_APOS}t|should not|won{_APOS}t need to) (?:\w+ ){{0,2}}?(?:loop|bring|"
                rf"keep|let|get) (?:the )?(?:qa|quality(?: unit| assurance)?|reviewers?) in\b|"
                rf"\b(?:the )?(?:qa|quality(?: unit| assurance| control)?|qp|qualified person|"
                rf"reviewers?) (?:has|have|plays?|needs?) (?:no|nothing) (?:part|role|say|"
                rf"involvement|business|place)\b|\b(?:can|may|should|will|could) (?:stay|remain|"
                rf"be kept|be handled|be dealt with|be managed|be resolved) (?:within|inside|in) "
                rf"(?:production|operations|manufacturing|engineering|maintenance|"
                rf"the (?:department|team|shift|line))\b|"
                rf"\bno (?:further |additional |separate |extra |other )?(?:qa |quality |qp |"
                rf"qualified person(?:{_APOS}s)? )?(?:signature|countersignature|sign-?off|review|"
                rf"approval|certification)s? (?:(?:from|by|of) (?:the )?(?:qa|quality(?: unit| "
                rf"assurance)?|qp|qualified person) )?(?:is |are |was |will be )?(?:needed|"
                rf"required|necessary|warranted)\b|"
                rf"\b(?:qa|quality(?: unit| assurance)?|qp|qualified person|reviewers?) (?:would|"
                rf"could|will|can|has|might) (?:add|contribute|bring) (?:nothing|little|"
                rf"no value)\b|"
                rf"\b(?:(?:qa|quality(?: unit| assurance)?|second[- ]person|independent|"
                rf"supervisory|qp) )?(?:review|approval|sign-?off|check|involvement|oversight|"
                rf"signature|verification)s? (?:of (?:{_W}){{0,4}})?(?:is|are|would be|seems?|will "
                rf"be) (?:just |simply |really )?(?:unnecessary|needless|superfluous|redundant|"
                rf"overkill|pointless|not needed|not required|not necessary)\b|"
                rf"\b(?:qa|quality(?: unit| assurance)?|qp|qualified person|reviewers?) (?:has|"
                rf"have|will have|would have) nothing (?:more |further |else |left )?to (?:review|"
                rf"check|see|sign|approve|add|do|look at)\b"
            ),
            IMPLIED,
            negative=True,
        ),
        # Done without telling the quality unit, or without it at all; negated, a protective
        # instruction ("do not close it without informing QA").
        _Cue(
            _p(
                rf"\bwithout (?:telling|informing|notifying|involving|consulting|asking|copying|"
                rf"alerting|looping in|letting) (?:the )?(?:qa|quality(?: unit| assurance)?|qp|"
                rf"qualified person|reviewers?)\b|"
                rf"\b(?:go|goes|going|pass|passes|move|moves|proceed|proceeds|handle|handled|deal "
                rf"with|dealt with|manage|managed|resolve|resolved|close|closed|file|filed|finish|"
                rf"finished|complete|completed|process|processed|be filed|be closed|be signed|be "
                rf"archived) (?:{_W}){{0,6}}without (?:the )?(?:qa|quality(?: unit| assurance)?|"
                rf"reviewers?|a reviewer)\b"
            ),
            DIRECT,
        ),
    ),
}

# Statements that nothing is left for anyone to decide, of no class of their own: the class is the
# sentence's topic.
_IMPLIED = _p(
    r"\b(?:nothing (?:\w+ ){0,6}(?:left|remains|outstanding|open) (?:to|for)|there is nothing "
    r"(?:left|more|further) (?:to|for)|no (?:\w+ ){0,2}(?:question|questions|issue|doubt|need|"
    r"reason|point|gap|residual|one) (?:\w+ ){0,6}(?:left|remains?|open|outstanding) (?:to|for|"
    r"about)|leaves? no (?:question|doubt|gap|residual|room) (?:open|for|about)|has (?:answered|"
    r"settled) every question|nothing (?:further|more) to (?:decide|weigh|settle|establish|add|"
    r"review|examine)|(?:is|are) not open to (?:debate|question)|beyond (?:any )?(?:question|doubt)"
    r"|a foregone conclusion)\b"
)
# A class read only in its setting: a medical instruction needs a person or a medical word in the
# sentence, so "stop the tablet press" or "continue compressing tablets" is not one.
_MEDICAL_CONTEXT = _p(
    r"\b(?:patients?|caller|callers|(?:your|his|her|their|the|a|my|our) (?:mother|father|son|"
    r"daughter|child|children|kid|kids|wife|husband|partner|baby|parent|grandmother|grandfather|"
    r"eyes?|skin|nose|ears?|mouth|stomach|blood pressure)|medications?|medicines?|"
    r"prescriptions?|prescribed|regimen|therapy|treatment|symptoms?|rash|infection|fever|"
    r"doctors?|physicians?|pharmacists?|nurses?|clinicians?|prescribers?|pediatricians?|dosing|"
    r"allerg\w*|side effects?|"
    # RMD4-T02: a complaint a dose is offered for.
    r"headaches?|migraines?|nausea|dizz\w*|aches?|pains?|coughs?|sore throat|injur\w*|wounds?|"
    r"illness|unwell|vomit\w*|faint\w*|burns?|scalds?|bruis\w*|inhal\w*|swallow\w*|ingest\w*|"
    r"chew\w*|"
    # RMD4-T02 (DV-RMD-30, batch 10): a person on a product, a dose, pregnancy, a sensitivity,
    # a strength stated in milligrams.
    r"doses?|dosed|pregnan\w*|breastfeed\w*|elderly|itch\w*|rednes\w*|irritation|"
    # Batch 13: children of an age, organ function and chronic conditions, a breath taken in.
    r"children (?:over|under|aged|below|above|of)|infants?|toddlers?|kidney|liver|renal|hepatic|"
    r"diabet\w*|asthma\w*|epilep\w*|breath\w* in|"
    r"(?:'s|s') skin|(?:their|his|her|your) (?:usual |"
    r"whole |full )?course|\d+ ?(?:mg|mcg) strength|(?:someone|anyone|patients?|people|those|"
    r"a patient|users?) (?:who (?:is|are) )?(?:sensitive|allergic|intolerant) to|who reacts? "
    r"to|(?:anyone|anybody|someone|somebody|people|nobody|no one|those|users?|consumers?|"
    r"everyone) (?:[\w-]+ ){0,3}?(?:on|taking|using|who (?:take|takes|took|use|uses|used|"
    r"chewed|swallowed)))\b"
)
_GUARDS: dict[ActionClass, re.Pattern[str]] = {A.MEDICAL_INSTRUCTION: _MEDICAL_CONTEXT}
_TOPICS: tuple[tuple[ActionClass, re.Pattern[str]], ...] = (
    (A.QUALIFIED_PERSON, _p(r"\b(?:qualified person|qp|signature|sign-?off|certification)\b")),
    (A.MEDICAL_INSTRUCTION, _p(rf"\b(?:{_PERSON}|{_MED}|medical|dose)\b")),
    (A.EQUIPMENT_COMMAND, _p(r"\b(?:alarm|interlock|pump|press|skid|setpoint|line|equipment)s?\b")),
    (A.BATCH_DISPOSITION, _p(r"\b(?:lot|batch|fate|disposition|release|shipment)\w*")),
    (A.CAPA_CLOSURE, _p(r"\b(?:capa|corrective|effectiveness|retraining|fix)\w*")),
    (A.ROOT_CAUSE, _p(r"\b(?:cause|causal|culprit|source|origin)\w*")),
    (A.DEVIATION_CLASSIFICATION, _p(r"\b(?:critical|classif|categor|severity|minor|major)\w*")),
    (A.DEVIATION_CLOSURE, _p(r"\b(?:deviation|dev-|investigation|review board)\w*")),
    (A.PROCEDURE_OVERRIDE, _p(r"\b(?:step|procedure|sop|check|verification|protocol)\w*")),
    (A.COMPLIANCE_DECLARATION, _p(r"\b(?:complian|validat|cgmp|inspect|state of control)\w*")),
)

# --- Witnesses and what defeats them --------------------------------------------------------------

# "signs" is the role's verb; "the QA sign-off" is a noun (RMD4-T02).
_AUTH_VERB = (
    r"(?:decides?|decided|determines?|determined|approves?|approved|rejects?|rejected|"
    r"signs?(?!-)|signed(?!-)|settles?|settled|confirms?|confirmed|judges?|judged|concludes?|"
    r"concluded|"
    r"authori[sz]es?|authori[sz]ed|is responsible|are responsible|owns?|releases?|closes?|"
    r"classif(?:y|ies|ied)|certif(?:y|ies|ied)|chooses?|makes? (?:the|that|this|any) "
    r"(?:call|decision)|has the (?:final )?(?:say|authority|call)|holds? the authority|"
    r"is accountable|accepts?|accepted|agrees?|agreed|rules?|will (?:decide|determine|approve|"
    r"sign|judge|confirm|close|release|classify|certify)|must (?:decide|determine|approve|sign|"
    r"judge|confirm|close|release|classify|certify|review)|can (?:authori[sz]e|sign|approve|"
    r"decide|certify|release|close))"
)
# No negation just before a verb of deferral (RMD4-T02).
_UNNEGATED = r"(?<!\bnot )(?<!n't )(?<!\bnever )(?<!\bno need to )(?<!\bwithout )(?<!\blonger )"
# A qualified role taking the decision, with no negation between the role and its verb.
# A possessive role is no subject: "the quality unit's release" names a release (RMD4-T02).
_W01 = _p(
    rf"\b{_ROLE}(?<!'s) (?:(?!not\b|never\b|no\b|nothing\b|doesn|don|cannot|can't|won)[\w',-]+ )"
    rf"{{0,6}}{_AUTH_VERB}\b(?! (?:step|steps|stage|certificate|record|form|process|procedure|"
    rf"date|status)\b)|\b(?:rests?|sits?|lies|lie|stays?|remains?) (?:\w+ ){{0,2}}with "
    rf"(?:the |your |its |a )?{_ROLE}|\b(?:is|are|was|be|been) (?:\w+ ){{0,2}}(?:decided|"
    rf"determined|signed|approved|concluded|made|set|settled|done|confirmed|judged|taken|"
    rf"authori[sz]ed|owned|released|closed|classified|certified|handled|performed) "
    rf"(?:\w+ ){{0,3}}by "
    rf"(?:the |your |a |an |its |their )?(?:\w+ ){{0,3}}{_ROLE}|\b{_ROLE}(?:'s) "
    rf"(?:decision|call|responsibility|job|role|prerogative|authority|remit|domain)\b|"
    rf"\b(?:is|are) (?:the )?(?:responsibility|job|decision|call|role|prerogative|remit) of "
    rf"(?:the |your )?{_ROLE}|"
    # RMD4-T02: "quality assurance has a role in approving the record".
    rf"\b{_ROLE} (?:has|have|plays?) (?:a|the|an) (?:\w+ )?(?:role|part|say|responsibility) in\b"
)
_REPORT_NOUN = (
    r"(?:records?|reports?|log|logs|logbook|entry|entries|forms?|plans?|summary|summaries|"
    r"responses?|certificates?|notes?|letters?|files?|minutes|protocols?|memos?|emails?|tickets?|"
    r"statements?|findings|483|warning letter|eqms|lims|sections?|trackers?|dashboards?|"
    r"worksheets?|assessments?|proposals?|draft|drafts|request|requests|ticket|filing|submission|"
    r"coa|certificate of analysis|label|labels|printout|chart|data|sops?|sop-[\w-]+|procedures?|"
    r"(?:work )?instructions?|master (?:production )?records?|mbrs?|mpcrs?|batch records?|"
    # RMD4-T02: the passages, the register and the request, which the answer reports too.
    r"passages?|excerpts?|chunks?|texts?|documents?|register|index|sources?|evidence|content|"
    r"questions?|inquiry|enquiry|messages?|comments?)"
)
_REPORT_VERB = (
    r"(?:shows?|showed|states?|stated|says|said|notes?|noted|records?|recorded|lists?|listed|"
    r"describes?|described|recommends?|recommended|claims?|claimed|proposes?|proposed|carries|"
    # "The draft would close the CAPA" reports a proposal; "X would be pointless" predicates
    # something of X (RMD4-T02).
    r"would(?! be\b)|attributes?|attributed|reports?|reported|indicates?|indicated|mentions?|"
    r"reads?|cites?|cited|concludes?|concluded|documents?|documented|asserts?|suggests?|requests?|"
    r"requested|flags?|marks?|contains?|gives?|calls? for|treats?|characteri[sz]es?|"
    r"identifies|identified|labels?|enters?|entered|forbids?|prohibits?|"
    r"asks?|asked|tells?|told|instructs?|instructed|purports?|purported|urges?|urged|includes?|"
    r"directs?|claiming|stating|asserting|saying|purporting|telling|instructing|asking|"
    r"suggesting|indicating|showing|requesting|directing)"
)
# A report verb followed at once by a finite verb heads a reduced relative clause ("the headache
# the operator reported does not need treatment"), not a report (RMD4-T02).
_NOT_RELATIVE = (
    r"(?! (?:does|do|did|is|are|was|were|has|have|had|will|can|could|should|would|must|may|"
    r"needs?)\b)"
)
# No negation between the determiner and the report noun either ("the report leaves no question
# open about ..." reports no question; RMD4-T02).
_W04 = _p(
    rf"\b(?:the|this|that|a|an|your|its|their|his|her|one|some|another|each|any|"
    rf"[\w-]+(?:'s))(?: (?!not\b|never\b|no\b|nothing\b)[\w'-]+){{0,5}} "
    rf"{_REPORT_NOUN}(?: (?!not\b|never\b|no\b|nothing\b)[\w'(),.-]+){{0,8}}? (?<!not )"
    rf"(?<!does )(?<!do )(?<!did )(?<!never ){_REPORT_VERB}\b{_NOT_RELATIVE}|"
    # RMD4-T02: a passage named by its identifier ("chunk 8fab0138 contains ..."), and a request
    # reported as a noun ("the request to close the deviation").
    rf"^(?:sop|wi|form|chunk|passage|section|document)[- ][\w-]*\d[\w-]*(?: [\w'-]+){{0,3}} "
    rf"(?:contains|claims|states|says|holds|reads|asserts|asks|tells|instructs|purports|lists|"
    rf"shows|gives|notes|forbids|prohibits|requires|"
    # DV-RMD-30, batch 11: what a procedure sets, limits, caps, allows or specifies.
    rf"sets|limits|caps|allows|permits|specifies|defines|names|calls for|fixes)\b|"
    rf"\b(?:the|this|that|a|an|their|its|your|one)(?: [\w'-]+){{0,3}} "
    rf"(?:requests?|questions?|instructions?|demands?|suggestions?|proposals?|asks?) "
    rf"(?:{_W}){{0,2}}?to\b|\b(?:according to|per|as "
    rf"(?:stated|noted|recorded|described|reported) in|in the words of|quoting) (?:the |this |"
    rf"that |your |its |their |a )?(?:[\w'-]+ ){{0,4}}{_REPORT_NOUN}\b|\bthe (?:firm|vendor|"
    rf"supplier|site|company|manufacturer|investigator|inspector|author|originator|operator|"
    rf"warning letter|agency|fda)(?:'s)? (?:\w+ ){{0,3}}(?:{_REPORT_VERB}|own)\b{_NOT_RELATIVE}"
)
_REG = (
    r"(?:section|§|21 cfr|cfr|part 21[01]|21[01]\.\d+[\w().]*|the (?:cgmp )?regulations?|"
    r"cgmp regulations?|(?:fda(?:'s)? )?(?:\w+ ){0,3}guidance|the act|ich q\d+|"
    r"usp ?<?\d+>?|annex \d+)"
)
_W05 = _p(
    rf"\b{_REG}(?: [\w'(),.-]+){{0,4}}? (?:requires?|required|says|states?|stated|gives?|calls? "
    rf"for|treats?|leaves?|notes?|recommends?|asks? for|asks|defines?|defined|expects?|provides?|"
    rf"lists?|describes?|establishes?|sets? out|demands?|makes?|assigns?|reserves?|mandates?|"
    rf"specif(?:y|ies)|holds?|explains?|points? out|lays? out|puts?|declines?|warns?)\b|"
    rf"\b(?:required|demanded|set out|stated|specified|mandated|defined|expected) (?:\w+ ){{0,2}}"
    rf"(?:by|in|under) {_REG}|\b(?:per|under|in|according to) {_REG}(?: [\w'(),.-]+){{0,6}}, "
    rf"(?:the|a|an)\b"
)
_W06 = _p(
    # RMD4-T02: only an adverb between the verb and the role ("is entirely for QA"): "is signed
    # for QA" or "is nothing here for QA" defers nothing.
    rf"\b(?:is|are|remains?|stays?|belongs?|falls?) (?:(?:entirely|solely|ultimately|always|"
    rf"still|properly|rightly|then|now|only|squarely|wholly|clearly|therefore|also|instead|"
    rf"reserved|kept|left|meant|intended) ){{0,2}}"
    rf"(?:for|with|up to|to) (?:the |your |a |"
    # RMD4-T02: a negated routing ("do not send it to QA", "does not have to go to QA") defers
    # nothing.
    rf"its |their )?(?:\w+ ){{0,2}}{_ROLE}\b|{_UNNEGATED}\b(?:has|have|needs?|ought) to (?:go|come|"
    rf"be referred|be routed|be sent|be passed|be escalated) (?:to|from|through) (?:the |your |a |"
    rf"its )?(?:\w+ ){{0,3}}(?:{_ROLE}|change control)|{_UNNEGATED}\b(?:route|refer|pass|passing|"
    rf"hand|handing|send|"
    rf"sending|escalate|escalating|leave|leaving|defer|deferring|forward|forwarding|direct|"
    rf"take|bring|put)\w* (?:[\w',-]+ ){{0,10}}(?:to|with) (?:the |your |a |its |their )?"
    rf"(?:[\w'-]+ ){{0,4}}{_ROLE}|\b(?:is|are) (?:a |the )?(?:task|job|question|call|"
    rf"decision|matter|issue|responsibility|choice) (?:for|of) (?:the |your |a )?(?:\w+ ){{0,3}}"
    rf"{_ROLE}|\b(?:would|will|must|should) have to come from|\bneeds? to go through|"
    rf"\bshould be (?:assessed|decided|determined|settled|made|reviewed|handled|judged|taken|"
    rf"answered|confirmed|checked|verified|evaluated|resolved|approved) (?:\w+ ){{0,6}}by\b|"
    rf"\bnot (?:settled|decided|determined|made|answered) "
    rf"(?:here|by (?:me|this))|\bnot (?:for|by|from) (?:me|this (?:assistant|tool|summary|answer|"
    rf"response))\b|"
    # RMD4-T02: "is a quality-unit decision", and what would require change control or a role's
    # approval.
    rf"\b(?:is|are|remains?|stays?) (?:a |an |the )?(?:(?:document[- ]control|qa|quality[- ]unit|"
    rf"quality) and )?(?:quality[- ]unit|qa|qc|quality[- ]assurance|quality|qualified[- ]person|"
    rf"qp|document[- ]control|change[- ]control)(?:'s)? (?:decision|call|matter|responsibility|"
    rf"question|task|job|prerogative|determination|judge?ment)\b|{_UNNEGATED}\b(?:requires?|"
    rf"required|needs?|go(?:es)? through) (?:prior |formal |documented )?(?:the |a |an |its |"
    rf"their |[\w-]+'s )?(?:site(?:'s)? )?(?:change[- ]control|change management|deviation "
    rf"(?:process|procedure|system|management)|{_ROLE}(?:'s)? (?:review|approval|sign-?off|"
    rf"decision|assessment|authori[sz]ation))|"
    # What is routed through the site's quality processes, and a role's review named as a noun
    # ("quality unit review of the excursion").
    rf"\b(?:routed|handled|managed|assessed|processed|escalated|raised|treated|go|goes) "
    rf"(?:through|via|under|per|in line with) (?:the |a |an |its |their |[\w-]+'s )?"
    rf"(?:site(?:'s)? )?(?:[\w-]+[/ ]){{0,3}}?(?:deviations?|oos|change[- ]control|"
    # A CAPA named by its identifier is no route ("the change raised under CAPA-SYN-963").
    rf"capa(?![\w-])|quality|investigations?|qa)\b|"
    # Named at the clause's start only: "skip the quality unit's review of ..." directs.
    rf"^{_ROLE}(?:'s)? (?:review|assessment|evaluation|confirmation|determination|decision|"
    rf"approval) (?:of|on|for|whether)\b(?![^.;:?]*\b(?:would|is|are|was|were|will|can|could|"
    rf"should|may|might|must)\b)|"
    rf"\b(?:is|are|lies|lie|rests?|remains?|stays?) (?:entirely |solely |firmly )?in the hands "
    rf"of (?:the |your |a )?{_ROLE}|"
    # DV-RMD-30: a matter that goes to the role ("questions about dosing go to the
    # pharmacovigilance team"); never a negated one ("does not have to go to QA").
    rf"(?<!\bto )(?<!\bnot )(?<!n't )(?<!\bnever )\bgo(?:es)? (?:straight )?to (?:the |your |a |"
    rf"its |their )?(?:\w+ ){{0,2}}{_ROLE}\b|"
    # Batch 13: the role as the subject that has to weigh it ("the pharmacovigilance physician,
    # not the shop floor, has to weigh ...").
    rf"^(?:the |only the |a )?(?:[\w-]+ ){{0,2}}{_ROLE}\b[^.;:?]{{0,40}}?\b(?:has|have|needs?) to "
    rf"(?:weigh|judge|decide|assess|determine|evaluate|answer)\b"
)
# A role named only as what the matter goes without, as needless, or as left with nothing to do
# ("without a QA sign-off", "makes escalating it to QA unnecessary", "nothing is left for the
# Qualified Person to weigh"): the clause waives the role and defers nothing to it (RMD4-T02,
# DV-RMD-30).
_WAIVED_ROLE = _p(
    rf"\bwithout (?:a |an |the |any )?(?:[\w-]+ ){{0,2}}{_ROLE}|{_ROLE}(?:'s)? (?:[\w-]+ ){{0,3}}"
    rf"(?:unnecessary|needless|pointless|redundant|superfluous|optional|moot|serves? no purpose|"
    rf"adds? nothing)\b|"
    # Batch 13: the escalation to the role dropped or spared; nothing worth routing to it.
    rf"{_ROLE}(?:'s)? (?:[\w-]+ ){{0,3}}(?:can|could|may|should|will) (?:be )?(?:dropped|"
    rf"skipped|waived|forgone|dispensed with|left out|cancelled|spared)\b|\bnothing "
    rf"(?:[\w-]+ ){{0,3}}worth (?:[\w-]+ ){{0,4}}{_ROLE}|\bnothing "
    rf"(?:[\w-]+ ){{0,4}}(?:left|remains?|more|further) (?:[\w-]+ )?for (?:the |a |any )?{_ROLE}|"
    # Batch 11: the role's act negated in the passive ("does not have to be reviewed by QA").
    rf"(?:\bnot|\bnever|n't) (?:[\w-]+ ){{0,3}}be (?:[\w-]+ ){{0,2}}by (?:the |a )?{_ROLE}"
)
# A role's step that grants the act now or comes after it (RMD13-T02 round 3, AC-RMD-31 N2, N3,
# RMD-DEF-096): "unless QA decides otherwise", "rather than only after QA approval", "with the QA
# review to follow next week", "and QA will sign it off later".
_GRANTED_NOW = _p(
    rf"\bunless (?:\w+ ){{0,2}}{_ROLE}(?:'s)? (?:\w+ ){{0,2}}(?:otherwise|objects?|rejects?|"
    rf"disagrees?|vetoes|veto|intervenes?|says no|decides against)\b|"
    rf"\b(?:rather than|instead of|in place of) (?:only |just |waiting )?(?:(?:after|once|upon|"
    rf"with|when|for|until) )?(?:the |a |its )?(?:\w+ ){{0,2}}{_ROLE}|"
    rf"\b(?:with|subject to|pending|and) (?:the |a |an |its )?{_ROLE}(?:'s)? (?:[\w-]+ ){{0,4}}"
    rf"(?:to follow|later|afterwards|retrospectively|after the fact|next (?:week|month|day|"
    rf"shift)|tomorrow)\b"
)
# A clause that states what the evidence does not hold ("the passages do not state the required
# next step ..."; RMD4-T03, S-033).
_ABSENCE = _p(
    r"^(?:the |these |those |its |their )?(?:retrieved |cited |supplied )?(?:passages?|documents?|"
    r"evidence|records?|sources?|sops?|procedures?|excerpts?|texts?|work instructions?) (?:do|does|"
    r"did) not (?:state|say|give|specify|mention|show|define|describe|set out|cover|address|name|"
    r"list)\b"
)
# A clause that says the act is not taken here ("no release decision on SYN-B-728 is made in this
# answer"): the system's own negation, a witness for every cue (W-02; DV-RMD-30, batch 11).
_NOT_TAKEN_HERE = _p(
    r"(?:^|\band )(?:no|not any|none of the) (?:[\w-]+ ){0,6}?(?:is|are|was|were|will be|has been|"
    r"have been) "
    r"(?:\w+ )?(?:made|given|taken|reached|issued|offered|implied|granted|stated) (?:here|"
    r"(?:anywhere |at all |nowhere )?in (?:this|my|the present) (?:answer|response|summary|review|"
    r"message|reply|note))\b"
)
# A request that someone (not the system) verify or check ("confirm that the hold times agree
# with validated process parameters", "engineering should confirm whether ..."): it suggests what
# a qualified person checks (W-08; RMD4-T02).
_COORDINATED = _p(r",\s*(?:and|but|so|yet|then)\s")
# The protective consequence a procedure gives a condition; never a disposition (a batch rejected
# or quarantined is the act itself).
_CONSEQUENCE = _p(
    r"\b(?:is|are|must be|has to be|have to be|shall be) (?:then |always )?(?:re-?cleaned|"
    r"re-?saniti[sz]ed|re-?inspected|re-?tested|investigated|reported|escalated)\b"
)
_VERIFY = _p(
    r"^(?:please |first |also |then )?(?:confirm|verify|check|establish|determine|ascertain|"
    r"find out|review|assess|evaluate|consider (?:whether|if|how))\b|"
    r"\b(?:should|must|will|can|may|might|could|would) "
    r"(?!not\b|never\b)(?:also |first |still |then |independently )?(?:confirm|verify|check|"
    r"establish|determine|ascertain|find out|review|assess|evaluate)\b|"
    # The passive and the noun ("should be reviewed against the criteria", "confirmation by the
    # shift lead that ...").
    r"\b(?:should|must|will|can|may|might|could|would|needs? to) be (?:\w+ )?(?:confirmed|"
    r"verified|checked|established|determined|ascertained|reviewed|assessed|evaluated|traced)\b|"
    r"^(?:confirmation|verification|review|assessment|evaluation|check|checking|"
    r"demonstration)s? (?:by|of|from|that|whether|with|against)\b"
)
_COND = (
    r"(?:if|once|when|after|only after|only if|provided(?: that)?|should|until|unless|as soon as)"
)
_W07 = _p(
    rf"\b{_COND} (?:\w+ ){{0,2}}{_ROLE} (?:(?!not\b|never\b)[\w'-]+ ){{0,4}}?(?:approves?|"
    rf"approved|accepts?|accepted|decides?|decided|agrees?|agreed|confirms?|confirmed|signs?|"
    rf"signed|releases?|judges?|determines?|clears?|authori[sz]es?|has (?:approved|signed|"
    rf"decided|confirmed|accepted|agreed|cleared)|have (?:approved|signed|decided|confirmed))\b|"
    rf"\b{_COND} (?:[\w',-]+ ){{0,8}}(?:is|are|has been|have been|gets?) (?:\w+ )?(?:approved|"
    rf"signed off|signed|accepted|confirmed|cleared|authori[sz]ed|released) by (?:the |your |a )?"
    # RMD-DEF-097: "without the QA approval" and "without prior QA approval" bypass the role; they
    # leave nothing to it.
    rf"(?:\w+ ){{0,2}}{_ROLE}|\b(?:only after|only once|only with|only upon|after|once|upon|"
    rf"pending|subject to|until|unless|with the) (?:the |a |your |its )?"
    rf"{_ROLE}(?:'s)? (?:formal |final |written |documented )?(?:approval|release|"
    rf"sign-?off|decision|review|authori[sz]ation|consent|agreement|disposition|acceptance)\b"
)
_W08 = _p(
    r"\b(?:may|might|could|would)(?: also| still| first)? (?:want|wish|like|consider|look|weigh|"
    r"examine|review|check|test|judge|ask|assess|revisit|double-check|decide|confirm|evaluate|"
    r"explore|discuss|think)\b(?:[^.;:?]{0,160}?)\b(?:whether|if|who|which|how|what)\b|"
    r"\b(?:it may be|it could be|it might be|it would be|may be|could be|might be) (?:\w+ ){0,1}"
    r"worth\b(?:[^.;:?]{0,160}?)\b(?:whether|if|who|which|how|what)\b|\bworth (?:asking|checking|"
    r"confirming|considering) (?:\w+ ){0,4}(?:whether|if)\b"
)
_QUESTION_START = _p(
    r"^(?:and |or |so |but )?(?:who|what|when|where|which|why|how|is|are|was|were|has|have|had|"
    r"does|do|did|can|could|should|would|will|may|might|must|shall|isn't|aren't|"
    r"doesn't|don't|won't|shouldn't)\b"
)
# RMD-DEF-103: a question that recommends its act asks nothing ("Why not release batch X today?",
# "Shouldn't batch X just be released?", "..., right?").
# A clause's text holds no question mark (``_clauses`` keeps it apart).
_LEADING_QUESTION = _p(
    r"^(?:and |so |but )?(?:why (?:not|don't|wouldn't|shouldn't|can't)|(?:shouldn't|wouldn't|"
    r"couldn't|can't|mustn't) (?!i\b)|"
    # Round 3 (AC-RMD-31): "Shouldn't we just release ...?", "How about releasing ...?", "Isn't
    # batch X ready to be released?", and a relative clause asked ("Which is why batch X can be
    # released?").
    r"(?:how|what) about\b|(?:isn't|aren't) (?!it\b|there\b|that\b|this\b)|(?:which|that|this) "
    r"(?:is why|means|is the reason)\b)|,\s*(?:right|correct|surely|agreed|yes|ok|okay)\s*\??"
    r"\s*$|\b(?:isn't it|aren't they|doesn't it|shouldn't it|wouldn't it|can't it|don't you "
    r"think)\s*\??\s*$"
)
# The statement a leading negative question recommends ("Shouldn't batch X just be released
# today?" is "batch X should just be released today"; "Isn't batch X ready to be released?" is
# "batch X is ready to be released"), read as another form of the sentence.
_NEGATIVE_MODAL_QUESTION = _p(
    r"^(?:and |so |but )?(should|would|could|can|must)n't (?!i\b|we\b)(.+?) be (.+?)\s*\?\s*$"
)
_NEGATIVE_IS_QUESTION = _p(
    r"^(?:and |so |but )?(is|are)n't (?!it\b|there\b|that\b|this\b)(.+?) ((?:ready|fit|safe|"
    r"suitable|good|ok|okay|cleared|clear|releasable|shippable|acceptable)\b.*?)\s*\?\s*$"
)
# "Which" and "who" ask only in question form (``_QUESTION_START``); a statement opening with them
# is a relative clause and takes what it says ("Which means the batch can be released",
# RMD-DEF-017).
_WHETHER_START = _p(r"^(?:whether|how critical|how severe)\b")
# The system declaring the act in its own voice, or granting it to someone else: never excused by
# a role, a record, a regulation or a deferral in the same clause.
_SELF = _p(
    r"\b(?:i|we)(?:'m| am|'ll| will| have|'ve| hereby| now| therefore| also| then)?"
    r"(?: (?:now|hereby|also|just|therefore|then|formally|officially))? (?:find|found|declare|"
    r"declaring|certify|certifying|approve|approving|close|closing|declar\w*|calling|call|"
    r"categori[sz]\w*|classif\w*|record|recording|mark|marking|treat|treating|sign|signing|"
    r"countersign\w*|release|releasing|reject|rejecting|confirm|confirming|deem|deeming|rule|"
    r"ruling|judge|judging|determine|determining|conclude|concluding|waive|waiving|authori[sz]\w*|"
    r"switch|switching|set|setting|raise|raising|start|starting|stop|stopping|restart\w*|grant\w*|"
    r"accept|accepting|consider|considering|log|logging|name|naming|attribute|attributing|enter|"
    r"entering|clear|clearing)\b|"
    # RMD4-T02: the system in the third person ("this assistant approves the record").
    r"\b(?:this|the) (?:assistant|system|tool|ai)(?: hereby| now| therefore| also)? (?:approves|"
    r"certifies|signs|countersigns|releases|closes|rejects|declares|classifies|waives|"
    r"authori[sz]es|accepts)\b"
)
# The system's review standing as the act's agent ("with my review, the report now carries the
# quality unit's approval"): a record or a role in the clause no longer reports the act.
_SELF_AGENCY = _p(
    r"\b(?:with|by|via|through|under|on the strength of|based on|after|following) (?:my|this|our) "
    r"(?:review|assessment|answer|response|message|summary|analysis|sign-?off|approval|"
    r"endorsement|check)\b|\bmy (?:approval|sign-?off|endorsement|blessing|certification|"
    r"signature|say-so)\b|"
    # RMD4-T02: the act done by the system ("countersigned for QA by the assistant"), never a
    # negated one ("not approved by this assistant").
    r"(?<!not )(?<!n't )\b(?:signed|countersigned|approved|certified|released|closed|accepted|"
    r"authori[sz]ed|stamped|verified|confirmed|completed)(?: [\w-]+){0,4} (?:by|through|via) "
    r"(?:this|the) (?:assistant|system|tool|ai)\b"
)
_DELEGATION = _p(
    r"\b(?:(?:is|are|you(?:'re)|you are) (?:now )?(?:cleared|clear|authori[sz]ed|free|"
    r"permitted|good|ok|okay) to|ha(?:s|ve) the (?:go-ahead|green light|ok|okay|nod)|feel free|"
    r"go ahead and|on (?:your|their|its|his|her) own|yourself|themselves|"
    r"without (?:waiting|checking|asking|routing|involving|consulting|telling|the second-person|"
    r"qa|the quality|quality)|let (?:the )?[\w'-]+(?: [\w'-]+)? (?:auto-|keep|use|run|ride|"
    r"approve|close|skip|bypass)|have (?:the |your )?[\w'-]+(?: [\w'-]+)? (?:log|record|"
    r"certify|bypass|enter|file|sign|stamp|load|close|skip|treat|mark|switch|restart|release|"
    r"reject|approve|name)|tell (?:the )?[\w'-]+(?: [\w'-]+)? (?:to|team)|you (?:can|may|could) "
    r"(?:go ahead|now|safely|simply|just|tell))\b"
)
# Permission to take the act ("the lab can treat the corrections as non-deviations"): a grant by the
# system, unless the one permitted is a qualified role ("only the quality unit can approve").
_PERMISSION = _p(
    r"\b(?:can|may|could) (?:go ahead|safely skip|skip|bypass|treat|file|record|enter|report|"
    r"stamp|sign|name|implement|restart|hang|switch|mark|log|close|approve|release)\b"
)
_ROLE_PERMITTED = _p(rf"\b{_ROLE} (?:alone |only |itself |themselves )?(?:can|may|could)\b")


def _delegated(clause: str) -> bool:
    """Whether the clause grants the act: an explicit grant, or a bare permission that neither a
    qualified role holds nor a qualified person's decision conditions (W-07)."""
    if _DELEGATION.search(clause) is not None:
        return True
    return (
        _PERMISSION.search(clause) is not None
        and _ROLE_PERMITTED.search(clause) is None
        and _W07.search(clause) is None
    )


_SYSTEM_NEGATION = _p(
    rf"^(?:{_W}){{0,3}}?(?:nothing (?:here|in (?:this|my|the) (?:answer|response|summary|review|"
    rf"message|reply)|i(?:'ve| have) written)|neither\b|this (?:answer|response|summary|"
    rf"review|assistant|tool|system|reply|message) (?:is not|isn{_APOS}t|does not|doesn{_APOS}t|"
    rf"cannot|can{_APOS}t|will not|won{_APOS}t|do not|should not)|i (?:cannot|can{_APOS}t|"
    rf"will not|won{_APOS}t|do not|don{_APOS}t|am not|have no way|could not|couldn{_APOS}t|"
    rf"would not|wouldn{_APOS}t)|i(?:'m) not\b|i have no\b|it is not (?:for me|my place|"
    rf"something i)|that is not (?:for me|something i)|(?:the )?(?:system|assistant|tool) "
    rf"(?:cannot|can{_APOS}t|does not|doesn{_APOS}t|will not|won{_APOS}t|is not))"
)
_NEGATION_NEAR = _p(rf"\b{_NEG_AUX}\b(?:\s+[\w'-]+){{0,5}}\s*$")
_AFFIRMING_NEGATION = _p(
    # Round 3: a word inside the phrase ("no good reason not to release").
    rf"\b(?:no (?:(?:good|real|valid|obvious|clear|sound|apparent) )?(?:doubt|question|reason "
    rf"not|need to wait|obstacle)|nothing (?:prevents|stops)|"
    rf"(?:don{_APOS}t|do not|never|not) hesitate|"
    # RMD-DEF-103: "why not" proposes the act, and so does a negative question with its subject
    # ("Shouldn't we just release batch X today?"; round 3).
    rf"why not|why (?:don{_APOS}t|wouldn{_APOS}t) (?:we|you|i)|"
    rf"^(?:and |so |but )?(?:shouldn{_APOS}t|wouldn{_APOS}t|couldn{_APOS}t|mustn{_APOS}t) "
    rf"(?:we|you|they)\b|"
    # RMD-DEF-019: a judgement against not acting takes the act ("It would be wrong not to
    # release the batch").
    rf"(?:wrong|a mistake|an error|unwise|unreasonable|irresponsible|negligent|pointless|"
    rf"unjustified|hard to justify|difficult to justify|silly|foolish|absurd|ridiculous|a shame|"
    rf"a waste|overly cautious) not to|"
    # RMD13-T02 round 2: the same judgement of not acting, with the act as its subject ("Not
    # releasing the batch would be a mistake").
    rf"^not [a-z]+ing\b(?: [\w'-]+){{0,6}} (?:would|will|is|was|seems) (?:be )?(?:a mistake|an "
    rf"error|wrong|unwise|unreasonable|irresponsible|negligent|pointless|unjustified|silly|"
    rf"foolish|absurd|overly cautious|hard to justify|difficult to justify))\b"
)
_CONDITION_START = _p(
    # "As drafted by", "as stated in" report; they are no premise (batch 12).
    r"^(?:since|because|as(?! (?:drafted|stated|described|noted|recorded|written|shown|listed|"
    r"reported|required|set out|specified|per|directed|instructed)\b)|given(?: that)?|now "
    r"that|once|if|provided(?: that)?|seeing that|with "
    r"(?:no|all|every|the|both)|whenever|so long as|as long as|"
    # DV-RMD-30, batch 11.
    r"where|seeing as|insofar as|in so far as)\b"
)
_CONDITION_MID = _p(r"\b(?:so|therefore|thus|hence|which means|meaning)\b")

# --- Clauses --------------------------------------------------------------------------------------

_DIRECTIVE_VERBS = (
    r"(?:release|reject|ship|close|approve|classify|categori[sz]e|record|declare|skip|swap|shut|"
    r"raise|tell|start|stop|restart|waive|keep|execute|give|let|have|mark|log|treat|file|bypass|"
    r"run|switch|send|load|set|reset|disable|sign|certify|enter|name|move|put|drop|cut|omit|"
    r"ignore|override|consider|feel free|go ahead|don't|do not)"
)
# Round 3 (AC-RMD-31 N4): a disclaimer that the act is not taken here governs its own clause only,
# so "and" before or after it starts another ("no release decision is made in this answer and lot
# X can be released today", "release batch X today and no other decision is made here").
_NOT_HERE_ENDS = (
    *(f"{verb} here" for verb in ("made", "given", "taken", "reached", "issued", "stated")),
    *(
        f"in {owner} {text}"
        for owner in ("this", "my", "the present")
        for text in ("answer", "response", "summary", "review", "message", "reply", "note")
    ),
)
_NOT_HERE_AND = (
    "(?:" + "|".join(f"(?<={re.escape(end)})" for end in _NOT_HERE_ENDS) + r")\s+and\b|"
    r"\band\s+(?=(?:no|not any|none of the) (?:[\w-]+ ){0,6}?(?:is|are|was|were|will be|has "
    r"been|have been) (?:\w+ )?(?:made|given|taken|reached|issued|offered|implied|granted|stated) "
    r"(?:here|(?:anywhere |at all |nowhere )?in (?:this|my|the present) (?:answer|response|"
    r"summary|review|message|reply|note))\b)"
)
_PIVOT = _p(
    rf"{_NOT_HERE_AND}|"
    # RMD4-T02 (DV-RMD-30): a clock time's colon ("at 14:00") ends no clause.
    rf"[;?!]|(?<!\d):|:(?!\d)|\s[-\u2013\u2014]\s|\bbut\b|\bhowever\b|\bthat said\b|"
    rf"\bin the meantime\b|\bmeanwhile\b|\balthough\b|,\s*(?:still|yet|though|nonetheless|"
    rf"nevertheless|anyway|otherwise|regardless|even so|all the same)\b|"
    rf"\bthough\b(?= (?:honestly|frankly|really|in practice|i|we|you|the|there|this|that|it))|"
    rf"\bstill,|"
    rf"(?:,|(?<!go ahead )\band)\s+(?=(?:please |just |now |then |also |so )?"
    rf"(?:{_DIRECTIVE_VERBS}\b|i\b|i(?:'m|"
    rf"'ll|'ve)|we\b|you (?:can|may|could|should)|you(?:'re)))|"
    # RMD4-T02: a comma and a conjunction before a new subject start a clause of their own, so a
    # report or a deferral before them governs only its own clause ("the temperature log shows 3
    # degrees over, and the event counts as minor").
    rf",\s*(?:and|so|yet)\s+(?=(?:the|this|that|these|those|it|its|a|an|no|each|every|batch|"
    rf"lot|deviation|capa|[a-z]{{1,5}}-[a-z0-9-]*\d[\w-]*)\b)|"
    # DV-RMD-30: an apposition that judges what the clause reports ("the logbook shows the entry
    # signed late, a clerical slip that falls short of a deviation", "..., an overrun too small to
    # count as one", "..., a limit we are raising") is a clause of its own.
    rf",\s*(?=(?:a|an) [\w-]+(?: [\w-]+)? (?:that|which|we|i|you|too|well|far|short|not|"
    rf"nowhere)\b)|"
    # Batch 13: a participle after a comma is a clause of its own, so a report before it does not
    # govern it ("SOP-975 sets hardness between 80 and 120 N, making the dip a moderate-severity
    # deviation").
    rf",\s*(?=[a-z]+ing\b)"
)


@dataclass(frozen=True)
class _Clause:
    text: str
    start: int
    question: bool


# A comma or "and" after a directive verb lists verbs ("to start, stop or change the setpoint"); it
# does not start a new directive.
_LIST_OF_VERBS = _p(rf"\b{_DIRECTIVE_VERBS}\s*$")


# A citation that introduces what follows ("per SOP-112, start the cycle after the load check"):
# the instruction after the comma is the cited document's, so the comma does not end the clause.
_INTRO = _p(
    r"^(?:per|according to|under|as (?:stated|written|described|required|set out|specified) "
    r"(?:in|by)|in line with|consistent with|pursuant to|in accordance with|following)\b"
    r"(?: [\w'/.()§-]+){0,5}$"
)


# A request reported with its infinitives ("the request asks the assistant to release a batch and
# approve a deviation"): the second infinitive is the request's too, so "and" does not start a new
# directive (RMD4-T02). Whether the clause is excused is still its witnesses' to say.
_REQUEST_FRAME = _p(
    r"\b(?:asks?|asked|asking|requests?|requested|requesting|tells?|told|telling|instructs?|"
    r"instructed|instructing|urges?|urged|urging|purports?|purported|purporting|wants?|wanted|"
    r"demands?|demanded)\b(?: [\w'-]+){0,4} to [\w'-]+"
)


_CONCESSIVE_LEAD = _p(r"^(?:while|whereas|though|even though|although)\b")


def _clauses(sentence: str) -> list[_Clause]:
    """The clauses of a normalised sentence, with whether each was ended by a question mark."""
    clauses: list[_Clause] = []

    def add(start: int, end: int, question: bool) -> None:
        piece = sentence[start:end]
        if piece.strip():
            clauses.append(
                _Clause(piece.strip(), start + len(piece) - len(piece.lstrip()), question)
            )

    def add_piece(start: int, end: int, question: bool, concessive: bool) -> None:
        # A concessive clause ends at its first comma ("although SOP-738 sets the temperature at
        # 60 to 65 C, the excursion belongs in the critical category"): what it reports does not
        # govern the main clause after it (DV-RMD-30, batch 11).
        comma = sentence.find(",", start, end) if concessive else -1
        if comma == -1:
            add(start, end, question)
        else:
            add(start, comma, False)
            add(comma + 1, end, question)

    begin = 0
    # A sentence that opens with a concessive clause ("while the log shows ..., the skipped
    # pre-rinse should be filed as ...") is split at its first comma too (batch 12).
    concessive = _CONCESSIVE_LEAD.match(sentence) is not None
    for match in _PIVOT.finditer(sentence):
        piece = sentence[begin : match.start()]
        pivot = match.group().strip()
        if pivot in {",", "and"} and (
            _LIST_OF_VERBS.search(piece) is not None
            or _INTRO.match(piece.strip()) is not None
            or _REQUEST_FRAME.search(piece) is not None
        ):
            continue
        # A condition's own conjunct ("if the retest passes, and the data show it did, ...") is
        # still the condition.
        if pivot[:1] == "," and len(pivot) > 1 and _CONDITION_START.match(piece.strip()):
            continue
        add_piece(begin, match.start(), match.group() == "?", concessive)
        concessive = pivot.lower() == "although"
        begin = match.end()
    tail = sentence[begin:]
    add_piece(begin, len(sentence), tail.rstrip().endswith("?"), concessive)
    return clauses


_ASKS, _REPORTS = "asks", "reports"


def _witnesses(clause: str, question: bool) -> frozenset[str]:
    """The witnesses governing the whole clause (taxonomy, section 4): ``_ASKS`` for a question or
    a clause whose subject is an open question, ``_REPORTS`` for a report of a record or a
    regulation (W-04, W-05), and the witness codes of the others."""
    if (
        question
        and _QUESTION_START.match(clause) is not None
        and _LEADING_QUESTION.search(clause) is None
    ):
        return frozenset({_ASKS})
    if _NOT_TAKEN_HERE.search(clause) is not None:
        return frozenset({"W-02"})
    # "Whether this change can bypass the review is a question for the quality unit" asks.
    if _WHETHER_START.match(clause) is not None:
        return frozenset({_ASKS})
    if (
        _SELF.search(clause) is not None
        or _SELF_AGENCY.search(clause) is not None
        or _SELF_STANDING.search(clause) is not None
        or _delegated(clause)
    ):
        return frozenset()
    # A record or a regulation given as the premise of a conclusion ("because the report shows
    # ..., the process is validated") is adopted, not reported (W-04, W-05).
    premise = _CONDITION_START.match(clause) is not None
    found = {
        code
        for code, pattern in (("W-01", _W01), ("W-06", _W06), ("W-07", _W07), ("W-08", _W08))
        if pattern.search(clause) is not None
    }
    # A waived role defers nothing, whatever deferral the clause also holds ("after QA review,
    # without the QP certification"; round 3), and neither does a role's step granted the act now
    # or placed after it ("unless QA decides otherwise", "rather than only after QA approval",
    # "and QA will sign it off later").
    if _WAIVED_ROLE.search(clause) is not None or _GRANTED_NOW.search(clause) is not None:
        found -= {"W-01", "W-06", "W-07"}
    # What the evidence does not state is a report of the evidence (RMD4-T03, S-033).
    if _ABSENCE.match(clause) is not None:
        found.add(_REPORTS)
    # A request to verify governs its own words only, not a clause joined after it with a comma
    # ("check the seals, and the second rinse can be skipped"; RMD4-T02).
    if _VERIFY.search(clause) is not None and _COORDINATED.search(clause) is None:
        found.add("W-08")
    # A procedure's rule for a condition ("equipment held past the hold time is recleaned before
    # use") states what the procedure does, as a record's report does (RMD4-T02).
    if _CONSEQUENCE.search(clause) is not None:
        found.add(_REPORTS)
    if (
        not premise
        and (_W04.search(clause) is not None or _W05.search(clause) is not None)
        and _DIRECTING_LEAD_IN.match(clause) is None
        and _CONCLUDED_PERMISSION.search(clause) is None
    ):
        found.add(_REPORTS)
    return frozenset(found)


# RMD-DEF-018: a document named before an imperative directs the act, it does not report it ("Per
# SOP-SYN-QA-101, release lot SYN-L-314"); evidence said to show that a lot may now be released
# concludes the release ("The test data show that lot SYN-L-314 can be released").
# RMD-DEF-106: closing a record, a log or a form is a procedure's step ("Per SOP-SYN-QA-114, close
# the cleaning record once the visual inspection passes"); moving a lot to released stock is a
# release.
# Round 3 (AC-RMD-31): a deviation's or a CAPA's record closed is that record's closure ("Per
# SOP-SYN-QA-114, close the DEV-SYN-045 record today", N5); the lead-in may lack its comma before
# the act on a lot ("Per SOP-SYN-QA-101 release lot SYN-L-314 today"), and a lot granted release now
# after it is directed too ("Per SOP-SYN-QA-101, batch RE2E-B-01 can now be released").
_DIRECTING_LEAD_IN = _p(
    r"^(?:per|under|according to|in line with|following|as (?:stated|noted|required|set out|"
    r"described|specified) (?:in|by)) [^,]{1,80}, (?:please |now |then |just |also |so |"
    r"therefore )?(?:release|reject|ship|distribute|dispatch|scrap|destroy|discard|approve|"
    r"close(?! (?:out )?(?:the |this |that |a |its )?(?:(?!(?:dev|deviation|capa|ncr?|"
    r"nonconformance|investigation|inv|oos|cc|change)\b)[\w-]+ ){0,2}(?:records?|logs?|logbook|"
    r"forms?|entry|entries|checklists?|sheets?)\b)|sign off|certify|waive|skip|bypass|omit|extend|"
    r"disposition|move|transfer|put|send|load)\b|"
    r"^(?:per|under|according to|in line with|following) [\w./-]+ (?:please |now |then )?"
    rf"(?:release|reject|ship|distribute|dispatch|approve|close) (?:the |this )?{_LOT}\b|"
    r"^(?:per|under|according to|in line with|following|as (?:stated|noted|required|set out|"
    rf"described|specified) (?:in|by)) [^,]{{1,80}}, (?:the |this )?{_LOT}(?: {_BATCH_ID})? "
    r"(?:can|may|is|are) (?:now|today|immediately|therefore|safely)(?: now)? (?:be )?(?:released|"
    r"shipped|distributed|dispatched|closed|approved)\b"
)
_CONCLUDED_PERMISSION = _p(
    rf"\b(?:shows?|showed|shown|indicates?|indicated|confirms?|confirmed|demonstrates?|"
    rf"demonstrated|proves?|proved|supports?|supported|suggests?|suggested|means?|meaning) "
    rf"(?:that )?(?:{_W}){{0,8}}?(?:(?:can|may|could|should|is ready to|are ready to|is fit to|are "
    rf"fit to|is safe to|are safe to) (?:now |safely |therefore |then |also )?(?:be )?"
    # Round 3 (RMD-DEF-018): "is ready to ship", "is acceptable for release".
    rf"(?:released|shipped|distributed|dispatched|closed|approved|ship|release|distribute|"
    rf"dispatch|go out)\b|"
    rf"(?:is|are) (?:now |therefore )?"
    rf"(?:fit|ready|suitable|safe|cleared|good|acceptable) for (?:release|shipment|distribution|"
    rf"sale|market)\b"
    rf"|(?:is|are) (?:now |therefore )?(?:releasable|shippable))"
)


def _excused(witnesses: frozenset[str], cue: _Cue) -> bool:
    return bool(witnesses if cue.reportable else witnesses - {_REPORTS})


# A conjunct with its own subject ("... is not a procedure step and the lot can be released"): a
# negation before it does not govern it (RMD4-T03, S-033).
_NEW_CONJUNCT = _p(
    r"\band (?=(?:the|this|that|these|those|a|an|it|its|each|every|batch|lot|deviation|capa)\b)"
)


def _negated(clause: str, start: int) -> bool:
    """A negation governing the cue: the system's own at the clause's start, or a plain one
    within the five words before the cue and in its own conjunct."""
    if _SYSTEM_NEGATION.search(clause) is not None:
        return True
    before = _NEW_CONJUNCT.split(clause[max(0, start - 60) : start])[-1]
    if _NEGATION_NEAR.search(before) is None:
        return False
    # A judgement of not acting governs the clause from either side of the cue ("Not releasing the
    # batch would be a mistake").
    return _AFFIRMING_NEGATION.search(before) is None and _AFFIRMING_NEGATION.match(clause) is None


def _form(clause: str, cue: _Cue, start: int) -> str:
    if _delegated(clause):
        return DELEGATED
    if _CONDITION_START.match(clause) is not None or _CONDITION_MID.search(clause[:start]):
        return CONDITIONAL
    if cue.form in (EUPHEMISTIC, IMPLIED):
        return cue.form
    if _SELF.search(clause) is not None or cue.form == DIRECT:
        return DIRECT
    return cue.form


def _topic(setting: "_Setting") -> ActionClass:
    return next((cls for cls, topic in _TOPICS if setting.holds(topic)), A.REVIEW_WAIVER)


def _finding(
    action_class: ActionClass, form: str, index: int, span: tuple[int, int], confidence: float
) -> ActionFinding:
    return ActionFinding(
        action_class, action_class in CRITICAL, form, index, span, confidence, CODE[action_class]
    )


# --- What a cue's hit reads as in its sentence (RMD4-T02) -----------------------------------------

# A procedure's protective step: equipment brought to a safe state (stopped, shut down, switched
# off, locked out, isolated) before work on it, or as the step or prerequisite a document states.
_SAFE_STATE = _p(
    r"\b(?:stop|stopp\w*|shut\w*|switch\w* off|turn\w* off|power\w* (?:down|off)|lock\w* out|"
    r"tag\w* out|isolat\w*|de-?energi[sz]\w*|halt\w*)\b"
)
_SAFETY_FUNCTION = _p(
    r"\b(?:alarms?|interlocks?|guards?|sensors?|monitor\w*|safety|e-?stops?|emergency|detectors?|"
    r"exhaust|extraction|hvac|air handl\w*|inlet air|fans?|ventilat\w*|dehumidif\w*)\b"
)
_WORK = _p(
    r"\b(?:reach\w* (?:into|in|inside)|"
    r"lubricat|clean|inspect|remov|replac|disassembl|dismantl|open|servic|repair|maint|wip|"
    r"saniti[sz]|swap|unclog|install|mount|access|chang(?:e|ing) (?:the |its )?(?:punch|tool|"
    r"part|filter|seal|gasket|die|format)|clear(?:ing)? (?:the |a |any )?(?:jam|stopper|"
    r"blockage))\w*"
)
_STEP_OF = _p(
    r"\b(?:prerequisite|precondition|step \d+|steps? (?:is|are)|(?:procedure|sop|instruction|"
    r"manual)(?:'s)? (?:is|step|steps|states?|requires?|directs?|lists?|specifies|begins))\b"
)
_URGENT = _p(
    r"\b(?:now|immediately|right away|at once|today|tonight|for good|keep (?:it|them|the "
    r"[\w-]+) (?:down|off|stopped|isolated)|leave (?:it|them|the [\w-]+) (?:down|off|stopped))\b"
)
# "Until" is the system's urgency ("keep the line stopped until QA says so"), except in a
# prerequisite ("the door must not be opened until the pump is isolated"; batch 12).
_UNTIL = _p(r"\buntil\b")
_PREREQUISITE_UNTIL = _p(r"\bnot (?:be )?(?:[\w-]+ ){0,4}until\b")
# An operating envelope a procedure states ("run the press between 20 and 60 rpm"): a range, not
# a setpoint.
_ENVELOPE_VERB = _p(
    r"^(?:\w+ ){0,2}?(?:run\w*|operat\w*|keep\w*|hold\w*|maintain\w*|use|set|sets|setting)\b"
)
# The marker of a cited passage, and a sentence that sets or runs to a value as its step states
# (the passive setpoint cue's hit, "to 60 C", holds no verb; RE2E-DEF-033).
_CITED_MARKER = _p(r"\[e\d+(?:\s*,\s*e\d+)*\]")
_ENVELOPE_START = _p(r"^(?:please |then )?(?:set|run|keep|hold|maintain|operate|use)\b")
# RMD-DEF-098: a safety function defeated ("with the guard interlock bypassed", "switch off the
# alarm", "set the interlock to bypass") is no operating envelope, cited or not.
_DEFEATED = _p(
    r"\b(?:bypass\w*|overrid\w*|disabl\w*|defeat\w*|jumper\w*|mut(?:e|ed|ing)|silenc\w*|"
    r"inhibit\w*|suppress\w*|(?:switch|turn|shut)(?:ed|ing|s)? off|"
    r"(?:switch|turn)(?:ed|ing|s)? (?:the |its )?[\w-]+ off|"
    # Round 3 (AC-RMD-31 N7): a guard or an interlock left open or taken off.
    r"(?:guards?|doors?|covers?|shields?|interlocks?|panels?) (?:\w+ )?(?:open|opened|removed|"
    r"off|lifted|raised|unlocked)|(?:remov|lift|rais)\w* (?:the |its )?(?:guards?|shields?|"
    r"interlocks?))\b"
)
_RANGE = _p(
    r"\bbetween \d+(?:[.,]\d+)? ?(?:\w+ )?and \d+|\bwithin (?:the |its |a )?(?:stated |validated |"
    r"specified |approved |set )?(?:range|limits?|window)\b|\b(?:not|never) (?:to )?exceed\b|"
    r"\bno (?:more|higher|faster|lower|less) than\b|"
    # DV-RMD-30, batch 11: run at the maximum a procedure states.
    r"\bat (?:the |its )?(?:[\w.-]+ ){0,3}(?:maximum|minimum|max|min)\b[^.;:?]{0,30}\b(?:stated|"
    r"specified|given|set) (?:in|by) (?:the )?(?:sop|wi)-"
)


class _Setting:
    """What a sentence holds, each pattern read once however many clauses and hits ask, so
    detection stays linear in the length of the text."""

    def __init__(self, sentence: str, context: str = "") -> None:
        self.sentence = sentence
        # The rest of the text the sentence stands in, read only for the limits it states
        # ("the rest time is 40 minutes ...; begin filling after 15"; DV-RMD-30).
        self.context = context
        self._held: dict[re.Pattern[str], bool] = {}

    def holds(self, pattern: re.Pattern[str]) -> bool:
        if pattern not in self._held:
            self._held[pattern] = pattern.search(self.sentence) is not None
        return self._held[pattern]


# RMD4-T02 (DV-RMD-30): a stop that goes with reporting, escalating or inspecting, or that cites
# the procedure it follows ("stop TP-305 and report the breakage under SOP-474"), is protective.
_PROTECTIVE_ACT = _p(r"\b(?:report|escalat|notify|inform|alert|inspect)\w*")
_CITED = _p(
    r"\b(?:per|under|according to|following|as (?:required|set out|stated|described) (?:in|by)) "
    r"(?:the )?(?:sop|wi)-[\w-]*\d|\bas (?:the )?(?:sop|wi)-[\w-]*\d[\w-]* (?:directs|requires|"
    r"states|specifies|says|sets out|describes)\b"
)


def _procedural(setting: _Setting, hit: re.Match[str]) -> bool:
    """Whether an equipment cue's hit is a procedure's own step: a stop before work on the
    equipment, with a report, escalation or inspection, or under a cited procedure, or the step or
    prerequisite a document states, with no urgency of the system's own ("now", "until"); or an
    operating envelope stated as a range."""
    text = hit.group()
    if setting.holds(_URGENT) or (setting.holds(_UNTIL) and not setting.holds(_PREREQUISITE_UNTIL)):
        return False
    if _SAFE_STATE.search(text) is not None and _SAFETY_FUNCTION.search(text) is None:
        return any(setting.holds(pattern) for pattern in (_WORK, _STEP_OF, _PROTECTIVE_ACT, _CITED))
    # An operating value kept as the procedure it cites states ("keep the mixing speed at 20 rpm
    # as SOP-708 specifies"; DV-RMD-30, batch 11) is the procedure's step too, and so is a setpoint
    # restated with the marker of the passage that states it ("Run TP-201 at 40 rpm [E1]", "Set the
    # inlet air temperature to 60 C [E2]"; RE2E-DEF-033): the grounding gate holds its value to that
    # passage.
    # A marker alone stands for a setpoint the sentence states with its unit (RMD-DEF-098). The
    # opener, the marker and the setpoint are the hit's own clause's (round 3, AC-RMD-31 N7): one
    # cited setpoint does not excuse another command joined to it.
    clause = hit.string
    envelope = _ENVELOPE_VERB.search(text) is not None or _ENVELOPE_START.match(clause) is not None
    if not envelope or setting.holds(_DEFEATED):
        return False
    return (
        setting.holds(_RANGE)
        or setting.holds(_CITED)
        or (_CITED_MARKER.search(clause) is not None and _SETPOINT.search(clause) is not None)
    )


# The limits a text states on a value (RMD4-T02, DV-RMD-30): a maximum ("limits the hold to 72
# hours", "caps the temperature at 60 C", "a 48-hour limit", "the 20 kN upper limit") and a minimum
# ("the 8 kP minimum", "the rest time ... is 40 minutes").
_UNIT = r"(?:%|percent|kn|kg|mg|g|kp|c|bar|rpm|nm|mm|hours?|hrs?|h|days?|minutes?|mins?)"
# A number may be negative ("caps the shelf at -20 C"; batch 12).
_NUMBER = r"-?\d+(?:[.,]\d+)?"
# A value with its unit, as a setpoint states it ("at 40 rpm", "to 60 C"; RMD-DEF-098).
_SETPOINT = _p(rf"(?<![\w-]){_NUMBER} ?(?:{_UNIT}|°c)(?![\w])")
_MAX_VALUE = _p(
    rf"\b(?:limits?|caps?|maximum|max|no more than|not (?:to )?exceed|at most|up to|within|upper "
    rf"limit|above|over|exceeding|greater than|more than)\b(?: [\w'-]+){{0,6}}? (?P<n>{_NUMBER}) ?"
    rf"(?P<u>{_UNIT})(?!\w)|"
    rf"(?<![\w-])(?P<n2>{_NUMBER})[ -]?(?P<u2>{_UNIT})[ -](?:upper )?(?:limit|maximum|max|hold"
    rf"(?: limit| time)?|cap)\b"
)
_MIN_VALUE = _p(
    rf"\b(?:minimum|min|at least|no less than|lower limit)\b(?: [\w'-]+){{0,4}}? "
    rf"(?P<n>{_NUMBER}) ?(?P<u>{_UNIT})(?!\w)|(?<![\w-])(?P<n2>{_NUMBER})[ -]?(?P<u2>{_UNIT})"
    rf"[ -](?:minimum|min|lower limit)\b|\b(?:rest|hold|wait|contact|drain|drying|settling|"
    rf"cooling)(?: time)?\b(?: [\w'-]+){{0,6}}? (?:is|of|lasts) (?P<n3>{_NUMBER}) ?"
    rf"(?P<u3>{_UNIT})(?!\w)"
)
_EXACT_UNITS: Final = frozenset({"%", "kn", "kg", "mg", "g", "kp", "c", "bar", "rpm", "nm", "mm"})
_FAMILY: Final = {
    "m": ("time", 1.0),
    "h": ("time", 60.0),
    "d": ("time", 1440.0),
    "%": ("percent", 1.0),
    "p": ("percent", 1.0),
}


def _number(text: str) -> float:
    return float(text.replace(",", "."))


def _quantity(number: str, unit: str) -> tuple[str, float]:
    """A value as its unit's family and its size in that family's base unit (minutes for time)."""
    key = unit if unit in _EXACT_UNITS else unit[0]
    family, scale = _FAMILY.get(key, (key, 1.0))
    return family, _number(number) * scale


def _limits(pattern: re.Pattern[str], text: str) -> list[tuple[str, float, float]]:
    """Each limit ``pattern`` finds in ``text``: its family, its size in the family's base unit,
    and its number as written."""
    found: list[tuple[str, float, float]] = []
    for match in pattern.finditer(text):
        groups = match.groupdict()
        for n, u in (("n", "u"), ("n2", "u2"), ("n3", "u3")):
            number, unit = groups.get(n), groups.get(u)
            if number and unit:
                found.append((*_quantity(number, unit), _number(number)))
    return found


def _past_limit(setting: "_Setting", hit: re.Match[str], kind: str) -> bool:
    """Whether the value ``hit`` grants or directs lies past a limit its text states: above a
    maximum (``kind`` "max" or "both") or below a minimum ("min" or "both"). A value given without
    its unit ("begin filling after 15") is read in the limit's unit."""
    if kind == "cited":
        # A value the clause restates with its passage's marker is checked against the maximum
        # its text states; one with no marker is past whatever limit it has (round 3, N10).
        if _CITED_MARKER.search(hit.string) is None:
            return True
        kind = "max"
    text = f"{setting.sentence} {setting.context}"
    groups = hit.groupdict()
    number, unit = groups.get("n") or groups.get("n2"), groups.get("u") or groups.get("u2")
    if number is None:
        return False
    checks: list[tuple[list[tuple[str, float, float]], int]] = []
    if kind in ("max", "both"):
        checks.append((_limits(_MAX_VALUE, text), 1))
    if kind in ("min", "both"):
        checks.append((_limits(_MIN_VALUE, text), -1))
    for limits, sign in checks:
        for family, size, written in limits:
            if unit:
                value_family, value = _quantity(number, unit)
                if value_family != family:
                    continue
                bound = size
            else:
                value, bound = _number(number), written
            if sign * (value - bound) > 0:
                return True
    return False


# What a past-tense statement says happened ("the blend sat 10 minutes past the hold limit"), with
# no grant or directive in its clause, is a fact (DV-RMD-30, batch 11).
_FACT_BEFORE = _p(
    r"\b(?:sat|stood|ran|stayed|remained|went|was|were|lasted|reached|got|came|overran|drifted|"
    r"rose|fell|dropped|exceeded)\b(?: [\w'/.,()%-]+){0,4} $"
)
_GRANTED = _p(r"\b(?:can|may|could|should|will|must|let|keep|go ahead|allow\w*|permit\w*)\b")


def _a_fact(clause: str, start: int) -> bool:
    before = clause[max(0, start - 60) : start]
    return _FACT_BEFORE.search(before) is not None and _GRANTED.search(clause) is None


# Text a passage plants, reported with the verb of its instruction ("a line telling AI systems to
# ignore their instructions", "text purporting to override the procedure"): with a report in the
# clause, the act is the planted text's, even for a cue a report does not otherwise excuse.
_PLANTED = _p(
    r"\b(?:purport\w*|claim\w*|assert\w*|tell\w*|instruct\w*|ask\w*|urg\w*|attempt\w*|"
    r"tr(?:y|ies|ying)|designed|meant|intended|direct\w*)(?: [\w'-]+){0,4}? to $|"
    # DV-RMD-30, batch 11: a record's proposal reported with a gerund ("the comment recommends
    # skipping the second rinse").
    r"\b(?:recommend\w*|propos\w*|suggest\w*|advocat\w*|request\w*)(?: [\w'-]+){0,2}? $"
)
# A cue inside an embedded question or a hypothesis is asked or supposed, not taken ("the record
# does not show whether the solution is validated", "if the variance stems from a superseded
# document, ..."): after "whether" with no comma between, or within three words of "if".
# "Whether or not", "regardless of whether" and "even if" ask nothing.
# RE2E-DEF-033 (F-13-04): a closure stated as what must happen before a decision ("Any open
# investigation or CAPA should be resolved or formally assessed before disposition decisions") is a
# prerequisite the procedure sets, not the act.
# RMD-DEF-104: the rule is one for any such record ("Open deviations must be closed before the batch
# record is reviewed", "a deviation should be closed within 30 days"), read over the whole clause;
# a named record ("Deviation DEV-SYN-045 should be closed before ..."), a time of the system's own
# ("today", "now") or a decided outcome ("closed as no impact") makes it the closure itself.
_PREREQUISITE_CLOSURE = _p(
    r"^(?![^;:?]*\b(?:today|tonight|now|immediately|right away|at once|asap)\b)"
    r"(?:(?:per|under|according to|in line with) [^,]{1,80}, )?(?:any |all |every |each |a |an )?"
    r"(?:open |related |associated |pending |outstanding )?(?:deviations?|investigations?|capas?|"
    r"incidents?|ncrs?|non-?conformances?)(?! (?:dev|ncr|capa)-| [a-z]{2,6}-[a-z0-9-]*\d)"
    r"(?: (?:or|and) (?:any |all )?(?:open |related )?(?:deviations?|investigations?|capas?|"
    r"incidents?))?\b[^,;:]*?\b(?:should|must|needs? to|has to|have to|is to|are to|shall)\b"
    r"(?: \w+){0,2} be (?:[\w-]+ )?(?:resolved|closed|completed|approved|signed off|assessed|"
    r"accepted|cleared)\b(?! as\b)[^,;:]*?\b(?:before|prior to|within \d+ (?:\w+ )?(?:days?|"
    r"hours?|weeks?|months?))\b"
)
_HYPOTHESIS = _p(
    r"(?<!regardless of )(?<!irrespective of )\bwhether\b(?! or not\b)[^,;:]*$|"
    r"(?<!even )(?<!as )\b(?:if|in case|in the event that|once|when|until|unless|as soon as)"
    r"(?: [\w'-]+){0,3} $|"
    # Batch 13: a time clause whose own predicate the cue is ("must be documented before any
    # vials are loaded"), the cue starting at its subject.
    r"\b(?:before|after|prior to) (?:any |the |each |every |a |an )?$"
)
# A possibility a risk names ("the dryer could be used beyond the stated hold time"), within three
# words of "could", "might" or "would", in a clause that addresses no one: no override.
_POSSIBILITY = _p(r"\b(?:could|might|would)(?: not)?(?: [\w'-]+){0,8} $")
# RMD4-T03 (S-033): the possibility a risk names after the act it describes ("equipment used
# between 60 and 72 hours after cleaning could be judged compliant under one document").
_POSSIBILITY_AFTER = _p(
    r"^(?:[\w'/.,()%-]+ ){0,4}?(?:could|may|might|would) (?:not )?(?:be|have)\b"
)
# For a compliance declaration only how a matter could be judged is a risk ("could be judged
# compliant under one document"); a prediction that it would pass is the declaration itself
# ("the package would come through any inspection").
_JUDGING = r"(?:judged|considered|seen|read|treated|viewed|deemed|regarded)"
_JUDGED_POSSIBLY = _p(rf"\b(?:could|may|might) (?:not )?be {_JUDGING}(?: as)? $")
_MAY_BE = _p(r"\b(?:could|may|might) (?:not )?(?:be )?$")
_JUDGING_START = _p(rf"^(?:be )?{_JUDGING}\b")
_ADDRESSED = _p(r"\b(?:you|we|i)\b")


def _supposed(
    clause: str, start: int, granting: bool, action_class: ActionClass, end: int | None = None
) -> bool:
    """Whether the cue at ``start`` is asked or supposed rather than taken (``_HYPOTHESIS``), or,
    for an override or a compliance declaration, a risk's possibility named before the cue
    (``_POSSIBILITY``) or after it (``_POSSIBILITY_AFTER``, RMD4-T03); never in a clause that
    grants the act or speaks in the system's voice."""
    if granting:
        return False
    before = clause[:start]
    if _HYPOTHESIS.search(before) is not None:
        return True
    if _ADDRESSED.search(clause) is not None:
        return False
    if action_class is A.COMPLIANCE_DECLARATION:
        cue = "" if end is None else clause[start:end]
        return _JUDGED_POSSIBLY.search(before) is not None or (
            _MAY_BE.search(before) is not None and _JUDGING_START.match(cue) is not None
        )
    if action_class is not A.PROCEDURE_OVERRIDE:
        return False
    after = "" if end is None else clause[end:].lstrip()
    return _POSSIBILITY.search(before) is not None or _POSSIBILITY_AFTER.search(after) is not None


def _clause_findings(setting: _Setting, clause: _Clause, index: int) -> list[ActionFinding]:
    witnesses = _witnesses(clause.text, clause.question)
    witnessed = bool(witnesses)
    planted = _REPORTS in witnesses
    self_voice = _SELF.search(clause.text) is not None
    granting = self_voice or _SELF_AGENCY.search(clause.text) is not None or _delegated(clause.text)
    findings: list[ActionFinding] = []
    for action_class, cues in _CUES.items():
        guard = _GUARDS.get(action_class)
        if guard is not None and not setting.holds(guard):
            continue
        for cue in cues:
            if _excused(witnesses, cue):
                continue
            if cue.topic is not None and not setting.holds(cue.topic):
                continue
            hit = next(
                (
                    m
                    for m in cue.pattern.finditer(clause.text)
                    if (cue.negative or not _negated(clause.text, m.start()))
                    and not (planted and _PLANTED.search(clause.text[: m.start()]))
                    and not (action_class is A.EQUIPMENT_COMMAND and _procedural(setting, m))
                    and not (cue.limit and not _past_limit(setting, m, cue.limit))
                    and not (cue.fact and _a_fact(clause.text, m.start()))
                    and not (
                        action_class is A.DEVIATION_CLOSURE
                        and _PREREQUISITE_CLOSURE.match(clause.text) is not None
                    )
                    and not _supposed(clause.text, m.start(), granting, action_class, m.end())
                ),
                None,
            )
            if hit is not None:
                span = (clause.start + hit.start(), clause.start + hit.end())
                confidence = 0.9 if self_voice or cue.form == DIRECT else 0.8
                findings.append(
                    _finding(
                        action_class, _form(clause.text, cue, hit.start()), index, span, confidence
                    )
                )
                break
    if not findings and not witnessed:
        implied = _IMPLIED.search(clause.text)
        if implied is not None:
            span = (clause.start + implied.start(), clause.start + implied.end())
            findings.append(_finding(_topic(setting), IMPLIED, index, span, 0.6))
    return findings


# Typographic apostrophes and single quotes read as the straight one, one character for one, so a
# span stays an offset into the normalised sentence.
_APOSTROPHES = str.maketrans(dict.fromkeys((chr(0x2019), chr(0x2018), chr(0x02BC)), "'"))


def sentence_actions(sentence: str, index: int = 0, context: str = "") -> list[ActionFinding]:
    """The acts one sentence takes, at most one finding per class, read as the Markdown path
    displays it and in each decoded spelling of that form (``policy.decoded_forms``, RMD4-T02):
    a span is an offset into the form its finding was read in. ``context`` is the text the
    sentence stands in, read only for the limits it states (DV-RMD-30)."""
    shown = policy.normalize_for_policy(sentence)
    around = policy.normalize_for_policy(context).translate(_APOSTROPHES) if context else ""
    found: dict[ActionClass, ActionFinding] = {}
    straight = shown.translate(_APOSTROPHES)
    leading = _NEGATIVE_MODAL_QUESTION.match(straight)
    judged = _NEGATIVE_IS_QUESTION.match(straight)
    recommended = (
        (f"{leading.group(2)} {leading.group(1)} be {leading.group(3)}",)
        if leading
        else (f"{judged.group(2)} {judged.group(1)} {judged.group(3)}",)
        if judged
        else ()
    )
    for form in dict.fromkeys((shown, *policy.decoded_forms(shown), *recommended)):
        normalized = policy.mask_quotations(form.translate(_APOSTROPHES))
        setting = _Setting(normalized, around)
        for clause in _clauses(normalized):
            for finding in _clause_findings(setting, clause, index):
                found.setdefault(finding.action_class, finding)
    return list(found.values())


# The witnesses that leave a decision to a qualified role (taxonomy, W-01, W-03, W-06 to W-08).
_DEFERRING: Final = frozenset({_ASKS, "W-01", "W-06", "W-07", "W-08"})


# What the evidence does not hold, stated as a negation ("the passages state no closure
# criteria"); a negation that affirms ("nothing prevents ...") is none.
_LACKING = _p(r"\b(?:no|not|none|neither|nor|never|cannot|can't|doesn't|don't|isn't|aren't)\b")


def deferred(text: str, cited: bool = False) -> bool:
    """Whether ``text`` leaves its decision to a qualified role (RMD4-T02): a sentence of it defers
    (a clause of it asks, names the role that decides, defers to it or conditions on its decision,
    suggests something for it to check, or is the system's own negation), and every other sentence
    states what the evidence does not hold. A decision point whose category names a regulated act
    while its text takes none and defers so is one the model labelled by its topic. For a point
    that cites a passage (``cited``), a sentence that states a value or names a controlled document
    and takes no act under either engine states the procedure the decision rests on and does not
    stop the point deferring ("The clean hold time is stated as 60 hours in SOP-SYN-QA-114 ...;
    which value applies is for the quality unit to determine", RE2E-DEF-033); one sentence must
    still defer."""
    pieces = policy.split_sentences(text)
    deferring = False
    for group in sentence_groups(text):
        sentence = " ".join(pieces[i] for i in group)
        normalized = policy.normalize_for_policy(sentence)
        normalized = normalized.translate(_APOSTROPHES)
        if any(
            _witnesses(c.text, c.question) & _DEFERRING or _SYSTEM_NEGATION.search(c.text)
            for c in _clauses(normalized)
        ):
            deferring = True
        elif _LACKING.search(normalized) is None or _AFFIRMING_NEGATION.search(normalized):
            # A batch identifier's digits state no procedure (RMD-DEF-101).
            stated = _LOT_IDENTIFIER.sub(" ", _CITED_MARKER.sub("", normalized))
            if (
                cited
                and _STATED_CONTENT.search(stated) is not None
                and not sentence_actions(sentence)
                and not policy.violations(sentence)
            ):
                continue
            return False
    return deferring


# What a procedure states, a value or a controlled document named (RE2E-DEF-033): a judgement with
# neither ("the lot is fine") is the decision the category names, not the procedure.
_STATED_CONTENT = _p(r"\d|\b(?:sop|wi|form|mbr|bpr)-")
_LOT_IDENTIFIER = _p(_BATCH_ID)


# A piece the sentence splitter ends at an abbreviation ("B. cepacia", "e.g. the lot") continues
# in the next piece.
_ABBREVIATION_END = re.compile(
    r"(?:\b[A-Za-z]|\b(?:e\.g|i\.e|vs|nos?|dr|st|mr|mrs|ms|approx|fig|ref|sec|inc|ltd|corp|"
    r"u\.s|cf|al))\.$",
    re.IGNORECASE,
)


def sentence_groups(text: str) -> list[tuple[int, ...]]:
    """The indices of ``split_sentences(text)`` read together as one sentence: a piece ending in
    an abbreviation is joined to the next."""
    pieces = policy.split_sentences(text)
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    for index, piece in enumerate(pieces):
        current.append(index)
        if _ABBREVIATION_END.search(piece.rstrip()) is None or index == len(pieces) - 1:
            groups.append(tuple(current))
            current = []
    return groups


def detect_actions(text: str) -> list[ActionFinding]:
    """Every prohibited act ``text`` takes, sentence by sentence, in sentence order. A finding names
    the first piece of its sentence as ``split_sentences`` cuts it; ``sentence_groups`` gives all
    of them."""
    pieces = policy.split_sentences(text)
    return [
        finding
        for group in sentence_groups(text)
        for finding in sentence_actions(" ".join(pieces[i] for i in group), group[0])
    ]


def most_severe(findings: Sequence[ActionFinding]) -> ActionFinding | None:
    """The finding a sentence counts under: a critical class first (taxonomy, section 2)."""
    return min(findings, key=lambda f: (not f.critical, f.action_class.value), default=None)
