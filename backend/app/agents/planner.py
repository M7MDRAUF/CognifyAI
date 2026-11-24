"""Planner agent (FR-AGENT-001)."""

from app.agents.base import AgentResult, JsonAgentRunner
from app.agents.prompts import SAFETY_RULES, flatten
from app.integrations.llm.base import ModelRole
from app.schemas.agents import PlannerOutput
from app.schemas.projects import MAX_PROJECT_LABEL_LENGTH, MAX_PROJECT_PARAGRAPH_LENGTH
from app.schemas.queries import MAX_QUESTION_LENGTH

_SYSTEM = (
    "You are the planning agent for an evidence-based decision system. Do not state whether "
    "evidence was or was not supplied: you have not seen any. Given a user question, "
    "normalize it, break it into sub-questions, and produce a retrieval plan (search queries) "
    "and the kinds of evidence and risks to look for.\n" + SAFETY_RULES
)


class PlannerAgent:
    def __init__(self, runner: JsonAgentRunner) -> None:
        self._runner = runner

    async def run(self, question: str, domain: str, objective: str) -> AgentResult[PlannerOutput]:
        """Assemble the one prompt that carries the *request* text rather than model output.

        Every other agent reaches ``prompts`` for its interpolations; this one reached for none
        of them, and it is the only agent whose three caller-controlled values all sit on
        labelled lines of the instruction region.

        **Length.** A ``max_length`` on ``ProjectCreate`` bounds what can be stored from now on
        and nothing that is stored already: ``run_query_job`` reads ``domain`` and ``objective``
        back out of the ``projects`` table, so a row written before the bound existed is still a
        row this method would interpolate -- three times over, once per rung of
        ``JsonAgentRunner.run``. The same constants are therefore applied again here, at the
        point the value actually reaches a billed prompt. Same constants on both sides, so the
        stored ceiling and the sent ceiling cannot drift. ``question`` gets its own request cap
        rather than ``format_question``'s ``MAX_QUESTION_CHARS``: that limit governs *model*
        output (``normalized_question``), and borrowing it here would silently halve a question
        the API expressly accepts.

        **Line-opening.** ``schemas/_text.py`` permits ``\\n``, ``\\r`` and ``\\t`` in all of
        this text on purpose -- they are legitimate in a multi-line question -- so nothing on
        the request side stops a value opening a line. The grammar it would open a line in is
        this one: ``Question: ``/``Project domain: ``/``Analysis objective: `` at column zero,
        outside any fence. ``flatten`` is what confines a value to the line its label gave it,
        and it collapses every terminator ``str.splitlines()`` honours, not only ``\\n``.
        """
        user = (
            f"Question: {flatten(question, MAX_QUESTION_LENGTH)}\n"
            f"Project domain: {flatten(domain, MAX_PROJECT_LABEL_LENGTH) or '(unspecified)'}\n"
            "Analysis objective: "
            f"{flatten(objective, MAX_PROJECT_PARAGRAPH_LENGTH) or '(unspecified)'}\n\n"
            'Return JSON: {"normalized_question": str, "sub_questions": [str], '
            '"retrieval_plan": {"queries": [str], "semantic_top_k": int, "lexical_top_k": int}, '
            '"required_evidence": [str], "risk_focus": [str]}'
        )
        result = await self._runner.run(_SYSTEM, user, PlannerOutput, role=ModelRole.SMALL)
        if not result.output.retrieval_plan.queries:
            # Degenerate plan: fall back to the normalized question as the only query.
            result.output.retrieval_plan.queries = [result.output.normalized_question or question]
        return result
