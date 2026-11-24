"""What a model can actually accept, as far as we have verified from vendor docs.

Only fields that change the shape of an outgoing request live here. Tool calling, token
streaming and reasoning-trace capture are deliberately absent: nothing in this application
consumes them, so they would be unread configuration. The verified per-provider matrix for
those lives in docs/providers.md instead.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.integrations.llm.base import ThinkingMode

# The output budget requested when a caller does not name one and the dialect's default models
# answer directly, without a hidden reasoning phase. This is the value every request used to
# carry unconditionally (it was ``TextRequest.max_tokens``'s hard-coded default), so leaving it
# as the field default keeps OpenAI's -- and the fake provider's -- behaviour byte-identical.
DIRECT_ANSWER_OUTPUT_TOKENS = 1_024

# The output budget requested when a dialect's default models reason before answering. A
# reasoning model spends this budget on a chain of thought *first* and returns the visible answer
# in what is left, so a budget sized for a direct answer is consumed entirely by reasoning and the
# response arrives with ``finish_reason="length"`` and an empty ``content`` -- see
# ``openai_provider._read_content`` for the failure this used to become.
#
# 49,152 is still conservative: every affected vendor's *own* default for its reasoning models is
# larger (DeepSeek documents 64K in thinking mode, Kimi 131,072 for ``kimi-k3``), and every
# dialect's accepted maximum is at least 128,000. It costs nothing to set generously --
# ``max_tokens`` is a ceiling on generation, not a committed spend; only tokens the model actually
# produces are billed. It was 16,384 until live runs outgrew it: once the agents read every
# supplied chunk whole (FR-017), the final decision on ``deepseek-flash`` spent all 16,384 in two
# of three long-context runs and was refused as truncated (DV-RX-04). The longest decision
# measured to succeed since then used 21,470 (DV-RX-04b); ``tests/unit/test_reasoning_budget.py``
# keeps the budget at least twice that, and within every vendor's ceiling.
REASONING_OUTPUT_TOKENS = 49_152

# The output budget requested when a reasoning dialect's request has thinking switched off and
# names no budget (PLV9-T03, PLV-D-033). The reasoning budget is then only room for a runaway: a
# thinking-disabled decision on ``deepseek-v4-pro`` once ran to all 49,152 tokens in five minutes
# and failed its run (PLV-DEF-022). The 26 thinking-disabled decision replies measured live used
# 499 to 1,952 tokens, so this is over four times the longest; ``test_provider_modes`` keeps it so.
THINKING_OFF_OUTPUT_TOKENS = 8_192


class StructuredOutputMode(StrEnum):
    NONE = "none"  # no server-side constraint; rely on the repair ladder
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    NVEXT_GUIDED_JSON = "nvext_guided_json"  # NVIDIA NIM: extra_body.nvext.guided_json


class MaxTokensParam(StrEnum):
    MAX_TOKENS = "max_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"


@dataclass(frozen=True)
class ModelCapabilities:
    """The two token numbers are different questions and must not be conflated.

    ``max_output_tokens`` is the **ceiling the vendor accepts** for this dialect's models: the
    clamp ``build_chat_payload`` applies so a caller can never ask for more than the endpoint
    will take. ``default_max_output_tokens`` is **what we choose to ask for** when the caller
    names no budget, which is every production call path -- no call site in ``app/`` passes
    ``max_tokens`` today. Collapsing them into one number is what produced the defect this
    separation fixes: a single 1,024 constant sized for a direct-answering model, applied to four
    providers whose shipped "powerful" defaults reason first and answer second.
    """

    structured_output: StructuredOutputMode = StructuredOutputMode.JSON_OBJECT
    supports_temperature: bool = True
    max_tokens_param: MaxTokensParam = MaxTokensParam.MAX_TOKENS
    max_output_tokens: int = 4_096
    default_max_output_tokens: int = DIRECT_ANSWER_OUTPUT_TOKENS
    # The thinking mode sent when a request names none, for a dialect whose endpoint takes
    # DeepSeek's thinking switch (PLV8-T01, PLV-D-026). None: the dialect has no switch, and its
    # requests carry no thinking field at all.
    default_mode: ThinkingMode | None = None
    # What we ask for when a request's mode has thinking disabled and it names no budget; None: the
    # default budget applies whatever the mode (PLV-D-033).
    direct_max_output_tokens: int | None = None
