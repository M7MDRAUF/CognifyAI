"""Application settings loaded from environment variables.

Startup must fail fast when a required variable is missing or invalid,
so every field without a default below is mandatory (SDD section 12.2).
Values come from the process environment, with an optional ``.env`` file
as fallback; the environment always wins.
"""

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from app.core.logging import redact_url_credentials

VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

DEFAULT_ENV_FILE = Path(".env")


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LLMProviderName(StrEnum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    NVIDIA = "nvidia"
    GLM = "glm"
    KIMI = "kimi"
    FAKE = "fake"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_env: AppEnv
    app_log_level: str
    database_url: str
    qdrant_url: str
    qdrant_collection_name: str
    llm_provider: LLMProviderName
    llm_api_key: SecretStr
    llm_text_model_small: str
    llm_text_model_strong: str
    llm_embedding_model: str
    llm_embedding_dimensions: int = Field(gt=0)
    # The advertised per-file cap, and the one number every expansion ceiling behind it is
    # derived from: ``app.core.security`` bounds what an archive may decompress to,
    # ``app.integrations.parsers.tabular`` bounds the materialized grid, and
    # ``app.services.chunking_service`` bounds the chunks one document may buy. Raising this
    # therefore raises those in step -- which is the point: they used to be hardcoded against a
    # 25 MB deployment, so an operator who raised this started getting ordinary documents
    # refused by a ceiling that had not moved.
    #
    # The upper bound is new and exists because of that same derivation. Peak memory for one
    # document scales linearly from here (a full-size CSV materializes roughly 12 bytes of
    # Python objects per uploaded byte before it is chunked), so an unbounded setting is an
    # unbounded allocation in the request path. At the maximum below, one document may
    # materialize a ~6 GB grid and buy ~3,500 embedding requests; past that the setting is
    # almost certainly a typo, and a typo here is an out-of-memory kill rather than a refusal.
    max_upload_size_mb: int = Field(gt=0, le=512)
    max_project_size_mb: int = Field(default=250, gt=0)
    # Admission control for the query workflow. One accepted POST /projects/{id}/queries
    # commits a run that issues 7-20 billed vendor chat completions, so both of these bound
    # what a single (anonymous) caller can spend -- see `QueryService.create` and
    # `app.workers.jobs._execute_workflow`.
    max_running_queries_per_project: int = Field(default=3, gt=0)
    query_run_deadline_seconds: int = Field(default=600, gt=0)
    # A running run's lease (RMD5-T01): its job renews it every third of this, and the reconciler
    # closes a run whose lease has lapsed with no live task in its own process. A run therefore
    # ends at most this plus one reconcile interval after its process is gone.
    query_run_lease_seconds: int = Field(default=60, ge=3)
    run_reconcile_interval_seconds: int = Field(default=30, gt=0)
    # A document's ingestion lease (RMD5-T02), renewed the same way; a PROCESSING document with no
    # lease is closed once its last update is older than `ingestion_stale_after_seconds`.
    ingestion_lease_seconds: int = Field(default=60, ge=3)
    ingestion_stale_after_seconds: int = Field(default=900, gt=0)
    # The one global top-K applied to the final context (D-9, SDD 9.4.1). A setting rather than
    # a constant so a deployment can change it and the value is recorded in the run's trace --
    # a baseline measured at one K is not comparable with one measured at another, so the number
    # has to be visible rather than compiled in.
    final_context_size: int = Field(default=8, gt=0, le=64)
    # Rollout gates (plan section 26; P8-T09). Each switches one high-risk change off without a
    # revert, and both default on: the behaviour every baseline was measured under.
    #
    # Off, `rerank_expanded_union` restores per-round reranking (C-10's prior behaviour: round two
    # is reranked on its own and pooled uncut, so an expanded run can hand up to 2K chunks on).
    # Off, `routing_profile_requests_enabled` refuses a request that names a routing profile; a
    # request that names none runs on its `answer_model`, the path that predates the field.
    rerank_expanded_union: bool = True
    routing_profile_requests_enabled: bool = True
    # Off (`PHARMA_PROFILE_ENABLED=false`), creating a pharmaceutical manufacturing project is
    # refused with `domain_profile_disabled` (400; REQ-PRF-04). General creation and every
    # existing project, pharmaceutical ones included, keep working.
    pharma_profile_enabled: bool = True
    upload_dir: str = "./data/uploads"
    # FR-012: one origin, or several separated by commas (`cors_allowed_origins`). The default is
    # both spellings of the loopback frontend: the documented start command prints the 127.0.0.1
    # one, and an allow-list of `localhost` alone broke every API call made from it.
    cors_allowed_origin: str = "http://localhost:3000,http://127.0.0.1:3000"
    llm_base_url: str | None = None

    # Per-provider configuration (all optional). Each provider gets an API key, base URL and
    # fast/powerful model override, read from LLM_<PROVIDER>_* env vars mechanically derived by
    # `_collect_raw_settings` below — no loader change needed to add these. The `LLM_` prefix on
    # every one is deliberate: `openai_api_key` would read `OPENAI_API_KEY`, the variable the
    # OpenAI SDK itself falls back to, which is a real credential-confusion risk.
    llm_openai_api_key: SecretStr | None = None
    llm_openai_base_url: str | None = None
    llm_openai_model_fast: str | None = None
    llm_openai_model_powerful: str | None = None
    llm_deepseek_api_key: SecretStr | None = None
    llm_deepseek_base_url: str | None = None
    llm_deepseek_model_fast: str | None = None
    llm_deepseek_model_powerful: str | None = None
    llm_nvidia_api_key: SecretStr | None = None
    llm_nvidia_base_url: str | None = None
    llm_nvidia_model_fast: str | None = None
    llm_nvidia_model_powerful: str | None = None
    llm_glm_api_key: SecretStr | None = None
    llm_glm_base_url: str | None = None
    llm_glm_model_fast: str | None = None
    llm_glm_model_powerful: str | None = None
    llm_kimi_api_key: SecretStr | None = None
    llm_kimi_base_url: str | None = None
    llm_kimi_model_fast: str | None = None
    llm_kimi_model_powerful: str | None = None
    llm_enabled_providers: str | None = None  # optional comma-separated allow-list
    # The providers that may embed document text, in the same form; unset or blank, the chat list
    # above applies (`registry.embedding_allow_list`).
    llm_enabled_embedding_providers: str | None = None
    llm_default_model: str | None = None  # an option id, e.g. "openai:gpt-4o"
    # Embeddings are pinned to their own provider, independent of the chat provider LLM_PROVIDER
    # names — see `app.integrations.llm.embeddings` for why only a verified subset is accepted.
    llm_embedding_provider: LLMProviderName = LLMProviderName.OPENAI

    @field_validator(
        "llm_base_url",
        "llm_openai_base_url",
        "llm_deepseek_base_url",
        "llm_nvidia_base_url",
        "llm_glm_base_url",
        "llm_kimi_base_url",
    )
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        """Reject a base URL that is not an absolute http(s) endpoint.

        These values decide where a provider credential is *sent*, and nothing else validated
        them: `file://`, `ftp://` and bare non-URLs were all accepted, and a typo produced a
        confusing SDK error at first use rather than a clear refusal at boot.

        This could not be added until `load_settings` redacted its own failures. Pydantic echoes
        the rejected input verbatim into `str(ValidationError)` -- confirmed, not assumed -- and
        this message is raised at startup where uvicorn logs it. Since the documented self-host
        escape hatch lets an operator put credentials in the URL, a validator that printed what
        it rejected would leak exactly the value it exists to police. Blank is left to
        `provider_setting`, which already treats it as unset.
        """
        if value is None or value.strip() == "":
            return value
        stripped = value.strip()
        if not stripped.startswith(("http://", "https://")):
            msg = "must be an absolute http:// or https:// URL"
            raise ValueError(msg)
        return stripped

    @field_validator(
        "database_url",
        "qdrant_url",
        "qdrant_collection_name",
        "llm_text_model_small",
        "llm_text_model_strong",
        "llm_embedding_model",
    )
    @classmethod
    def refuse_blank(cls, value: str) -> str:
        """Refuse a required value that is empty after stripping (FR-008).

        Each names a resource the application cannot work without, and a blank one used to boot
        and fail later and further away: a blank DSN as a driver error on first use, a blank model
        id sent to the vendor. The message names the reason only; `load_settings` names the field
        and redacts every value. A real value is kept exactly as written.
        """
        if value.strip() == "":
            msg = "must not be blank"
            raise ValueError(msg)
        return value

    @field_validator("app_log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in VALID_LOG_LEVELS:
            msg = f"APP_LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}, got {value!r}"
            raise ValueError(msg)
        return upper

    @property
    def cors_allowed_origins(self) -> tuple[str, ...]:
        """The origins `cors_allowed_origin` lists, each stripped, in order, blanks dropped."""
        return tuple(
            origin.strip() for origin in self.cors_allowed_origin.split(",") if origin.strip()
        )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def max_project_size_bytes(self) -> int:
        return self.max_project_size_mb * 1024 * 1024

    # There used to be an `llm_is_live` property here (`llm_provider is not FAKE and
    # llm_api_key is non-blank`). It depended only on the legacy `LLM_API_KEY`, so a deployment
    # with that field blank and a real `LLM_OPENAI_API_KEY` (or any other per-provider key) still
    # resolved a working provider and billed it -- while this property reported `False`. Whether
    # *any* provider actually resolves can only be answered from `available_models`, which
    # requires importing the provider dialects -- an import this module must never make (the
    # dialects themselves import `LLMProviderName` *from* here, so the reverse would be
    # circular). That decision now lives in `app.integrations.llm.registry.is_live` (and
    # `LLMRouter.is_live`); `/llm/config` and `/health` both ask it instead.

    def provider_setting(self, provider: LLMProviderName, suffix: str) -> str | None:
        """Read a per-provider field such as ("deepseek", "base_url") without importing dialects.

        Returns None when the field is unset or blank. Secret fields are unwrapped here and
        nowhere else.
        """
        value: object = getattr(self, f"llm_{provider.value}_{suffix}", None)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped if stripped else None


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE dotenv file. Comments and blanks are skipped."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _collect_raw_settings(env_file: Path | None) -> dict[str, str]:
    file_values = parse_env_file(env_file) if env_file is not None else {}
    collected: dict[str, str] = {}
    for field_name in Settings.model_fields:
        env_name = field_name.upper()
        if env_name in file_values:
            collected[field_name] = file_values[env_name]
        if env_name in os.environ:
            collected[field_name] = os.environ[env_name]
    return collected


def _describe_without_values(exc: ValidationError) -> str:
    """Field names and reasons, never the rejected values.

    Rebuilt from ``exc.errors()`` rather than scrubbed out of ``str(exc)``. Running a generic
    secret scrubber over the rendered text was tried and is wrong twice over: it masks anything
    long, so ``llm_embedding_dimensions`` came back as ``[redacted]`` and the message stopped
    naming the field an operator has to fix -- and it would still be a denylist guessing at which
    substrings are sensitive, on a string that contains the value by construction. Omitting the
    value entirely is the only version that cannot be wrong about it.

    ``msg`` still passes through ``redact_url_credentials`` as defence in depth: a validator is
    free to interpolate the offending value into its own message (``validate_log_level`` does),
    and the one field where that value is realistically credential-bearing is a base URL.
    """
    lines = ["Invalid configuration:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"  {location}: {redact_url_credentials(error['msg'])}")
    return "\n".join(lines)


class SettingsValidationError(RuntimeError):
    """An invalid configuration, carrying a message with every value redacted."""


def load_settings(env_file: Path | None = DEFAULT_ENV_FILE) -> Settings:
    """Build Settings from the environment; raises on an incomplete or invalid configuration.

    The failure is re-raised with its text redacted, because pydantic includes the rejected input
    in the message (``input_value='...'``) and this runs during startup, where uvicorn writes the
    whole traceback to a log that is usually aggregated and long-retained. Several settings here
    are credential-bearing in normal use -- `DATABASE_URL` embeds a password, and the documented
    self-hosted escape hatch lets a provider base URL carry `user:password@` -- so an invalid one
    would put the credential in that log at the moment the operator is least able to notice.

    ``SettingsValidationError`` rather than the original ``ValidationError``: the structured
    ``errors()`` list carries the same raw input in a field this cannot rewrite, so letting the
    original escape would leave the leak one ``exc.errors()`` call away from returning. The field
    names and the reason survive; only the values are masked.
    """
    try:
        return Settings.model_validate(_collect_raw_settings(env_file))
    except ValidationError as exc:
        raise SettingsValidationError(_describe_without_values(exc)) from None


def read_database_url(env_file: Path | None = DEFAULT_ENV_FILE) -> str:
    """Resolve DATABASE_URL only (used by Alembic without full settings)."""
    url = os.environ.get("DATABASE_URL")
    if url is None and env_file is not None:
        url = parse_env_file(env_file).get("DATABASE_URL")
    if url is None:
        msg = "DATABASE_URL must be set in the environment or .env file."
        raise RuntimeError(msg)
    return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def upload_cap_mb() -> int:
    """The advertised per-file upload cap, for the expansion ceilings derived from it.

    Read through a function rather than captured in a module constant at import time, for two
    reasons. The modules that derive from it (``parsers.tabular``, ``services.chunking_service``)
    are imported by tooling that never loads a configuration, so an import-time read would turn a
    missing environment variable into an import error far from where it matters. And a test --
    or an operator reload -- that changes the cap must change the ceilings with it; captured at
    import, the two would silently disagree, which is the exact failure this derivation exists to
    remove.

    It is deliberately the only thing those modules take from settings: each one owns its own
    expansion factor, measured for what it expands, and states it next to the code that spends it.
    """
    return get_settings().max_upload_size_mb
