"""Cost estimation against a versioned rate table (C-8, ADR-004).

A repository-wide search for ``price``, ``pricing``, ``cost_usd`` and ``cost_per`` over
``backend/app`` previously returned **zero matches**: there was no rate data anywhere, so no cost
could be computed at all.

The failure mode this module exists to prevent is specific. A model missing from the rate table
must **never** contribute ``0.0`` to a running total, because a free-looking attempt never trips a
budget and a spend ceiling that cannot fire is not a ceiling. Absence is reported as
``unpriced`` -- a distinct, visible state that a report has to carry rather than absorb.

Three further rules follow from that one:

* **Rates live in a dated, versioned file**, not in code. Every report that quotes a cost also
  quotes ``rate_table_version``, so a figure can be reproduced against the rates that produced it.
* **A total containing any unpriced attempt is a lower bound**, and says so. Silently omitting
  unpriced attempts from a sum produces a number that reads as "the cost" and is not.
* **Call and token ceilings are enforced independently of cost.** They need no rate data, so a
  deployment with an empty rate table still has a working budget -- it simply reports ``unpriced``
  throughout. That is the intended degraded mode if decision D-1 is never taken.
"""

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final, Literal

_RATES_PATH: Final = Path(__file__).with_name("pricing_rates.json")

# Rates are published per million tokens; metering records raw token counts.
_TOKENS_PER_RATE_UNIT: Final = Decimal(1_000_000)

PricingStatus = Literal["priced", "unpriced"]


@dataclass(frozen=True)
class ModelRate:
    """List price for one provider/model pair, in USD per million tokens."""

    prompt: Decimal
    completion: Decimal


@dataclass(frozen=True)
class CostEstimate:
    """The cost of one metering record, or an explicit statement that it has no price.

    ``amount_usd`` is ``None`` exactly when ``status`` is ``"unpriced"``. The two fields are
    deliberately both present: a consumer that reads only the amount cannot mistake ``None`` for
    a parse failure, and a consumer that reads only the status cannot silently coerce.
    """

    amount_usd: Decimal | None
    status: PricingStatus
    rate_table_version: str

    @property
    def is_priced(self) -> bool:
        return self.status == "priced"


class RateTable:
    """The loaded rate file. Immutable once constructed."""

    def __init__(self, version: str, rates: dict[str, dict[str, ModelRate]]) -> None:
        self._version = version
        self._rates = rates

    @property
    def version(self) -> str:
        return self._version

    @classmethod
    def load(cls, path: Path | None = None) -> "RateTable":
        raw = json.loads((path or _RATES_PATH).read_text(encoding="utf-8"))
        rates: dict[str, dict[str, ModelRate]] = {}
        for provider, models in raw["rates"].items():
            rates[provider] = {
                model: ModelRate(
                    # `Decimal(str(...))` rather than `Decimal(float)`: the literal 0.27 is not
                    # exactly representable in binary, and a cost total is a money figure that
                    # will be compared against an invoice.
                    prompt=Decimal(str(entry["prompt"])),
                    completion=Decimal(str(entry["completion"])),
                )
                for model, entry in models.items()
            }
        return cls(str(raw["rate_table_version"]), rates)

    def rate_for(self, provider: str, model: str) -> ModelRate | None:
        """The published rate, or ``None`` when this pair is not in the table.

        Matching is exact on both provider and model. No prefix matching and no fallback to
        another model of the same vendor: a near-miss that silently priced one model at another's
        rate would produce a plausible, wrong, unfalsifiable number -- worse than `unpriced`,
        which at least announces itself.
        """
        return self._rates.get(provider, {}).get(model)

    def estimate(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> CostEstimate:
        """Cost for one provider attempt.

        ``unpriced`` for two distinct reasons, both of which must not read as free:

        * the provider/model pair has no published rate here;
        * the vendor reported no usage at all (``None``), so there is nothing to multiply. A
          vendor-reported **zero** is different -- that is a measurement, and it prices to zero
          honestly.
        """
        rate = self.rate_for(provider, model)
        if rate is None or (prompt_tokens is None and completion_tokens is None):
            return CostEstimate(None, "unpriced", self._version)

        amount = (
            Decimal(prompt_tokens or 0) * rate.prompt
            + Decimal(completion_tokens or 0) * rate.completion
        ) / _TOKENS_PER_RATE_UNIT
        return CostEstimate(amount, "priced", self._version)


_table: RateTable | None = None


def rate_table() -> RateTable:
    """The process-wide rate table, loaded once.

    Not a module-level constant: reading a file at import time makes every importer pay for it
    and makes a malformed file an import error rather than a usable failure at the point of use.
    """
    global _table  # noqa: PLW0603 -- one process-wide cache, deliberately
    if _table is None:
        _table = RateTable.load()
    return _table


@dataclass(frozen=True)
class CostSummary:
    """A total over several attempts, plus how much of it is missing.

    ``is_lower_bound`` is the field that keeps the total honest: with any unpriced attempt in the
    set, ``total_usd`` is what the priced attempts cost and **not** what the run cost.
    """

    total_usd: Decimal
    priced_attempts: int
    unpriced_attempts: int
    rate_table_version: str

    @property
    def is_lower_bound(self) -> bool:
        return self.unpriced_attempts > 0


def summarize(estimates: list[CostEstimate]) -> CostSummary:
    """Sum priced estimates and count the unpriced ones separately."""
    table_version = estimates[0].rate_table_version if estimates else rate_table().version
    total = sum(
        (estimate.amount_usd for estimate in estimates if estimate.amount_usd is not None),
        Decimal(0),
    )
    priced = sum(1 for estimate in estimates if estimate.is_priced)
    return CostSummary(
        total_usd=total,
        priced_attempts=priced,
        unpriced_attempts=len(estimates) - priced,
        rate_table_version=table_version,
    )
