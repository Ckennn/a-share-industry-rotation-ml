"""Public Shenwan first-level continuity sample."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IndustryRecord:
    code: str
    name: str


CONTINUITY_SAMPLE = (
    IndustryRecord("801010", "Agriculture, Forestry, Animal Husbandry and Fishery"),
    IndustryRecord("801030", "Basic Chemicals"),
    IndustryRecord("801040", "Steel"),
    IndustryRecord("801050", "Non-ferrous Metals"),
    IndustryRecord("801080", "Electronics"),
    IndustryRecord("801110", "Household Appliances"),
    IndustryRecord("801120", "Food and Beverage"),
    IndustryRecord("801130", "Textiles and Apparel"),
    IndustryRecord("801140", "Light Manufacturing"),
    IndustryRecord("801150", "Pharmaceuticals and Biotechnology"),
    IndustryRecord("801160", "Utilities"),
    IndustryRecord("801170", "Transportation"),
    IndustryRecord("801180", "Real Estate"),
    IndustryRecord("801200", "Retail and Trade"),
    IndustryRecord("801210", "Social Services"),
    IndustryRecord("801230", "Conglomerates"),
    IndustryRecord("801710", "Building Materials"),
    IndustryRecord("801720", "Construction and Decoration"),
    IndustryRecord("801730", "Electrical Equipment"),
    IndustryRecord("801740", "Defense"),
    IndustryRecord("801750", "Computers"),
    IndustryRecord("801760", "Media"),
    IndustryRecord("801770", "Communications"),
    IndustryRecord("801780", "Banks"),
    IndustryRecord("801790", "Non-bank Financials"),
    IndustryRecord("801880", "Automobiles"),
    IndustryRecord("801890", "Machinery"),
)


def continuity_codes() -> tuple[str, ...]:
    return tuple(record.code for record in CONTINUITY_SAMPLE)


def continuity_names() -> dict[str, str]:
    return {record.code: record.name for record in CONTINUITY_SAMPLE}
