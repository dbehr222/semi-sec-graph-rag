"""
Ontology for the SEC filings knowledge-graph RAG project.

This file is the single source of truth for:
  1. what the extraction LLM is allowed to return (strict schema),
  2. what the validator accepts (types, allowed endpoint pairs, evidence check),
  3. what gets written to Neo4j (labels, edge types, edge properties),
  4. the definitions block pasted into the extraction prompt.

Design rules:
  - Closed vocabulary. 5 extracted entity types, 11 relationship types.
  - One direction per relationship. "A relies on B for wafers" is normalized to
    B -[SUPPLIER_OF]-> A. There is no CUSTOMER_OF type; customers are read by
    traversing SUPPLIER_OF backwards. Customer concentration is an edge property.
  - The LLM never sets chunk ids, fiscal years or accession numbers. Code adds
    those after validation (see `finalize`).
  - Every edge carries a verbatim evidence quote that must appear in the chunk.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
class EntityType(str, Enum):
    COMPANY = "COMPANY"
    PERSON = "PERSON"
    PRODUCT = "PRODUCT"
    GEOGRAPHY = "GEOGRAPHY"
    RISK_CATEGORY = "RISK_CATEGORY"
    # FILING is structural: the loader creates it from metadata.json, the LLM never emits it.


class RelType(str, Enum):
    SUPPLIER_OF = "SUPPLIER_OF"
    DISTRIBUTES_FOR = "DISTRIBUTES_FOR"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    ACQUIRED = "ACQUIRED"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    OFFERS = "OFFERS"
    OPERATES_IN = "OPERATES_IN"
    EXPOSED_TO_RISK = "EXPOSED_TO_RISK"
    DIRECTOR_OF = "DIRECTOR_OF"
    OFFICER_OF = "OFFICER_OF"


class SupplyType(str, Enum):
    """What the supplier provides. Only valid on SUPPLIER_OF."""
    FOUNDRY = "FOUNDRY"                    # wafer fabrication
    ASSEMBLY_TEST = "ASSEMBLY_TEST"        # packaging, OSAT
    EQUIPMENT = "EQUIPMENT"                # fab / test equipment
    MATERIALS = "MATERIALS"                # wafers, chemicals, gases, substrates
    COMPONENTS = "COMPONENTS"              # memory, substrates, modules sold into a product
    IP_LICENSE = "IP_LICENSE"              # licensed cores / IP blocks
    EDA_SOFTWARE = "EDA_SOFTWARE"
    FINISHED_PRODUCTS = "FINISHED_PRODUCTS"  # chips or systems sold to a customer
    OTHER = "OTHER"


class Activity(str, Enum):
    """Kind of presence in a place. Only valid on OPERATES_IN."""
    HEADQUARTERS = "HEADQUARTERS"
    MANUFACTURING = "MANUFACTURING"
    SALES = "SALES"
    RND = "RND"
    OTHER = "OTHER"


class RiskCategoryName(str, Enum):
    """Controlled risk vocabulary. A closed list is what makes cross-company
    risk comparison queries possible. Free-text risk labels would not match."""
    EXPORT_CONTROLS = "EXPORT_CONTROLS"
    CHINA_GEOPOLITICAL = "CHINA_GEOPOLITICAL"
    TAIWAN_CONCENTRATION = "TAIWAN_CONCENTRATION"
    TARIFFS_TRADE = "TARIFFS_TRADE"
    CUSTOMER_CONCENTRATION = "CUSTOMER_CONCENTRATION"
    SUPPLIER_CONCENTRATION = "SUPPLIER_CONCENTRATION"   # single or limited source
    CAPACITY_SUPPLY_CONSTRAINT = "CAPACITY_SUPPLY_CONSTRAINT"
    CYCLICALITY_INVENTORY = "CYCLICALITY_INVENTORY"
    AI_DEMAND_DEPENDENCE = "AI_DEMAND_DEPENDENCE"
    IP_LITIGATION = "IP_LITIGATION"
    CYBERSECURITY = "CYBERSECURITY"
    NATURAL_DISASTER_CLIMATE = "NATURAL_DISASTER_CLIMATE"
    REGULATORY_ANTITRUST = "REGULATORY_ANTITRUST"
    DEBT_LIQUIDITY = "DEBT_LIQUIDITY"
    TALENT_RETENTION = "TALENT_RETENTION"
    FX_INTEREST_RATES = "FX_INTEREST_RATES"


RISK_CATEGORY_NAMES = {r.value for r in RiskCategoryName}

E, R = EntityType, RelType

# (source_type, target_type) pairs allowed for each relationship type.
ALLOWED_ENDPOINTS: dict[RelType, set[tuple[EntityType, EntityType]]] = {
    R.SUPPLIER_OF: {(E.COMPANY, E.COMPANY)},
    R.DISTRIBUTES_FOR: {(E.COMPANY, E.COMPANY)},
    R.COMPETES_WITH: {(E.COMPANY, E.COMPANY)},
    R.PARTNERS_WITH: {(E.COMPANY, E.COMPANY)},
    R.ACQUIRED: {(E.COMPANY, E.COMPANY)},
    R.SUBSIDIARY_OF: {(E.COMPANY, E.COMPANY)},
    R.OFFERS: {(E.COMPANY, E.PRODUCT)},
    R.OPERATES_IN: {(E.COMPANY, E.GEOGRAPHY)},
    R.EXPOSED_TO_RISK: {(E.COMPANY, E.RISK_CATEGORY)},
    R.DIRECTOR_OF: {(E.PERSON, E.COMPANY)},
    R.OFFICER_OF: {(E.PERSON, E.COMPANY)},
}

# Relationships where the edge is "A -> B" in a specific direction. Used in the
# prompt so the model normalizes direction instead of inventing inverse types.
DEFINITIONS: dict[RelType, dict[str, str]] = {
    R.SUPPLIER_OF: {
        "meaning": "Source provides goods, services or manufacturing to target. Source is the supplier, target is the customer.",
        "example": "'We rely on TSMC to fabricate our GPUs' -> TSMC SUPPLIER_OF filer, supply_type=FOUNDRY",
        "attrs": "supply_type (required). revenue_share_pct if the text gives the customer's share of the supplier's revenue. is_sole_source=true only if the text says sole or single source.",
    },
    R.DISTRIBUTES_FOR: {
        "meaning": "Source is a distributor or channel partner that resells target's products.",
        "example": "'We sell through Arrow and Avnet' -> Arrow DISTRIBUTES_FOR filer; Avnet DISTRIBUTES_FOR filer",
        "attrs": "revenue_share_pct if given.",
    },
    R.COMPETES_WITH: {
        "meaning": "Source names target as a competitor. Record only competitors the filing explicitly names.",
        "example": "'We compete with Intel and AMD' -> filer COMPETES_WITH Intel; filer COMPETES_WITH AMD",
        "attrs": "none.",
    },
    R.PARTNERS_WITH: {
        "meaning": "Joint venture, strategic alliance, co-development or licensing collaboration that is not a plain supply relationship.",
        "example": "'Our joint venture with X' -> filer PARTNERS_WITH X",
        "attrs": "none.",
    },
    R.ACQUIRED: {
        "meaning": "Source acquired target (completed or announced). Source is the acquirer.",
        "example": "'We acquired Xilinx in February 2022' -> filer ACQUIRED Xilinx",
        "attrs": "none.",
    },
    R.SUBSIDIARY_OF: {
        "meaning": "Source is a subsidiary or majority-owned affiliate of target. Source is the child.",
        "example": "'NVIDIA Singapore Pte. Ltd. is our subsidiary' -> NVIDIA Singapore SUBSIDIARY_OF filer",
        "attrs": "none.",
    },
    R.OFFERS: {
        "meaning": "Company sells or develops a named product family or product line.",
        "example": "'Our data center products include H100' -> filer OFFERS H100",
        "attrs": "none. Product names only, not generic words like 'chips'.",
    },
    R.OPERATES_IN: {
        "meaning": "Company has headquarters, manufacturing, R&D or significant sales in a country or region.",
        "example": "'Our wafers are fabricated in Taiwan' -> supplier is a separate edge; this edge is only for the filer's own presence",
        "attrs": "activity (required).",
    },
    R.EXPOSED_TO_RISK: {
        "meaning": "Company discloses a risk that fits one controlled risk category. Target must be one of the RiskCategoryName values.",
        "example": "'US export restrictions could reduce our China sales' -> filer EXPOSED_TO_RISK EXPORT_CONTROLS",
        "attrs": "none.",
    },
    R.DIRECTOR_OF: {
        "meaning": "Person sits on the target company's board.",
        "example": "'John Doe has served as a director since 2019' -> John Doe DIRECTOR_OF filer",
        "attrs": "title optional.",
    },
    R.OFFICER_OF: {
        "meaning": "Person is an executive officer of the target company.",
        "example": "'Jane Roe, Chief Financial Officer' -> Jane Roe OFFICER_OF filer",
        "attrs": "title optional.",
    },
}


# --------------------------------------------------------------------------
# Name handling
# --------------------------------------------------------------------------
# Words that mean "the filer" and must never become entity names. The prompt
# passes the real filer name; the model has to use it.
BANNED_NAMES = {
    "we", "us", "our", "the company", "the registrant", "company", "registrant",
    "the issuer", "it", "its", "customer", "supplier", "competitor", "third party",
}

_LEGAL_SUFFIXES = (
    r"incorporated|inc|corporation|corp|company|co|limited|ltd|plc|llc|lp|"
    r"nv|n\.v|sa|ag|gmbh|holdings|holding|group"
)


def normalize_name(name: str) -> str:
    """Cheap first-pass normalization for entity resolution. Embedding similarity
    and CIK matching come after this. 'Acme Corp.' / 'ACME Corporation' -> 'acme'."""
    s = name.lower()
    s = re.sub(r"[.,'\"()]", " ", s)
    s = re.sub(r"&", " and ", s)
    s = re.sub(rf"\b({_LEGAL_SUFFIXES})\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# --------------------------------------------------------------------------
# What the LLM returns (strict)
# --------------------------------------------------------------------------
class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EntityType
    canonical_name: str = Field(min_length=2, max_length=120,
                                description="Full formal name, e.g. 'Taiwan Semiconductor Manufacturing Company', not 'TSMC' or 'the foundry'.")
    mentioned_as: list[str] = Field(default_factory=list,
                                    description="Other surface forms seen in this chunk, e.g. ['TSMC']. Becomes the alias list.")
    is_anonymous: bool = Field(default=False,
                               description="True for unnamed parties such as 'Customer A'. Use canonical_name 'Customer A' as written. Never guess who it is.")

    @field_validator("canonical_name")
    @classmethod
    def _no_pronouns(cls, v: str) -> str:
        if _squash(v) in BANNED_NAMES:
            raise ValueError(f"'{v}' is not a valid entity name. Use the real name.")
        return v.strip()

    @model_validator(mode="after")
    def _type_rules(self) -> "ExtractedEntity":
        if self.type == EntityType.RISK_CATEGORY and self.canonical_name not in RISK_CATEGORY_NAMES:
            raise ValueError(f"RISK_CATEGORY name must be one of {sorted(RISK_CATEGORY_NAMES)}")
        if self.is_anonymous and self.type != EntityType.COMPANY:
            raise ValueError("Only COMPANY entities can be anonymous.")
        return self


class ExtractedRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="canonical_name of the source entity, exactly as in entities.")
    source_type: EntityType
    rel_type: RelType
    target: str = Field(description="canonical_name of the target entity, exactly as in entities.")
    target_type: EntityType
    evidence: str = Field(min_length=10, max_length=300,
                          description="Short verbatim quote from the chunk that supports this edge. Copy it exactly.")
    confidence: float = Field(ge=0.0, le=1.0)

    # Edge attributes. Each is only valid on specific relationship types.
    supply_type: Optional[SupplyType] = None            # SUPPLIER_OF
    is_sole_source: Optional[bool] = None               # SUPPLIER_OF
    revenue_share_pct: Optional[float] = Field(default=None, ge=0, le=100)  # SUPPLIER_OF, DISTRIBUTES_FOR
    activity: Optional[Activity] = None                 # OPERATES_IN
    title: Optional[str] = Field(default=None, max_length=100)  # DIRECTOR_OF, OFFICER_OF

    @model_validator(mode="after")
    def _check_edge(self) -> "ExtractedRelationship":
        if (self.source_type, self.target_type) not in ALLOWED_ENDPOINTS[self.rel_type]:
            raise ValueError(
                f"{self.rel_type.value} cannot connect {self.source_type.value} -> {self.target_type.value}"
            )
        if _squash(self.source) == _squash(self.target):
            raise ValueError("Self-loop.")
        if _squash(self.source) in BANNED_NAMES or _squash(self.target) in BANNED_NAMES:
            raise ValueError("Edge endpoint is a pronoun or generic word. Use the real name.")

        rt = self.rel_type
        if rt == R.SUPPLIER_OF and self.supply_type is None:
            raise ValueError("SUPPLIER_OF requires supply_type.")
        if rt == R.OPERATES_IN and self.activity is None:
            raise ValueError("OPERATES_IN requires activity.")
        if rt != R.SUPPLIER_OF and (self.supply_type is not None or self.is_sole_source is not None):
            raise ValueError("supply_type / is_sole_source are only valid on SUPPLIER_OF.")
        if rt not in (R.SUPPLIER_OF, R.DISTRIBUTES_FOR) and self.revenue_share_pct is not None:
            raise ValueError("revenue_share_pct is only valid on SUPPLIER_OF / DISTRIBUTES_FOR.")
        if rt != R.OPERATES_IN and self.activity is not None:
            raise ValueError("activity is only valid on OPERATES_IN.")
        if rt not in (R.DIRECTOR_OF, R.OFFICER_OF) and self.title is not None:
            raise ValueError("title is only valid on DIRECTOR_OF / OFFICER_OF.")
        return self


class LLMExtraction(BaseModel):
    """Exact shape Claude must return for one chunk. Use `LLMExtraction.model_json_schema()`
    as the tool input_schema and validate the tool call with `LLMExtraction.model_validate`."""
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def _closed_world(self) -> "LLMExtraction":
        known = {(e.type, _squash(e.canonical_name)) for e in self.entities}
        for r in self.relationships:
            if (r.source_type, _squash(r.source)) not in known:
                raise ValueError(f"Relationship source '{r.source}' ({r.source_type.value}) is not in entities.")
            if (r.target_type, _squash(r.target)) not in known:
                raise ValueError(f"Relationship target '{r.target}' ({r.target_type.value}) is not in entities.")
        return self


# --------------------------------------------------------------------------
# What gets stored (LLM output + provenance added by code)
# --------------------------------------------------------------------------
class ChunkRef(BaseModel):
    chunk_id: str
    cik: str
    accession: str
    fiscal_year: int
    section_path: str
    text: str


class StoredRelationship(ExtractedRelationship):
    source_chunk_id: str
    accession: str
    cik: str
    fiscal_year: int


def check_evidence(result: LLMExtraction, chunk_text: str) -> list[str]:
    """Return a list of problems. Each evidence quote must appear verbatim
    (whitespace and case-insensitive) in the chunk. Failures go back to the
    model for a retry, or the edge is dropped. This is the anti-hallucination gate."""
    haystack = _squash(chunk_text)
    return [
        f"Evidence not found in chunk for {r.source} -{r.rel_type.value}-> {r.target}: '{r.evidence[:80]}'"
        for r in result.relationships
        if _squash(r.evidence) not in haystack
    ]


def finalize(result: LLMExtraction, chunk: ChunkRef) -> list[StoredRelationship]:
    """Attach provenance. Call only after schema validation and check_evidence pass."""
    return [
        StoredRelationship(
            **r.model_dump(),
            source_chunk_id=chunk.chunk_id,
            accession=chunk.accession,
            cik=chunk.cik,
            fiscal_year=chunk.fiscal_year,
        )
        for r in result.relationships
    ]


# --------------------------------------------------------------------------
# Prompt block and Neo4j mapping
# --------------------------------------------------------------------------
def build_ontology_prompt() -> str:
    """Paste this into the extraction system prompt so the prompt and the schema
    can never drift apart."""
    lines = ["ENTITY TYPES: " + ", ".join(e.value for e in EntityType), ""]
    lines.append("RISK CATEGORIES (the only allowed RISK_CATEGORY names): " + ", ".join(sorted(RISK_CATEGORY_NAMES)))
    lines.append("")
    lines.append("RELATIONSHIP TYPES (one direction only; never invent inverses):")
    for rt, d in DEFINITIONS.items():
        pairs = ", ".join(f"{s.value}->{t.value}" for s, t in sorted(ALLOWED_ENDPOINTS[rt], key=str))
        lines += [
            f"- {rt.value} ({pairs}): {d['meaning']}",
            f"    Example: {d['example']}",
            f"    Attributes: {d['attrs']}",
        ]
    lines += [
        "",
        "RULES:",
        "- Use the filer's real name (given in the prompt) wherever the text says 'we', 'us', 'our' or 'the Company'.",
        "- Use full formal names for canonical_name and put short forms in mentioned_as.",
        "- Unnamed parties ('Customer A', 'a large hyperscaler') become COMPANY with is_anonymous=true. Never guess their identity.",
        "- Only extract relationships the text states. Do not infer from general knowledge.",
        "- evidence must be a short exact quote from the chunk.",
        "- If nothing in the chunk fits, return empty lists.",
    ]
    return "\n".join(lines)


NEO4J_LABELS = {
    EntityType.COMPANY: "Company",
    EntityType.PERSON: "Person",
    EntityType.PRODUCT: "Product",
    EntityType.GEOGRAPHY: "Geography",
    EntityType.RISK_CATEGORY: "RiskCategory",
}
# Structural node, created by the loader (not the LLM): (:Filing {accession, cik, fiscal_year, form}).
# (:Chunk {chunk_id}) nodes are optional; chunk_id as an edge property is enough for citations.

# MERGE keys (entity resolution output decides the value):
#   Company:      cik when the company is a filer, otherwise normalized canonical name
#   Person:       normalized name + company cik
#   Product:      normalized name + owning company
#   Geography:    normalized name
#   RiskCategory: name (fixed vocabulary, seed once)
# Edge MERGE key: (source, rel_type, target, fiscal_year). On match, append source_chunk_id to
# a `source_chunk_ids` list and keep the max confidence, so re-running ingestion is idempotent.


if __name__ == "__main__":
    print(build_ontology_prompt())