"""
Pydantic v2 schemas for Brahman 2.0 Reasoning Oracle.
Strict type-safety to prevent injection attacks into the Panini engine.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from enum import Enum
import re


class KarakaRole(str, Enum):
    KARTR = "Kartr"
    KARMAN = "Karman"
    KARANA = "Karana"
    SAMPRADANA = "Sampradana"
    APADANA = "Apadana"
    ADHIKARANA = "Adhikarana"


class VibhaktiCase(str, Enum):
    NOMINATIVE = "nominative"
    ACCUSATIVE = "accusative"
    INSTRUMENTAL = "instrumental"
    DATIVE = "dative"
    ABLATIVE = "ablative"
    GENITIVE = "genitive"
    LOCATIVE = "locative"
    VOCATIVE = "vocative"


# ── Request Schemas ──────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """Input for the /v1/verify endpoint."""
    sentence: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="A Sanskrit sentence in Devanagari script to verify."
    )

    @field_validator("sentence")
    @classmethod
    def sanitize_sentence(cls, v: str) -> str:
        # Strip control characters but allow Devanagari, spaces, punctuation
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', v)
        if not cleaned.strip():
            raise ValueError("Sentence must contain visible characters.")
        return cleaned.strip()


class SolnetValidateRequest(BaseModel):
    """Input for the /v1/solnet/validate endpoint."""
    transaction_intent: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The transaction intent string to validate for logical consistency."
    )
    node_id: Optional[str] = Field(
        None,
        max_length=64,
        description="Optional SOLNET node identifier for audit trail."
    )


# ── Response Schemas ─────────────────────────────────────────────────────

class WordAnalysis(BaseModel):
    """Morphological analysis of a single word."""
    original: str
    pos: Optional[str] = None
    pratipadika: Optional[str] = None
    dhatu: Optional[str] = None
    vibhakti: Optional[List[str]] = None
    gender: Optional[str] = None
    number: Optional[str] = None
    tense: Optional[str] = None
    person: Optional[str] = None
    neural_karaka: Optional[str] = None
    karaka_valid: Optional[bool] = None


class VerifyResponse(BaseModel):
    """Output of the /v1/verify endpoint."""
    sentence: str
    status: str
    word_analyses: List[WordAnalysis]
    violations: List[str]
    is_logically_consistent: bool


class KarakaTrace(BaseModel):
    """A single entry in the Kāraka audit trace."""
    word: str
    predicted_karaka: str
    allowed_vibhaktis: List[str]
    detected_vibhaktis: List[str]
    is_valid: bool


class VerificationCertificate(BaseModel):
    """The SOLNET Oracle verification certificate."""
    is_logically_consistent: bool
    karaka_trace: List[KarakaTrace]
    logic_hash: str = Field(
        ...,
        description="SHA-256 signature of the verified linguistic trace."
    )
    violations: List[str]
    node_id: Optional[str] = None


class LinguisticViolationError(BaseModel):
    """Error body for 422 responses."""
    error: str = "LinguisticViolation"
    detail: str
    violations: List[str]
    karaka_trace: List[KarakaTrace]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    dhatus_loaded: int
    device: str
    model_loaded: bool
