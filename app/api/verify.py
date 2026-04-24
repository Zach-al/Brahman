"""
/v1/verify — Sanskrit sentence verification endpoint.
"""
from fastapi import APIRouter, HTTPException

from app.schemas.pydantic_models import (
    VerifyRequest, VerifyResponse, LinguisticViolationError
)

router = APIRouter(prefix="/v1", tags=["verify"])


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify a Sanskrit sentence against Pāṇinian logic",
    responses={
        422: {
            "model": LinguisticViolationError,
            "description": "The sentence contains a linguistic violation."
        }
    }
)
async def verify_sentence(request: VerifyRequest):
    """
    Run a Sanskrit sentence through the full Anvaya-Bodha pipeline.
    
    The Symbolic Core deterministically parses morphology, the Neural Bridge
    proposes semantic roles, and the Gatekeeper validates the combination.
    
    Returns 422 if a Linguistic Violation is detected.
    """
    from app.main import oracle  # Lazy import to avoid circular dependency

    result = oracle.verify_sentence(request.sentence)

    if not result.is_logically_consistent:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "LinguisticViolation",
                "detail": f"Sentence '{request.sentence}' failed Anvaya-Bodha verification.",
                "violations": result.violations,
                "word_analyses": [wa.model_dump() for wa in result.word_analyses]
            }
        )

    return result
