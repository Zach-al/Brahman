"""
/v1/solnet — SOLNET Oracle Bridge endpoints.
Provides Verification Certificates for decentralized intent validation.
"""
from fastapi import APIRouter, HTTPException, Depends

from app.schemas.pydantic_models import (
    SolnetValidateRequest, VerificationCertificate, LinguisticViolationError
)
from app.security.auth import require_api_key

router = APIRouter(prefix="/v1/solnet", tags=["solnet"])


@router.post(
    "/validate",
    response_model=VerificationCertificate,
    summary="Validate a transaction intent for logical consistency",
    dependencies=[Depends(require_api_key)],
    responses={
        422: {
            "model": LinguisticViolationError,
            "description": "The transaction intent is logically inconsistent."
        }
    }
)
async def validate_transaction(request: SolnetValidateRequest):
    """
    SOLNET Oracle Endpoint.
    
    Takes a transaction_intent string, runs it through the Anvaya-Bodha
    reasoning pipeline, and returns a VerificationCertificate containing:
    - is_logically_consistent: Boolean verdict.
    - karaka_trace: Full semantic role audit trail.
    - logic_hash: SHA-256 signature of the verified trace.
    
    Returns 422 Unprocessable Entity if a Linguistic Violation is detected.
    """
    from app.main import oracle  # Lazy import to avoid circular dependency

    certificate, has_violations = oracle.generate_solnet_certificate(
        transaction_intent=request.transaction_intent,
        node_id=request.node_id
    )

    if has_violations:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "LinguisticViolation",
                "detail": "Transaction intent failed logical consistency verification.",
                "violations": certificate.violations,
                "karaka_trace": [t.model_dump() for t in certificate.karaka_trace]
            }
        )

    return certificate
