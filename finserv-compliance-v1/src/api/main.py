"""
src/api/main.py
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from src.agent.compliance_agent import FallbackComplianceAgent, ComplianceAgentState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="FinServ Regulatory Compliance Assistant",
    version="1.0.0",
    description="AI-powered regulatory compliance assistant",
    docs_url="/docs",      # always enabled
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    valid_keys = ["dev-key-1234"]
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

# ── Models ────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Regulatory question")
    jurisdictions: Optional[list[str]] = None
    regulation_families: Optional[list[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the minimum CET1 capital ratio under Basel III?"
            }
        }

class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    counterparty_name: Optional[str] = None
    jurisdiction: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-XBORDER-001",
                "description": "Cross-border payment to unverified entity"
            }
        }

class ComplianceResponse(BaseModel):
    request_id: str
    timestamp: str
    query: str
    answer: str
    confidence: str
    applicable_regulations: list
    requires_human_review: bool
    processing_time_ms: float

# ── Agent singleton ───────────────────────────────────────────
_agent = None

@app.on_event("startup")
async def startup_event():
    global _agent
    logger.info("Starting Compliance Assistant...")
    try:
        _agent = FallbackComplianceAgent()
        logger.info("Compliance agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")

def get_agent():
    global _agent
    if _agent is None:
        _agent = FallbackComplianceAgent()
    return _agent

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "FinServ Regulatory Compliance Assistant",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query":              "POST /query",
            "screen_transaction": "POST /screen-transaction",
            "health":             "GET  /health",
            "docs":               "GET  /docs",
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    return {
        "status":    "healthy",
        "version":   "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent":     "loaded" if _agent is not None else "not loaded",
    }

@app.post("/query", response_model=ComplianceResponse, tags=["Compliance"])
async def regulatory_query(
    request: QueryRequest,
    _: str = Depends(verify_api_key),
):
    print(f"DEBUG main: regulation_families from request={request.regulation_families}")

    """Answer a natural language regulatory question with cited sources."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    logger.info(f"[{request_id}] Query: {request.query[:80]}")

    try:
        agent  = get_agent()
        result = agent.run(request.query,
                           regulation_filter=request.regulation_families[0] if request.regulation_families else None)   # FallbackComplianceAgent uses .run()
        ms     = (time.time() - start_time) * 1000

        return ComplianceResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            query=request.query,
            answer=result.get("answer", ""),
            confidence=result.get("confidence", "MEDIUM"),
            applicable_regulations=result.get("applicable_regulations", []),
            requires_human_review=result.get("requires_human_review", False),
            processing_time_ms=round(ms, 1),
        )

    except Exception as e:
        logger.error(f"[{request_id}] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screen-transaction", response_model=ComplianceResponse, tags=["Compliance"])
async def screen_transaction(
    request: TransactionRequest,
    _: str = Depends(verify_api_key),
):
    """Screen a financial transaction against regulatory frameworks."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    query = (
        request.description
        or f"Screen transaction {request.transaction_id} for compliance"
    )

    transaction_data = {
        "transaction_id":  request.transaction_id,
        "description":     request.description,
        "amount":          request.amount,
        "currency":        request.currency,
        "counterparty":    request.counterparty_name,
        "jurisdiction":    request.jurisdiction,
    }

    logger.info(f"[{request_id}] Screening: {query[:80]}")

    try:
        agent  = get_agent()
        result = agent.run(query, transaction=transaction_data)
        ms     = (time.time() - start_time) * 1000

        return ComplianceResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            query=query,
            answer=result.get("answer", ""),
            confidence=result.get("confidence", "MEDIUM"),
            applicable_regulations=result.get("applicable_regulations", []),
            requires_human_review=result.get("requires_human_review", False),
            processing_time_ms=round(ms, 1),
        )

    except Exception as e:
        logger.error(f"[{request_id}] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))