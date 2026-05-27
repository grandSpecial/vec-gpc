import openai
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from models import (
    Base,
    SessionLocal,
    engine,
    ClassificationLog,
    ClassificationCandidate,
)
import os
from dotenv import load_dotenv
import numpy as np
import logging
from classifier import GPCClassifier

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer()
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
assert API_AUTH_TOKEN is not None

EMBEDDING_MODEL = "text-embedding-3-small"
DESCRIPTION_MODEL = "gpt-4o"
PROMPT_VERSION = "receipt-description-v2"
TAXONOMY_VERSION = "GPC_v20240603"
DEFAULT_SOURCE = "Gouge Busters"
LOG_CANDIDATE_LIMIT = 5

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

app = FastAPI(dependencies=[Depends(validate_token)])
Base.metadata.create_all(
    bind=engine,
    tables=[ClassificationLog.__table__, ClassificationCandidate.__table__]
)

# Initialize OpenAI API client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to generate vector from input text using OpenAI
def create_vector(text: str):
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating vector: {e}")

def create_description(text):
    response = client.chat.completions.create(
      model=DESCRIPTION_MODEL,
      messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": """
                You take a short, often abbreviated item description and rewrite it as one short sentence that describes only
                the product itself for semantic classification.

                Focus on intrinsic attributes such as what the item is, its physical form, material, and composition.
                Do not mention what it is used with, what it holds, what it is served with, accessories, pairings, recipes,
                occasions, or nearby products in the same meal.
                If the item name mentions another product only as a serving style, filling, companion food, or intended use,
                omit that related product and keep the sentence centered on the purchased item itself.
                Do not add brand context or speculative details.
                Return a single plain sentence.
              """,
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": text
            }
          ]
        },
      ],
      temperature=0,
      max_tokens=128,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      response_format={
        "type": "text"
      }
    )
    return response

def log_classification_event(
    text: str,
    description: str,
    gpc_item,
    level_2_category: str,
    level_3_category: str,
    similarity_score: float,
    candidate_rows,
    latency_ms: int,
    source: str = DEFAULT_SOURCE,
):
    log_db = SessionLocal()
    try:
        classification_log = ClassificationLog(
            source=source,
            input_text=text,
            generated_description=description,
            predicted_gpc_id=gpc_item.id,
            predicted_gpc_code=gpc_item.code,
            predicted_title=gpc_item.title,
            predicted_full_title=gpc_item.full_title,
            level_2_category=level_2_category,
            level_3_category=level_3_category,
            definition=gpc_item.definition,
            active=gpc_item.active,
            embedding_model=EMBEDDING_MODEL,
            description_model=DESCRIPTION_MODEL,
            prompt_version=PROMPT_VERSION,
            taxonomy_version=TAXONOMY_VERSION,
            similarity_score=similarity_score,
            top_candidate_count=len(candidate_rows),
        )
        log_db.add(classification_log)
        log_db.flush()

        for rank, row in enumerate(candidate_rows, start=1):
            if hasattr(row, "gpc_item"):
                candidate_item = row.gpc_item
                similarity_score = row.similarity_score
            else:
                candidate_item = row["gpc_item"]
                similarity_score = row["similarity_score"]
            log_db.add(
                ClassificationCandidate(
                    classification_log_id=classification_log.id,
                    rank=rank,
                    gpc_id=candidate_item.id,
                    gpc_code=candidate_item.code,
                    title=candidate_item.title,
                    full_title=candidate_item.full_title,
                    similarity_score=similarity_score,
                )
            )

        log_db.commit()
    except Exception:
        log_db.rollback()
        logger.exception(
            "Failed to log classification event",
            extra={"input_text": text[:200], "latency_ms": latency_ms},
        )
    finally:
        log_db.close()

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/ping", dependencies=[Depends(validate_token)])
def ping():
    return {"ok": True}

# Endpoint to search for closest vector match and return corresponding GPCLevel row
@app.post("/search",dependencies=[Depends(validate_token)])
def search_item(
    text: str,
    include_candidates: bool = False,
    db: Session = Depends(get_db),
):
    try:
        classifier = GPCClassifier(db, create_description, create_vector)
        result = classifier.classify(text, include_candidates=include_candidates)
        gpc_item = result["log_candidates"][0].gpc_item

        log_classification_event(
            text=text,
            description=result["description"],
            gpc_item=gpc_item,
            level_2_category=result["category"],
            level_3_category=result["subcategory"],
            similarity_score=result["log_candidates"][0].similarity_score,
            candidate_rows=result["log_candidates"],
            latency_ms=result["latency_ms"],
        )

        result.pop("log_candidates", None)
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for item: {e}")
