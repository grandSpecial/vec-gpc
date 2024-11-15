import openai
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import GPCLevel, Items, SessionLocal  # Import your SQLAlchemy models and session
import os
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
import numpy as np

load_dotenv()

bearer_scheme = HTTPBearer()
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
assert API_AUTH_TOKEN is not None

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

app = FastAPI(dependencies=[Depends(validate_token)])

# Initialize OpenAI API client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to generate vector from input text using OpenAI
def create_vector(text: str):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating vector: {e}")

def create_description(text):
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You take a short item description and describe it in more detail for use in performing semantic search. You return a single sentence. "
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
      temperature=1,
      max_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      response_format={
        "type": "text"
      }
    )
    return response

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint to search for closest vector match and return corresponding GPCLevel row
@app.post("/search",dependencies=[Depends(validate_token)])
def search_item(text: str, db: Session = Depends(get_db)):
    response = create_description(text)
    description = response.choices[0].message.content
    # Generate vector from input text
    vector = create_vector(description)
    
    # Search for the closest vector in the items table
    try:
        closest_item = db.execute(
            select(Items).order_by(Items.vector.l2_distance(vector)).limit(1)
        ).scalar_one_or_none()

        if closest_item is None:
            raise HTTPException(status_code=404, detail="No matching item found")
        
        # Lookup corresponding GPCLevel row by id
        gpc_item = db.query(GPCLevel).filter_by(id=closest_item.id).first()

        if gpc_item is None:
            raise HTTPException(status_code=404, detail="GPCLevel item not found")

        return {
            "id": gpc_item.id,
            "code": gpc_item.code,
            "title": gpc_item.title,
            "full_title": gpc_item.full_title,
            "description": description,
            "definition": gpc_item.definition,
            "active": gpc_item.active
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for item: {e}")
