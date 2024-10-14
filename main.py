import openai
from fastapi import FastAPI, HTTPException, Depends, Header
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import GPCLevel, Items, SessionLocal  # Import your SQLAlchemy models and session
import os
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
import numpy as np

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

# Dependency that checks for the correct token
def verify_token(authorization: str = Header(None)):
    if authorization != f"Bearer {API_AUTH_TOKEN}":
        raise HTTPException(
            status_code=401, detail="Invalid or missing token"
        )

# Initialize OpenAI API client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to generate vector from input text using OpenAI
def create_vector(text: str, token: str = Depends(verify_token)):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating vector: {e}")

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint to search for closest vector match and return corresponding GPCLevel row
@app.post("/search")
def search_item(text: str, db: Session = Depends(get_db)):
    # Generate vector from input text
    vector = create_vector(text)
    
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
            "definition": gpc_item.definition,
            "active": gpc_item.active
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for item: {e}")
