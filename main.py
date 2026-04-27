import openai
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import select
from models import GPCLevel, Items, SessionLocal  # Import your SQLAlchemy models and session
import os
from dotenv import load_dotenv
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

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_level_3_category(gpc_item, db):
    """Get the Level 3 category for any GPC item by traversing up the hierarchy"""
    
    # If this item is already Level 3, return its category
    if gpc_item.level == 3:
        return gpc_item.level_3_category
    
    # If this item is Level 4 or 5, find its Level 3 parent
    current_item = gpc_item
    while current_item and current_item.level > 3:
        current_item = db.query(GPCLevel).filter_by(id=current_item.parent_id).first()
    
    # If we found a Level 3 parent, return its category
    if current_item and current_item.level == 3:
        return current_item.level_3_category
    
    # Fallback: extract from full_title
    title_parts = gpc_item.full_title.split(" > ")
    if len(title_parts) >= 3:
        return title_parts[2].strip()  # Level 3 title
    
    return gpc_item.title  # Final fallback

def get_level_2_category(gpc_item, db):
    """Get the Level 2 category for any GPC item by traversing up the hierarchy"""
    # If already Level 2
    if gpc_item.level == 2:
        return getattr(gpc_item, "level_2_category", None) or gpc_item.title

    # Traverse up until level 2
    current_item = gpc_item
    while current_item and current_item.level > 2:
        current_item = db.query(GPCLevel).filter_by(id=current_item.parent_id).first()

    if current_item and current_item.level == 2:
        return getattr(current_item, "level_2_category", None) or current_item.title

    # Fallback from full_title
    title_parts = gpc_item.full_title.split(" > ")
    if len(title_parts) >= 2:
        return title_parts[1].strip()

    return gpc_item.title

# Endpoint to search for closest vector match and return corresponding GPCLevel row
@app.post("/search",dependencies=[Depends(validate_token)])
def search_item(text: str, db: Session = Depends(get_db)):
    response = create_description(text)
    description = response.choices[0].message.content.strip()
    vector = create_vector(description)
    
    # Search for the closest vector in the items table
    try:
        closest_item = db.execute(
            select(Items).order_by(Items.vector.cosine_distance(vector)).limit(1)
        ).scalar_one_or_none()

        if closest_item is None:
            raise HTTPException(status_code=404, detail="No matching item found")

        gpc_item = db.query(GPCLevel).filter_by(id=closest_item.id).first()

        if gpc_item is None:
            raise HTTPException(status_code=404, detail="GPCLevel item not found")

        # Get the Level 3 category
        level_3_category = get_level_3_category(gpc_item, db)
        # Get the Level 2 category
        level_2_category = get_level_2_category(gpc_item, db)

        return {
            "id": gpc_item.id,
            "code": gpc_item.code,
            "title": gpc_item.title,
            "full_title": gpc_item.full_title,
            "level_2_category": level_2_category,
            "level_3_category": level_3_category,  # Updated field name
            "description": description,
            "definition": gpc_item.definition,
            "active": gpc_item.active
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for item: {e}")
