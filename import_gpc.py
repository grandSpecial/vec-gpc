import json
from openai import OpenAI
from sqlalchemy import text
from tqdm import tqdm
from sqlalchemy.dialects.postgresql import insert  # For bulk insertions
from models import GPCLevel, Items, SessionLocal, engine, EmbeddingRefreshState
from models import Model  # Import the Pydantic model
from embedding_text import build_gpc_embedding_text
import os  
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_VERSION = "gpc-embedding-v1"

# Create the database tables if they don't already exist
GPCLevel.metadata.create_all(bind=engine)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to generate vectors using OpenAI
def create_vector(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
        encoding_format="float"  
    )
    return response.data[0].embedding

# Function to insert/update the items table with vectors
def upsert_item_vector(session, gpc_item_id, embedding_text):
    vector = create_vector(embedding_text)

    session.execute(
        insert(Items)
        .values(id=gpc_item_id, vector=vector)
        .on_conflict_do_update(
            index_elements=[Items.id],
            set_={"vector": vector}
        )
    )
    session.commit()

def is_embedding_current(session, gpc_item_id):
    refresh_state = session.query(EmbeddingRefreshState).filter_by(gpc_item_id=gpc_item_id).first()
    return refresh_state is not None and refresh_state.embedding_version == EMBEDDING_VERSION

def mark_embedding_current(session, gpc_item_id):
    refresh_state = session.query(EmbeddingRefreshState).filter_by(gpc_item_id=gpc_item_id).first()
    if refresh_state:
        refresh_state.embedding_version = EMBEDDING_VERSION
    else:
        session.add(
            EmbeddingRefreshState(
                gpc_item_id=gpc_item_id,
                embedding_version=EMBEDDING_VERSION
            )
        )
    session.commit()

# Function to update the full_title field and vector for each row
def update_gpc_item(session, item, level, parent_id=None, parent_titles=""):
    # Create the concatenated title including the parent's titles
    full_title = (parent_titles + " " + item.Title).strip()

    # Check if the record already exists
    existing_item = session.query(GPCLevel).filter_by(code=item.Code).first()
    if existing_item:
        # Update existing record
        existing_item.full_title = full_title  # Update the full title field
        existing_item.title = item.Title
        existing_item.definition = item.Definition
        existing_item.definition_excludes = item.DefinitionExcludes
        existing_item.active = item.Active
        existing_item.parent_id = parent_id
        gpc_item_id = existing_item.id
    else:
        # Insert new record
        new_item = GPCLevel(
            level=level,
            code=item.Code,
            title=item.Title,
            full_title=full_title,  # Store the full title with parent titles
            definition=item.Definition,
            definition_excludes=item.DefinitionExcludes,
            active=item.Active,
            parent_id=parent_id
        )
        session.add(new_item)
        session.flush()  # Get the ID of the inserted item
        gpc_item_id = new_item.id

    embedding_text = build_gpc_embedding_text(item.Title, full_title, item.Definition)

    # Only refresh vectors that have not been updated for the current embedding version.
    if not is_embedding_current(session, gpc_item_id):
        upsert_item_vector(session, gpc_item_id, embedding_text)
        mark_embedding_current(session, gpc_item_id)

    # Update the child items recursively
    for child in tqdm(item.Childs, desc=f"Processing children of {item.Title}", leave=False):
        update_gpc_item(session, child, level + 1, gpc_item_id, full_title)

# Populate GPC table recursively with tqdm progress bar
def populate_gpc_table(session, gpc_data):
    for schema_item in tqdm(gpc_data.Schema, desc="Populating GPC table"):
        update_gpc_item(session, schema_item, level=1)

# Load and validate GPC data
def load_gpc_data(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    # Use Pydantic model to validate the data
    gpc_model = Model(**raw_data)
    return gpc_model

def main():
    session = SessionLocal()

    # Load GPC data
    gpc_data = load_gpc_data('GPC_v20240603.json')

    # Populate GPC table with a progress bar
    populate_gpc_table(session, gpc_data)

    session.commit()

if __name__ == "__main__":
    main()
