import openai
from sqlalchemy.orm import sessionmaker
from models import GPCLevel, engine
import os
from dotenv import load_dotenv
from tqdm import tqdm
import asyncio
import time

load_dotenv()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Use the async client
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_level_3_category_async(gpc_item, semaphore):
    """Generate a consumer-friendly category name for Level 3 GPC items (async version)"""
    
    async with semaphore:  # Limit concurrent requests
        prompt = f"""
        Given this Level 3 GPC classification:
        
        Title: {gpc_item.title}
        Full Path: {gpc_item.full_title}
        Definition: {gpc_item.definition}
        
        Generate a consumer-friendly category name that:
        1. Is intuitive and recognizable to consumers (like you'd see on a grocery receipt)
        2. Is 1-3 words maximum
        3. Groups similar products together (e.g., "apples" and "bananas" should both be "Produce")
        4. Is at an appropriate level of specificity for a Level 3 category
        
        Examples of good Level 3 consumer categories:
        - "Produce" (for fruits, vegetables)
        - "Dairy" (for milk, cheese, yogurt)
        - "Meat" (for beef, chicken, pork)
        - "Bakery" (for bread, pastries)
        - "Beverages" (for drinks, juice)
        - "Snacks" (for chips, crackers)
        - "Electronics" (for phones, computers)
        - "Clothing" (for shirts, pants)
        - "Household" (for cleaning supplies)
        - "Health" (for medicine, supplements)
        
        Return ONLY the category name, nothing else.
        """
        
        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at creating consumer-friendly product categories for retail and e-commerce."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            category = response.choices[0].message.content.strip()
            # Clean up the response (remove quotes, extra text, etc.)
            category = category.replace('"', '').replace("'", '').strip()
            
            return gpc_item.id, category
            
        except Exception as e:
            print(f"Error generating category for {gpc_item.title}: {e}")
            return gpc_item.id, gpc_item.title  # Fallback to original title

async def generate_level_2_category_async(gpc_item, semaphore):
    """Generate a consumer-friendly category name for Level 2 GPC items (async version)"""
    async with semaphore:
        prompt = f"""
        Given this Level 2 GPC classification:
        
        Title: {gpc_item.title}
        Full Path: {gpc_item.full_title}
        Definition: {gpc_item.definition}
        
        Generate a consumer-friendly category name that:
        1. Is intuitive and recognizable to consumers (like you'd see on a grocery receipt)
        2. Is 1-3 words maximum
        3. Is at an appropriate level of specificity for a Level 2 category
        
        Examples:
        - Produce, Dairy, Meat, Bakery, Beverages, Snacks, Frozen, Pantry, Household, Health
        
        Return ONLY the category name, nothing else.
        """
        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at creating consumer-friendly product categories for retail and e-commerce."},
                    {"role": "user", "content": prompt}
                ],
            )
            category = response.choices[0].message.content.strip()
            category = category.replace('"', '').replace("'", '').strip()
            return gpc_item.id, category
        except Exception as e:
            print(f"Error generating category for {gpc_item.title}: {e}")
            return gpc_item.id, gpc_item.title

async def generate_level_categories_async():
    """Generate consumer categories for Level 2 and Level 3 GPC items using async/await"""
    session = SessionLocal()
    
    try:
        # Get Level 2 and Level 3 items missing their categories
        level_2_items = session.query(GPCLevel).filter(
            GPCLevel.level == 2,
            GPCLevel.level_2_category.is_(None)
        ).all()
        level_3_items = session.query(GPCLevel).filter(
            GPCLevel.level == 3,
            GPCLevel.level_3_category.is_(None)
        ).all()
        
        print(f"Using async processing with asyncio...")
        print(f"Level 2 to generate: {len(level_2_items)} | Level 3 to generate: {len(level_3_items)}")
        
        start_time = time.time()
        
        # Limit concurrent requests to avoid rate limiting (adjust as needed)
        semaphore = asyncio.Semaphore(15)  # 15 concurrent requests
        
        # Create tasks for Level 2 and Level 3
        tasks_lvl2 = [generate_level_2_category_async(item, semaphore) for item in level_2_items]
        tasks_lvl3 = [generate_level_3_category_async(item, semaphore) for item in level_3_items]
        
        results_lvl2 = []
        if tasks_lvl2:
            for task in tqdm(asyncio.as_completed(tasks_lvl2), total=len(tasks_lvl2), desc="Generating Level 2 categories"):
                results_lvl2.append(await task)
        
        results_lvl3 = []
        if tasks_lvl3:
            for task in tqdm(asyncio.as_completed(tasks_lvl3), total=len(tasks_lvl3), desc="Generating Level 3 categories"):
                results_lvl3.append(await task)
        
        # Update database with results (batch writes)
        print("Updating database with results...")
        for item_id, category in tqdm(results_lvl2, desc="Updating Level 2 in DB"):
            item = session.query(GPCLevel).filter_by(id=item_id).first()
            if item:
                item.level_2_category = category
        for item_id, category in tqdm(results_lvl3, desc="Updating Level 3 in DB"):
            item = session.query(GPCLevel).filter_by(id=item_id).first()
            if item:
                item.level_3_category = category
        
        session.commit()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Level 2/3 category generation complete!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        total_items = max(1, (len(level_2_items) + len(level_3_items)))
        print(f"Average time per item: {elapsed_time/total_items:.2f} seconds")
        
        # Show some statistics
        total_level_2 = session.query(GPCLevel).filter(GPCLevel.level == 2).count()
        total_level_3 = session.query(GPCLevel).filter(GPCLevel.level == 3).count()
        categorized_level_2 = session.query(GPCLevel).filter(
            GPCLevel.level == 2,
            GPCLevel.level_2_category.isnot(None)
        ).count()
        categorized_level_3 = session.query(GPCLevel).filter(
            GPCLevel.level == 3,
            GPCLevel.level_3_category.isnot(None)
        ).count()
        
        print(f"Total Level 2 items: {total_level_2}")
        print(f"Categorized Level 2 items: {categorized_level_2}")
        print(f"Total Level 3 items: {total_level_3}")
        print(f"Categorized Level 3 items: {categorized_level_3}")
        
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    asyncio.run(generate_level_categories_async())