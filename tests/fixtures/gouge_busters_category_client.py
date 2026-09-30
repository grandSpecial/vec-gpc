# Extracted unchanged from Gouge-Busters/gouge-busters-api main.py at
# 3ccc841ab17913d362db5ead9fc914e454d4c964 (remote main verified 2026-09-30).
async def get_category_from_api(item_name: str) -> str:
    """
    Fetch category for an item from the GPC API.
    
    Args:
        item_name: Name of the item to categorize
    
    Returns:
        str: Category from API or None if failed
    """
    try:
        bearer_token = os.getenv("VEC_GPC_API_KEY")
        if not bearer_token:
            print("VEC_GPC_API_KEY not found in environment")
            return None
            
        url = "https://vec-gpc-84d0747f7862.herokuapp.com/search"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        params = {"text": item_name}
        
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("level_2_category")
            
    except Exception as e:
        logger.warning("Category lookup failed for '%s': %s", item_name, str(e))
        return None
