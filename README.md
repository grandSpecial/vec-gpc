# **VEC GPC**  
**Semantic Search for GS1's Global Product Classification**

VEC GPC is a powerful and intuitive API that allows users to perform semantic searches against GS1’s Global Product Classification (GPC) system using advanced vector-based search techniques. This API enables businesses to match raw, incomplete and user inputted product descriptions to standardized GPC categories, providing a streamlined and efficient way to classify products at scale.

### **Why VEC GPC?**
Global Product Classification (GPC) is the foundation of standardized product categorization across industries, but mapping unstructured product data to GPC codes manually can be cumbersome and error-prone. VEC GPC solves this by leveraging the power of AI and semantic search to quickly and accurately match any product description to the appropriate GPC category — allowing businesses to automate the classification process and ensure data consistency.

## **Key Features:**
- **Semantic Search**: Uses advanced vector embeddings for precise, meaning-based search rather than relying on exact keyword matching.
- **High Accuracy**: Built on state-of-the-art AI models for accurate classification, even when dealing with incomplete or unstructured product descriptions.
- **Simple Integration**: Easy-to-use REST API that integrates into any application with minimal setup.
- **Scalability**: Handles large-scale classification tasks efficiently, making it perfect for growing product catalogs.

## **Who Is It For?**
- **Retailers**: Automate product classification across vast and diverse inventories to ensure standardized GPC coding.
- **Manufacturers**: Quickly classify new or existing products into the global product taxonomy.
- **Logistics and Supply Chain**: Maintain consistent and accurate product categorization throughout your supply chain.
- **Marketplaces**: Easily categorize user-generated product data into standardized formats, providing consistency across listings.

## **Use Cases**:
1. **Automated Product Catalog Management**: Companies with large product inventories can automate the classification of new items, reducing manual effort and human error.
2. **Data Standardization**: Organizations looking to clean and standardize product data across multiple departments or vendors can use VEC GPC to ensure consistent GPC categorization.
3. **Supply Chain Optimization**: Streamline logistics and inventory management by ensuring that all products are properly classified and tracked in the supply chain.
4. **Product Matching in Marketplaces**: Easily categorize and match products from various vendors into a standardized format for efficient search and display on e-commerce platforms.

## **API Endpoint**:
### **Search for a Product Category**
- **Endpoint**: `/search`
- **Method**: `POST`
- **Input**: A simple text-based product description.
- **Response**: The GPC category that best matches the provided description, along with additional metadata such as category name, definition, and code.

### Example:
```json
{
  "text": "Banana"
}
```
**Response** (selected fields):
```json
{
  "code": 10005897,
  "title": "Bananas",
  "level_2_category": "Produce",
  "level_3_category": "Bananas",
  "category": "Produce",
  "subcategory": "Bananas",
  "active": true
}
```

## **Getting Started**:
1. **Sign up**: Visit [RapidAPI](https://rapidapi.com/) to subscribe to the VEC GPC API.
2. **Integrate**: Use your favorite HTTP client to make requests to our REST API.
3. **Classify**: Start matching your product descriptions to GS1 GPC categories seamlessly.

## **Pricing**:
- **Free Tier**: Ideal for testing and small-scale classification.
- **Pro Tier**: For businesses with larger classification needs and higher volume.
- **Enterprise Tier**: Fully customizable pricing for large-scale deployments.
## Compatibility and classification pipeline

The existing Gouge Busters integration remains supported: authenticated
`POST /search?text=...` returns a JSON object containing the string
`level_2_category`. The optional JSON body `{"text":"..."}` is also accepted;
a query parameter takes precedence. Existing response fields, GPC IDs/codes,
and the familiar category overrides (Produce, Bakery, Pantry, Dairy & Eggs,
Meat & Poultry, Beverages, Snacks & Candy, Prepared Foods) are retained.
Missing mappings now use canonical taxonomy ancestry and the existing database
level-2 category labels, such as Skin Care and Pet Food & Drinks.

The API embeds normalized receipt text alongside an optional short retrieval hint,
retrieves active level-4 product bricks,
adds candidates from exact taxonomy product-type aliases, and asks the model to
choose only among those candidates. Shared ingredients/attributes are no longer
promoted to arbitrary parent products. The original receipt always remains in the embedding. Final selection sees the
original receipt, not the retrieval hint as ground truth, and returns a short
description for compatibility.
Retrieval-hint failure falls back to the original text. Model selection failure falls back to deterministic ranking with `needs_review=true`.
Embedding service failures return a generic 502/504 rather than internal details.

`confidence` is a conservative ranking signal, **not a calibrated probability**.
Closely competing candidates, missing alternatives, and uncertain model selections
require review. The existing Gouge Busters consumer reads only `level_2_category`;
it does not currently act on `needs_review`. Ambiguous receipt lines and fees still
receive a best candidate to preserve that contract and should not be interpreted
as verified product identifications.

Retrieval hints have a 1.5-second timeout, embeddings 2 seconds, and selection
3 seconds, with no automatic retries in the request path. Embeddings and valid selections have bounded, process-local caches.
Database connections are checked before reuse, with bounded pool/connection waits
and statement timeouts. These are individual operation limits, not a guaranteed
end-to-end SLA; the consumer's existing 10-second timeout still applies.
`classification_result` and `classification_failed` structured log events provide
request IDs, selection/fallback status, model/prompt/ranking versions, latency,
and candidate decisions without requiring a database migration.

## Validation and rollout

Run offline regression and consumer contract tests:

```sh
python -m unittest discover -s tests -v
```

`tests/fixtures/gouge_busters_category_client.py` preserves the actual consumer
helper from Gouge-Busters/gouge-busters-api commit
`3ccc841ab17913d362db5ead9fc914e454d4c964`. Its query-string call is tested against
this application's ASGI endpoint. No consumer changes are required.

The model defaults to the pinned `gpt-4.1-mini-2025-04-14` snapshot and can be
configured with `CLASSIFICATION_MODEL`. It supports the existing Chat Completions
API and structured output schema ([OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)).
Model changes must be evaluated on both the regression and holdout sets.

Run the expanded labeled evaluation against the configured database (read-only;
calls the embedding and selection APIs, so it incurs API usage):

```sh
python evaluate_classifier.py --show-candidates
python evaluate_classifier.py --gold-set evaluation/holdout_receipt_queries.csv
```

Audit the active product embeddings without making changes:

```sh
python refresh_embeddings.py
```

A separate, explicit maintenance operation refreshes only active product-brick
vectors and their version markers, preserving all GPC rows and category names:

```sh
python refresh_embeddings.py --apply
```

Refresh markers include a hash of embedding content and model, so interrupted
runs can resume and changed definitions cannot silently retain old vectors.
Each batch commits vectors and markers together. The application works with
existing vectors; refreshing is not a schema prerequisite. Evaluate before and
after any refresh, and back up vectors/markers before changing a production index:
rolling back application code alone does not restore changed embeddings.
Deploy the code separately from any embedding refresh so failures are attributable.
No production writes or deployments are performed by the test suite.

Validation results and limits are recorded in [evaluation/validation_20260930.json](evaluation/validation_20260930.json).
