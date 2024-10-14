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
  "text": "Organic apple juice 1L carton"
}
```
**Response**:
```json
{
  "id": 1,
  "code": "10000004",
  "title": "Beverages",
  "full_title": "Food & Beverage > Non-Alcoholic Beverages > Juice > Apple Juice",
  "definition": "Juices from fresh apples...",
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