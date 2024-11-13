import os
import openai
import requests
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Azure OpenAI and Azure Cognitive Search configurations
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_version = "2023-06-01-preview"

embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
completion_model = os.getenv("AZURE_OPENAI_COMPLETION_MODEL")
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_SERVICE_API_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Step 1: Load data and generate embeddings
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def generate_embedding(text):
    response = openai.Embedding.create(
        engine=embedding_model,
        input=text
    )
    return response['data'][0]['embedding']

def create_documents_with_embeddings(df):
    documents = []
    for _, row in df.iterrows():
        content = row['content']  # Adjust based on your data's content column
        embedding = generate_embedding(content)
        document = {
            "id": str(row['id']),  # Adjust to your unique identifier column
            "content": content,
            "embedding": embedding
        }
        documents.append(document)
    return documents

# Step 2: Index data with embeddings into Azure Cognitive Search
def index_documents(documents):
    url = f"{search_endpoint}/indexes/{index_name}/docs/index?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_api_key
    }
    batch = {"value": documents}
    response = requests.post(url, headers=headers, json=batch)
    response.raise_for_status()
    print("Documents indexed successfully.")

# Step 3: Vector search in Azure Cognitive Search
def search_cognitive_search(query_embedding):
    url = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_api_key
    }
    search_payload = {
        "vector": {
            "value": query_embedding,
            "fields": "embedding",
            "k": 5  # Number of top results to retrieve
        },
        "top": 5
    }
    response = requests.post(url, headers=headers, json=search_payload)
    response.raise_for_status()
    return response.json()

# Step 4: Generate a response with context using Azure OpenAI
def generate_response_with_context(query, context):
    prompt = f"Context: {context}\n\nUser Query: {query}\n\nResponse:"
    response = openai.Completion.create(
        engine=completion_model,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Step 5: Complete RAG process
def rag_response(user_query):
    # Step 5.1: Generate embedding for user query
    query_embedding = generate_embedding(user_query)

    # Step 5.2: Search Azure Cognitive Search with the query embedding
    search_results = search_cognitive_search(query_embedding)

    # Step 5.3: Extract relevant content from search results
    context = "\n".join([doc["content"] for doc in search_results["value"]])  # Replace "content" with your content field name

    # Step 5.4: Generate response using Azure OpenAI with the retrieved context
    response = generate_response_with_context(user_query, context)
    return response

# Main function to run the full RAG implementation
def main():
    # Load and index data only once (comment this out after the first run)
    data_file = "your_data.csv"  # Path to your CSV file
    df = load_data(data_file)
    documents = create_documents_with_embeddings(df)
    index_documents(documents)

    # Run RAG with a sample user query
    user_query = "What are the effects of climate change?"
    response = rag_response(user_query)
    print("Response:", response)

# Execute the main function
if __name__ == "__main__":
    main()
