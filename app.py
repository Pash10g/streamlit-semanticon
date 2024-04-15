import streamlit as st
from pymongo import MongoClient
from PIL import Image
import io
import base64
import boto3
import os
import json

# AWS Bedrock client setup
bedrock_runtime = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
    region_name="us-east-1"
)

# Construct body for Bedrock API to generate embeddings
def construct_bedrock_body(query):
    return json.dumps({
        "inputText": query,
        "embeddingConfig": {"outputEmbeddingLength": 1024},
    })

# Fetch embedding from Bedrock
def get_embedding_from_titan_multimodal(body):
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    return response_body["embedding"]

# MongoDB connection setup
def get_mongo_connection(uri):
    client = MongoClient(uri)
    return client

# MongoDB vector search
def vector_search(embedding, collection):
    search_query = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": embedding,
                "numCandidates": 10,
                "limit": 10
            }
        }
    ]
    results = list(collection.aggregate(search_query))
    return results

def display_base64_image(base64_string, filename):
    """Decode a base64 image string and display it using Streamlit in a tile format.

    Args:
    base64_string (str): Base64 encoded string of the image.
    filename (str): Name of the file to use as a caption.
    """
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)
    
    # Convert bytes data to a PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    # Display the image using Streamlit with caption and using the column width
    return image, filename

def process_and_display_results(results):
    """Process and display each result including the image and file name in tiled format.

    Args:
    results (list of dicts): List containing the query results with image and filename.
    """
    # Define the number of columns for the tiled layout
    columns_per_row = 3  # You can adjust the number of columns per row here
    column_width = int(12 / columns_per_row)  # Calculate column width for equal spacing

    # Initialize columns
    cols = st.columns(columns_per_row)
    col_index = 0

    for result in results:
        # Retrieve image and filename
        image, filename = display_base64_image(result['image'], result['file_name'])
        
        # Display in the current column
        with cols[col_index]:
            
            st.image(image, caption=f"Image: {filename}", use_column_width=True)
        
        # Update column index and reset if it exceeds the number of columns
        col_index += 1
        if col_index >= columns_per_row:
            cols = st.columns(columns_per_row)
            col_index = 0
# Streamlit UI setup
st.title('SEMANTICON: Query MongoDB Icons Using semantic search')

# Connect to MongoDB
uri = os.getenv('MONGODB_ATLAS_URI') # Replace with your actual MongoDB URI
client = get_mongo_connection(uri)
db = client['mdb_icons']
collection = db['icons']

# Text input for query
user_query = st.text_input("Enter your query:")

if user_query:
    # Generate embeddings using AWS Bedrock
    body = construct_bedrock_body(user_query)
    query_embedding = get_embedding_from_titan_multimodal(body)

    # Search MongoDB using the generated embedding
    results = vector_search(query_embedding, collection)

    # Display the results
    if results:
        st.write(f"Found {len(results)} results!")
        process_and_display_results(results)
    else:
        st.write("No results found.")
        
