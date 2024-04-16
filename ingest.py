import os
from pymongo import MongoClient
from bson import json_util
import base64
from PIL import Image
import io
import boto3
import json

# Setup for AWS and Bedrock Runtime
bedrock_runtime = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
    region_name="us-east-1"
)

def construct_bedrock_body(base64_string,text):
    return json.dumps({
        "inputImage": base64_string,
        # "inputText": text,
        "embeddingConfig": {"outputEmbeddingLength": 1024},
    })

def get_embedding_from_titan_multimodal(body, insert_data):
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    if len(insert_data) % 100 == 0:
        print('Embedded percent ' + str(len(insert_data) / 10) + '%')
    return response_body["embedding"]

# Setup MongoDB connection
uri = os.environ.get('MONGODB_ATLAS_URI')
client = MongoClient(uri)
db_name = 'mdb_icons'
collection_name = 'icons'
celeb_images = client[db_name][collection_name]

insert_data = []

# Specify the main directory
main_directory = '/Users/pavel.duchovny/Downloads/MongoDB-selected-assets (2)'

# Generate list of subdirectories
image_directories = [os.path.join(main_directory, name) for name in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, name))]

# Load images from discovered subdirectories
for directory in image_directories:
    for file_name in os.listdir(directory):
        if "verse" in file_name:
            continue
        if "verted" in file_name:
            continue
        
        if file_name.endswith(".png"):  # Checks if it's a PNG file
            file_path = os.path.join(directory, file_name)
            with Image.open(file_path) as img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_byte = buffered.getvalue()
                img_base64 = base64.b64encode(img_byte)
                img_str = img_base64.decode('utf-8')

                # Create document
                doc_celeb = {
                    'image': img_str,
                    'type' : 'illustration',
                    'file_name': file_name
                }
                body = construct_bedrock_body(img_str, file_name)
                embedding = get_embedding_from_titan_multimodal(body, insert_data)
                doc_celeb['embeddings'] = embedding
                insert_data.append(doc_celeb)

                # Insert into MongoDB in batches of 1000
                if len(insert_data) == 1000:
                    celeb_images.insert_many(insert_data)
                    print("1000 records ingested")
                    insert_data = []

# Final insertion if there are remaining records
if len(insert_data) > 0:
    celeb_images.insert_many(insert_data)
    print("Data Ingested")
