import logging
from pymilvus import Collection, connections, utility
from pymilvus.model import DefaultEmbeddingFunction
from pymilvus import FieldSchema, CollectionSchema, DataType

# Set up logging
logging.basicConfig(filename='milvus_test.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Verify that milvus_model is installed
try:
    import milvus_model
    logging.info("milvus_model is installed.")
except ImportError as e:
    logging.error("milvus_model is not installed. Please install it using 'pip install pymilvus[model]'.")

# Create a Milvus client connection
connections.connect("default", host="localhost", port="19530")

# Step 1: Create a Collection
collection_name = "demo_collection"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Define the schema
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=255)
subject_field = FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=255)

schema = CollectionSchema(fields=[id_field, vector_field, text_field, subject_field])

collection = Collection(name=collection_name, schema=schema)
logging.info(f"Collection '{collection_name}' created successfully.")

# Step 2: Prepare Data
embedding_fn = DefaultEmbeddingFunction()

# Text strings to search from
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

# Generate vectors
vectors = embedding_fn.encode_documents(docs)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

logging.info(f"Data prepared: {data}")

# Step 3: Insert Data
mr = collection.insert(data)
logging.info(f"Data inserted: {mr}")

# Step 4: Create an Index
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 100},
    "metric_type": "L2"
}
collection.create_index(field_name="vector", index_params=index_params)
logging.info(f"Index created on collection '{collection_name}'.")

# Step 5: Load the Collection
collection.load()
logging.info(f"Collection '{collection_name}' loaded successfully.")

# Step 6: Prepare Search Parameters
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

# Step 7: Conduct a Vector Similarity Search
query_vector = vectors[0]  # Using the first vector for the search example
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param=search_params,
    limit=10,
    expr=None,
    output_fields=["id", "text", "subject"],
    consistency_level="Strong"
)

# Step 8: Process Search Results
if results:
    for result in results[0]:
        logging.info(f"Result ID: {result.id}, Distance: {result.distance}, Text: {result.entity.get('text')}, Subject: {result.entity.get('subject')}")
else:
    logging.warning("No search results found.")

# Step 9: Release the Collection
collection.release()
logging.info(f"Collection '{collection_name}' released from memory.")
