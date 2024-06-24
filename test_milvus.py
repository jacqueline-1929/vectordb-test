from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility, Index
import numpy as np

# Connect to Milvus server
connections.connect("default", host="localhost", port="19530")

# Define a field schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)  # Adjust the dimension as needed
]

# Define a collection schema
schema = CollectionSchema(fields)

# Create a collection
collection_name = "image_collection"

# Drop collection if it already exists
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)
print(f"Collection created: {collection.name}")

# Insert data into the collection
ids = [i for i in range(10)]
embeddings = np.random.rand(10, 128).tolist()
entities = [
    ids,
    embeddings
]

# Insert the entities into the collection
collection.insert(entities)

# Create an index for the collection
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

index = Index(collection, "embedding", index_params)
print("Index created")

# Load the collection into memory
collection.load()

# Search in the collection
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(embeddings[:1], "embedding", param=search_params, limit=3)

# Print search results
for result in results:
    print(result)
