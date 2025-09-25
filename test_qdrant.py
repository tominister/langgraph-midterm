from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to Qdrant (running locally on Docker)
client = QdrantClient(host="localhost", port=6333)

# Create a collection (recreate = drop if exists)
client.recreate_collection(
    collection_name="demo",
    vectors_config=VectorParams(size=3, distance=Distance.COSINE)
)

# Upload a few dummy vectors using PointStruct (not tuples!)
points = [
    PointStruct(id=1, vector=[0.1, 0.2, 0.3], payload={"name": "Item 1"}),
    PointStruct(id=2, vector=[0.3, 0.2, 0.1], payload={"name": "Item 2"}),
]

client.upload_points(
    collection_name="demo",
    points=points
)

# Search similar vector (yes, search still works even with warning)
hits = client.search(
    collection_name="demo",
    query_vector=[0.1, 0.2, 0.3],
    limit=1
)

print(hits)
