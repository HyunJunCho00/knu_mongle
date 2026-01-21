from qdrant_client import QdrantClient, models
from src.core.config import settings

def get_client():
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=60,
        verify=False
    )

def init_collection(client: QdrantClient):
    if not client.collection_exists(settings.COLLECTION_NAME):
        print(f"Creating collection: {settings.COLLECTION_NAME}")
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=1024, 
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8, 
                            always_ram=True
                        )
                    )
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                        full_scan_threshold=1000,
                        datatype="float32"
                    )
                )
            }
        )