from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ================== Cấu hình ==================
QDRANT_URL = "https://0b8720b3-9ed5-417b-bdb9-8bc6207ae487.eu-west-2-0.aws.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xia-hOQAAZmHndTFEk4Cct7BsbnKtDaZsEj3usU5f5E"
COLLECTION_NAME = "Family_Law"
EMBEDDING_MODEL = "BAAI/bge-m3"  # dimension 4096

# ================== Khởi tạo client + embedding ==================
client = QdrantClient(url=QDRANT_URL, api_key=API_KEY, check_compatibility=False)
model = SentenceTransformer(EMBEDDING_MODEL)

# ================== User query ==================
user_query = "Điều 50 Luật Hôn nhân và Gia đình về vô hiệu thỏa thuận tài sản"
query_vector = model.encode(user_query).tolist()

# ================== Query vector search ==================
results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=20  # số chunk muốn lấy
)

# ================== Stitch các chunk Điều 50 ==================
combined_text = ""
for hit in results:
    payload = hit.payload
    if payload.get("article_no") == 50:  # filter Điều 50
        combined_text += payload.get("content", "") + "\n---\n"

print("Kết quả tìm kiếm Điều 50:\n")
print(combined_text)
