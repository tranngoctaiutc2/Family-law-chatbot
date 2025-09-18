from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
# ================== Cấu hình ==================
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
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
