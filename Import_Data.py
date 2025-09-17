import json
import os
from typing import List
from uuid import uuid4

from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http import models

# ============ Config ============
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))

DATA_FILE = "d:/Project Python/family_law/Family-law-chatbot/hn2014_chunks.json"

# ============ Kết nối Qdrant ============
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
    timeout=QDRANT_TIMEOUT
)

# ============ Model ============
print(f"🔎 Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
VECTOR_DIM = model.get_sentence_embedding_dimension()

# ============ Collection helper ============
def ensure_collection():
    """Tạo collection nếu chưa tồn tại (không xoá dữ liệu cũ)."""
    cols = client.get_collections().collections
    if any(c.name == COLLECTION_NAME for c in cols):
        print(f"✅ Collection '{COLLECTION_NAME}' đã tồn tại.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIM,
            distance=models.Distance.COSINE
        )
    )
    print(f"✅ Đã tạo collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}, distance=COSINE).")

# ============ Upload theo batch ============
def encode_texts(texts: List[str]) -> List[List[float]]:
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False)
    return [v.tolist() for v in (vecs if hasattr(vecs, "tolist") else vecs)]

def upload_batch(points_batch):
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points_batch,
        wait=True  # chờ index xong để tránh query thiếu
    )

def load_and_upload(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"📦 Tổng số điểm cần upload: {total}")

    global_counter = 0
    for i in tqdm(range(0, total, BATCH_SIZE), desc="🔄 Uploading"):
        chunk = data[i : i + BATCH_SIZE]

        texts = [it["content"] for it in chunk]
        vectors = encode_texts(texts)

        points = []
        for it, vec in zip(chunk, vectors):
            meta = it.get("metadata", {})
            if "point_id" in meta:
                meta.pop("point_id")   # xóa khỏi payload

            payload = {"content": it["content"], **meta}

            points.append(
                models.PointStruct(
                    id=global_counter,   # dùng id tự tăng
                    vector=vec,
                    payload=payload
                )
            )
            global_counter += 1

        upload_batch(points)

    print("🚀 Hoàn tất upload.")


if __name__ == "__main__":
    ensure_collection()
    load_and_upload(DATA_FILE)
