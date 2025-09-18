import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Load biến môi trường từ .env
load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Family_Law")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Kết nối tới Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120
)

indexes = [
    ("article_no", rest.PayloadSchemaType.INTEGER),   # Điều số mấy
    ("clause_no", rest.PayloadSchemaType.INTEGER),    # Khoản số mấy
    ("point_letter", rest.PayloadSchemaType.KEYWORD), # Điểm a, b, c...
    ("chapter_number", rest.PayloadSchemaType.INTEGER), # Chương số mấy
    ("point_id", rest.PayloadSchemaType.KEYWORD),     # Ví dụ: dieu_50_khoan_1_diem_b
]


for field, schema in indexes:
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
        print(f"✅ Created index for {field} ({schema})")
    except Exception as e:
        print(f"⚠️ Failed to create index for {field}: {e}")

print("🎉 Done.")
