import json
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# ================= Cấu hình =================
# Đường dẫn đến file service account key
SERVICE_ACCOUNT_FILE = r"D:\Project Python\family_law\Family-law-chatbot\vertex-sa.json"

# Phạm vi quyền cần thiết cho Vertex AI
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Lấy ID dự án chính xác từ file JSON
PROJECT_ID = "precise-victory-472311-h3"

# Khu vực nơi bạn sử dụng mô hình
REGION = "us-central1"

# ================= Lấy access token =================
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
credentials.refresh(Request())

token = credentials.token
print("✅ Access token đã sẵn sàng:", token)

# ================= Gọi Vertex AI API =================
# Endpoint chính xác cho mô hình Gemini Pro
url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/gemini-pro:streamGenerateContent"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json; charset=utf-8"
}

data = {
    "contents": {
        "role": "user",
        "parts": [
            {"text": "Hãy tạo một ví dụ câu hỏi về luật Hôn nhân và Gia đình."}
        ]
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print("\n✅ Phản hồi:")
print(response.text)