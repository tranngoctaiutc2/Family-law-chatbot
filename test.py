from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests
import json

SERVICE_ACCOUNT_FILE = r"D:\Project Python\family_law\Family-law-chatbot\vertex-sa.json"


# Scope riêng cho Vertex AI / Gemini
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Tạo credentials
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# Refresh token
credentials.refresh(Request())

print("✅ Access token ready:", credentials.token)

# Gọi API Gemini
url = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText"
headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json; charset=utf-8"
}
data = {
    "prompt": {"text": "Hỏi một câu ví dụ về luật HN&GĐ."},
    "temperature": 0.7,
    "maxOutputTokens": 200
}

resp = requests.post(url, headers=headers, data=json.dumps(data))
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
