# Family Law Chatbot 🤖⚖️

Dự án **Family Law Chatbot** là một hệ thống hỏi đáp về **Luật Hôn nhân và Gia đình** Việt Nam.
Ứng dụng sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp với **cơ sở dữ liệu vector (Qdrant)** và **mô hình ngôn ngữ lớn Gemini 2.5 Flash** để trả lời chính xác các câu hỏi pháp lý.

---

## 🚀 Tính năng chính

* **Xử lý văn bản luật** từ file `.docx` hoặc `.txt` → tách thành các đoạn nhỏ (chunk) kèm metadata.
* **Embedding & lưu trữ** vào Qdrant để truy vấn nhanh chóng.
* **Hỏi đáp pháp lý**: chatbot trả lời câu hỏi của người dùng, có trích dẫn cơ sở pháp lý liên quan.

---

## 🛠️ Cài đặt môi trường

### 1. Clone repo

```bash
git clone https://github.com/tranngoctaiutc2/Family-law-chatbot.git
cd Family-law-chatbot
```

### 2. Tạo môi trường ảo Python

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Thiết lập biến môi trường (`.env`)

Tạo file `.env` trong thư mục gốc với nội dung mẫu:

```env
QDRANT_URL=
QDRANT_API_KEY=

GEMINI_API_KEY=
EMBEDDING_MODEL=BAAI/bge-m3
COLLECTION_NAME=
BATCH_SIZE=256
QDRANT_TIMEOUT=120
MONGO_URI=
GEMINI_MODEL_ID=models/gemini-1.5-flash

```

---

## 📂 Luồng xử lý dữ liệu

### 1. `chunking.py`

* **Đầu vào**: file văn bản luật (`.docx` hoặc `.txt`, UTF-8).
* **Xử lý**: chia nhỏ văn bản thành nhiều đoạn (chunk) kèm **metadata**:

  ```python
  {
      "base": base,
      "chapter": chapter,
      "section": section,
      "article_no": article_no,
      "article_title": article_title,
      "clause_no": clause_no,
      "point_letter": letter,
      "exact_citation": exact
  }
  ```
* **Đầu ra**: file JSON chứa dữ liệu đã chunking.

📌 Đây là bước tiền xử lý để chatbot có thể hiểu và trích dẫn chính xác các điều luật.

---

### 2. `Import_Data.py`

* **Đầu vào**: JSON từ bước `chunking.py`.
* **Xử lý**:

  * Sinh embedding cho từng chunk bằng `SentenceTransformer`.
  * Lưu embedding + metadata vào cơ sở dữ liệu vector **Qdrant**.
* **Mục tiêu**: chuẩn bị dữ liệu để chatbot có thể tìm kiếm văn bản pháp luật nhanh chóng.

---

### 3. `botchat_honnhan.py`

* **Đầu vào**: câu hỏi của người dùng.
* **Xử lý**:

  * Truy vấn Qdrant để lấy top các đoạn luật liên quan.
  * Dùng **Gemini 2.5 Flash** để sinh câu trả lời, có tham chiếu pháp lý.
* **Đầu ra**: câu trả lời tự nhiên, dễ hiểu, kèm trích dẫn điều luật.

---

## 🏃 Cách chạy dự án

1. Chunking văn bản luật:

   ```bash
   python chunking.py --input "luat_hon_nhan_va_gia_dinh.docx" --output "hn2014_chunks.json" --law-no "52/2014/QH13" --law-title "Luật Hôn nhân và Gia đình" --law-id "HN2014"
   ```

2. Import dữ liệu vào Qdrant:

   ```bash
   python Import_Data.py
   ```

3. Chạy chatbot:

   ```bash
   python botchat_honnhan.py
   ```

👉 Sau đó mở giao diện Gradio và bắt đầu hỏi chatbot!

---

## 📌 Công nghệ sử dụng

* **Python**
* **Gradio** (UI)
* **SentenceTransformers + Torch** (Embedding)
* **Qdrant** (Vector Database)
* **Gemini 2.5 Flash API** (LLM)
* **dotenv, tqdm** (tiện ích)

---

## 📖 Ghi chú

* Repo này phục vụ mục đích học tập & thử nghiệm.
* Không thay thế cho tư vấn pháp lý chuyên nghiệp.

---
