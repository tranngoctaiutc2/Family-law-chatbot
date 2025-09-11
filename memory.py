import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from pydantic import Field

# ----------------------------
# Load biến môi trường
# ----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI").strip();
if not MONGO_URI:
    raise ValueError("❌ Chưa cấu hình MONGO_URI trong file .env")

# ----------------------------
# Kết nối MongoDB
# ----------------------------
try:
    mongo_client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    mongo_client.admin.command("ping")
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    exit(1)

DB_NAME = "family_law_db"
COLLECTION_NAME = "chat_histories"
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

print("Collections in DB:", db.list_collection_names())

# ----------------------------
# Hàm khởi tạo message history
# ----------------------------
class AutoSaveMemory(ConversationBufferMemory):
    session_id: str = Field(..., description="Session ID for the memory")  # Define as Pydantic Field

    def add_message(self, message):
        print(f"[DEBUG] Adding message to memory: {message}")
        super().add_message(message)
        print(f"[DEBUG] Calling save_memory for session_id: {self.session_id}")
        save_memory(self.session_id, self)

    def add_user_message(self, message):
        print(f"[DEBUG] Adding user message: {message}")
        self.chat_memory.add_user_message(message)
        print(f"[DEBUG] Calling save_memory for session_id: {self.session_id}")
        save_memory(self.session_id, self)

    def add_ai_message(self, message):
        print(f"[DEBUG] Adding AI message: {message}")
        self.chat_memory.add_ai_message(message)
        print(f"[DEBUG] Calling save_memory for session_id: {self.session_id}")
        save_memory(self.session_id, self)

def get_memory(session_id: str):
    print(f"[DEBUG] Loading memory for session_id: {session_id}")
    memory = AutoSaveMemory(
        session_id=session_id,  # Pass session_id as a keyword argument
        memory_key="chat_history",
        return_messages=True
    )
    doc = collection.find_one({"session_id": session_id})
    if doc and "messages" in doc:
        print(f"[DEBUG] Found existing messages: {doc['messages']}")
        for msg in doc["messages"]:
            if msg["type"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["type"] == "ai":
                memory.chat_memory.add_ai_message(msg["content"])
    else:
        print("[DEBUG] No existing messages found.")
    return memory

# ----------------------------
# Hàm lưu memory vào MongoDB
# ----------------------------
def save_memory(session_id: str, memory: ConversationBufferMemory):
    messages = [
        {"type": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in memory.chat_memory.messages
    ]
    print(f"[DEBUG] Saving messages for session_id {session_id}: {messages}")
    try:
        result = collection.update_one(
            {"session_id": session_id},
            {"$set": {"messages": messages, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        print(f"[DEBUG] Matched {result.matched_count}, Modified {result.modified_count}")
    except Exception as e:
        print(f"[ERROR] Failed to save to MongoDB: {e}")

# ----------------------------
# Hàm lấy toàn bộ history
# ----------------------------
def get_history_messages(session_id: str):
    doc = collection.find_one({"session_id": session_id})
    if doc and "messages" in doc:
        return [
            {"role": m["type"], "content": m["content"]}
            for m in doc["messages"]
        ]
    return []

# ----------------------------
# Hàm xóa history
# ----------------------------
def clear_history(session_id: str):
    result = collection.delete_one({"session_id": session_id})
    print(f"[DEBUG] Deleted {result.deleted_count} document(s) for session_id {session_id}")

# ----------------------------
# Test thử
# ----------------------------
if __name__ == "__main__":
    session_id = "test_session_1"
    memory = get_memory(session_id)

    # Thêm tin nhắn
    memory.add_user_message("Hello!")
    memory.add_ai_message("Hi there!")

    print("Chat history in MongoDB:", get_history_messages(session_id))