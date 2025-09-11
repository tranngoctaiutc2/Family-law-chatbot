import os
from datetime import datetime
from textwrap import dedent

import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import uuid
from memory import get_memory, get_history_messages, clear_history

# ================== ENV ==================
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY =  os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
EMBEDDING_MODEL =  os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "models/gemini-1.5-flash")

if not (QDRANT_URL and QDRANT_API_KEY and GEMINI_API_KEY):
    raise RuntimeError("Thiếu QDRANT_URL / QDRANT_API_KEY / GEMINI_API_KEY trong .env")

# ================== INIT ==================
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
embedder = SentenceTransformer(EMBEDDING_MODEL)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)

# ================== SEARCH HELPERS ==================
def search_law(query: str, top_k: int = 7):
    """
    Tìm kiếm trong Qdrant và trả về danh sách điều luật + điểm tương đồng.
    Với BAAI/bge-m3 nên normalize để kết quả ổn định.
    """
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=max(1, min(int(top_k), 50)),
            with_payload=True
        ).points
    except Exception as e:
        print(f"[ERROR] Qdrant query failed: {e}")
        return []

    docs = []
    for r in results:
        p = r.payload or {}
        docs.append({
            "citation": p.get("exact_citation", ""),
            "chapter": p.get("chapter", ""),
            "article_no": p.get("article_no", ""),
            "article_title": p.get("article_title", ""),
            "clause_no": p.get("clause_no", ""),
            "point_letter": p.get("point_letter", ""),
            "content": (p.get("content") or "").strip(),
            "score": float(r.score or 0.0),
        })
    return docs

def law_line(d):
    cited = d.get("citation") or (
        f"Điều {d.get('article_no','')}"
        + (f", Khoản {d.get('clause_no')}" if d.get('clause_no') else "")
        + (f", Điểm {d.get('point_letter')}" if d.get('point_letter') else "")
    )
    chapter = f" ({d.get('chapter')})" if d.get("chapter") else ""
    title = f" — {d.get('article_title')}" if d.get("article_title") else ""
    return cited, chapter, title

def docs_to_markdown(docs):
    """
    Hiển thị Top-K điều luật ở dạng Markdown (tránh lỗi [Object Object]).
    """
    if not docs:
        return "❌ Không tìm thấy điều luật nào."
    lines = []
    for i, d in enumerate(docs, 1):
        cited, chapter, title = law_line(d)
        content = (d.get("content") or "").strip()
        score = round(d.get("score", 0.0), 4)
        lines.append(
            f"**{i}. {cited}{chapter}{title}**  \n"
            f"{content}  \n"
            f"<sub>Độ liên quan: {score}</sub>\n"
        )
    return "\n".join(lines)

# -------- Phân trang cho cơ sở pháp lý --------
def paginate_docs(docs, page: int, page_size: int):
    total = len(docs)
    if total == 0:
        return [], 0, 0, 0
    page = max(1, int(page))
    page_size = max(1, int(page_size))
    start = (page - 1) * page_size
    end = start + page_size
    sliced = docs[start:end]
    total_pages = (total + page_size - 1) // page_size
    return sliced, total, total_pages, start

def docs_page_markdown(docs, page: int, page_size: int):
    sliced, total, total_pages, start = paginate_docs(docs, page, page_size)
    if total == 0:
        return "(Chưa có dữ liệu)", "Trang 0/0"
    body = docs_to_markdown(sliced)
    page_label = f"Trang {page}/{total_pages} — hiển thị {start+1}–{min(start+len(sliced), total)} / {total}"
    return f"**{page_label}**\n\n{body}", page_label

# ================== CLASSIFY: có liên quan pháp lý? ==================
def is_legal_query(user_query: str) -> bool:
    """
    Dùng Gemini để phân loại nhanh xem câu hỏi có *liên quan tới pháp lý hôn nhân & gia đình VN 2014* hay không.
    Trả về True nếu LIÊN QUAN, False nếu không.
    """
    prompt = dedent(f"""
    Hãy phân loại câu sau có LIÊN QUAN đến tư vấn pháp lý theo Luật Hôn nhân & Gia đình Việt Nam 2014 hay không.

    YÊU CẦU:
    - Nếu LIÊN QUAN: trả về đúng một từ "LEGAL".
    - Nếu KHÔNG LIÊN QUAN (xã giao, chitchat, thời tiết, công nghệ, các luật khác...): trả về đúng một từ "NONLEGAL".
    - Không thêm lời giải thích.

    CÂU CẦN PHÂN LOẠI:
    ---
    {user_query}
    ---
    """).strip()

    try:
        cfg = genai.types.GenerationConfig(temperature=0.0)
        resp = gemini_model.generate_content(prompt, generation_config=cfg)
        text = (getattr(resp, "text", None) or "").strip().upper()
        if "LEGAL" in text and "NON" not in text:
            return True
        if text == "NONLEGAL" or "NONLEGAL" in text:
            return False
        # Nếu model trả rác, fallback heuristic đơn giản theo từ khóa pháp lý
        keywords = [
            "ly hôn", "ly hon", "kết hôn", "ket hon", "hôn nhân", "hon nhan",
            "con chung", "nuôi con", "cap duong", "cấp dưỡng", "tài sản chung",
            "chia tài sản", "giành quyền", "giám hộ", "giam ho",
            "ly thân", "ly than", "điều", "khoản", "điểm", "toà", "tòa", "toà án", "tòa án",
            "giấy đăng ký kết hôn", "hủy kết hôn", "cấm kết hôn"
        ]
        q = user_query.lower()
        return any(k in q for k in keywords)
    except Exception:
        # Lỗi gọi model => dùng heuristic
        keywords = [
            "ly hôn", "ly hon", "kết hôn", "ket hon", "hôn nhân", "hon nhan",
            "con chung", "nuôi con", "cap duong", "cấp dưỡng", "tài sản chung",
            "chia tài sản", "giành quyền", "giám hộ", "giam ho",
            "ly thân", "ly than", "điều", "khoản", "điểm", "toà", "tòa", "toà án", "tòa án",
            "giấy đăng ký kết hôn", "hủy kết hôn", "cấm kết hôn"
        ]
        q = user_query.lower()
        return any(k in q for k in keywords)

# ================== PROMPT ==================
def build_prompt(query: str, docs, history_msgs, law_name="Luật Hôn nhân và Gia đình 2014"):
    # Lấy tối đa 5 lượt hội thoại gần nhất
    history_block = ""
    if history_msgs:
        lines = []
        for i, m in enumerate(history_msgs[-5:], 1):
            role = m.get("role", "")
            content = m.get("content", "")
            role_label = "Người dùng" if role == "user" else "Trợ lý"
            lines.append(f"- {i}. {role_label}: {content}")
        history_block = "\nLịch sử hội thoại gần đây:\n" + "\n".join(lines)

    # Danh sách Top-K điều luật để mô hình chọn
    context_lines = []
    for idx, d in enumerate(docs, 1):
        cited, chapter, title = law_line(d)
        content = (d.get("content") or "").strip()
        context_lines.append(f"{idx}) {cited}{chapter}{title}: {content}")
    context = "\n".join(context_lines) if context_lines else "❌ Không có điều luật nào."

    prompt = dedent(f"""
    Bạn là **trợ lý pháp luật** chuyên phân tích và tư vấn theo **{law_name}**. 
    Vai trò: giải thích luật một cách chính xác nhưng dễ hiểu, giúp người dân nắm rõ quy định và áp dụng trong đời sống. 
    Không phải chỉ đọc luật, mà cần làm rõ ý nghĩa thực tế.

    Quy tắc bắt buộc:
    - Chỉ trả lời dựa trên các điều luật được cung cấp bên dưới.
    - Nếu không tìm thấy quy định phù hợp → trả lời: "Không tìm thấy quy định trong văn bản pháp luật hiện hành."
    - Không bịa, không đưa ý kiến cá nhân ngoài phạm vi luật.
    - Độ dài câu trả lời ≤ 350 từ.

    Yêu cầu khi trả lời:
    - Trích dẫn ngắn gọn Điều/Khoản/Điểm và nguyên văn nội dung liên quan.
    - Giải thích bằng ngôn ngữ dễ hiểu, gần gũi.
    - Có thể đưa ví dụ thực tế minh họa (nếu phù hợp).
    - Trình bày theo cấu trúc:
      1) **Tóm tắt câu hỏi/tình huống**
      2) **Cơ sở pháp lý** (trích dẫn ngắn gọn luật từ danh sách bên dưới)
      3) **Phân tích** ( có ví dụ minh họa nhưng vẫn phải trích dẫn các điều luật)
      4) **Kết luận** (dựa trên phân tích ở trên, tóm tắt vấn đề bằng ngôn ngữ dễ hiểu, nêu rõ quyền lợi của các bên, hậu quả pháp lý và hướng xử lý thực tế, tránh trích dẫn luật máy móc)


    Câu hỏi hiện tại:
    \"\"\"{query}\"\"\"{history_block}

    Các điều luật Top-K (bạn CHỈ được viện dẫn trong danh sách này):
    {context}
    """).strip()

    return prompt


# ================== LLM STREAM ==================
def stream_answer(prompt, temperature=0.2):
    try:
        cfg = genai.types.GenerationConfig(
            temperature=float(temperature),
            max_output_tokens=512
        )
        resp = gemini_model.generate_content(
            prompt, generation_config=cfg, stream=True
        )

        for ch in resp:
            if getattr(ch, "text", None):
                yield ch.text  # xuất từng đoạn nhỏ ngay
    except Exception as e:
        yield f"\n\nLỗi gọi mô hình: {e}"



# ================== STYLE ==================
CSS = """
:root {
  --brand: #1f2937; /* slate-800 */
  --accent: #4f46e5; /* indigo-600 */
}
.header {
  display:flex; align-items:center; gap:12px;
  padding: 8px 12px; border-radius: 14px;
  background: linear-gradient(135deg, #eef2ff, #f8fafc);
  border: 1px solid #e5e7eb;
}
.header .title {
  font-weight: 800; font-size: 20px; color: var(--brand);
}
.header .badge {
  font-size: 12px; padding: 4px 8px; border-radius: 999px;
  background:#eef2ff; color:#4338ca; border:1px solid #c7d2fe;
}
#chatbot { 
  height: 560px !important; 
  overflow-y: auto; /* Thêm cuộn dọc */
  display: flex;
  flex-direction: column;
}
.chat-message {
  margin: 8px 0;
  padding: 8px 12px;
  border-radius: 8px;
  max-width: 80%;
}
.chat-message.user {
  background: var(--accent);
  color: white;
  align-self: flex-end;
}
.chat-message.assistant {
  background: #e5e7eb;
  color: var(--brand);
  align-self: flex-start;
}
.footer {
  font-size: 12px; opacity: .8; text-align:center; margin-top: 8px;
}
/* Khung cuộn cho Cơ sở pháp lý */
#cites_md {
  max-height: 320px;
  overflow-y: auto;
}
"""

# ================== Hàm helper & core ==================
def format_history(history_msgs):
    """
    Chuyển history_msgs sang format [{"role": "...", "content": "..."}, ...] cho gr.Chatbot
    """
    formatted = []
    for msg in history_msgs:
        role = msg.get("role")
        content = msg.get("content", "")
        # Gradio Chatbot chỉ nhận "user" hoặc "assistant"
        if role in ["user", "assistant"]:
            formatted.append({"role": role, "content": content})
    return formatted


def respond(message, history_msgs, k, temperature, cur_page_size, session_id):
    if not (message and message.strip()):
        gr.Info("Vui lòng nhập câu hỏi.")
        return gr.update(), history_msgs, gr.update(), "", "", [], 1, "Trang 0/0", session_id

    # 🔹 Generate session_id nếu chưa có
    if session_id is None:
        session_id = str(uuid.uuid4())

    # 🔹 Chỉ load history từ DB nếu chatbot rỗng
    if not history_msgs:
        history_msgs = get_history_messages(session_id)

    # 0) Phân loại câu hỏi
    legal = is_legal_query(message)
    if not legal:
        reply = (
            "Mình chủ yếu hỗ trợ **các vấn đề pháp lý theo Luật Hôn nhân & Gia Đình 2014**.\n\n"
            "Bạn có thể cho mình biết tình huống pháp lý cụ thể (ví dụ: *thủ tục ly hôn, quyền nuôi con, chia tài sản, cấp dưỡng...*)? "
            "Nếu câu hỏi không thuộc phạm vi này, mình xin phép không tra cứu để tiết kiệm tài nguyên."
        )
        mem = get_memory(session_id)
        mem.add_user_message(message)
        mem.add_ai_message(reply)

        updated_history = history_msgs + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        formatted_history = format_history(updated_history)

        return gr.update(value=""), formatted_history, gr.update(value="(Chưa có dữ liệu)"), reply, "", [], 1, "Trang 0/0", session_id

    # 1) Tìm điều luật
    try:
        docs = search_law(message, top_k=int(k))
    except Exception as e:
        err = f"Lỗi tìm kiếm Qdrant: {e}"
        mem = get_memory(session_id)
        mem.add_user_message(message)
        mem.add_ai_message(err)

        updated_history = history_msgs + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": err},
        ]
        formatted_history = format_history(updated_history)
        return gr.update(value=""), formatted_history, gr.update(value="(Lỗi tra cứu)"), "", "(Lỗi tra cứu)", [], 1, "Trang 0/0", session_id

    # 2) Render trang 1
    first_page = 1
    cites_markdown, page_label = docs_page_markdown(docs, first_page, int(cur_page_size))

    # 3) Prompt
    prompt = build_prompt(message, docs, history_msgs)

    # 4) Thêm user vào memory
    mem = get_memory(session_id)
    mem.add_user_message(message)

    # 5) Stream assistant
    acc = ""
    for chunk in stream_answer(prompt, temperature=float(temperature)):
        acc += chunk
        # Cập nhật hiển thị kiểu ChatGPT
        temp_history = history_msgs + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": acc},
        ]
        formatted_history = format_history(temp_history)
        yield (
            gr.update(value=""),
            formatted_history,  # cập nhật toàn bộ history
            gr.update(value=cites_markdown),
            acc,
            cites_markdown,
            docs,
            first_page,
            page_label,
            session_id  # Persist session_id
        )

    # 🔹 Sau khi stream xong → lưu vào memory
    mem.add_ai_message(acc)

def on_clear(session_id):
    clear_history(session_id)
    return [], "(Chưa có dữ liệu)", "", "", [], 1, "Trang 0/0", None  # Reset session_id

# ================== UI ==================
with gr.Blocks(
    title="⚖️ Trợ lý Luật HN&GĐ 2014",
    theme=gr.themes.Monochrome(primary_hue="indigo", neutral_hue="slate"),
    css=CSS
) as demo:
    # Header
    with gr.Row():
        gr.HTML("""
        <div class="header">
          <div style="font-size:24px">⚖️</div>
          <div class="title">Trợ lý Luật Hôn Nhân & Gia Đình (2014)</div>
          <div class="badge">Luật sư ảo trực tuyến</div>
          <div class="badge">Chỉ mang tính tham khảo</div>
        </div>
        """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[],
                type="messages",
                show_copy_button=True,
                elem_id="chatbot",
                bubble_full_width=False,  # Tin nhắn không chiếm toàn chiều rộng
            )
            gr.Markdown(
                "> 💡 Mẹo: Mô tả tình huống (mốc thời gian, tài sản, con chung, thỏa thuận...) để phân tích chính xác hơn.",
                elem_classes=["card"]
            )
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### ⚙️ Tuỳ chọn", elem_classes=["card"])
                with gr.Row():
                    topk = gr.Slider(5, 30, value=20, step=1, label="Số điều luật lấy (Top-K)")
                    temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature (Độ sáng tạo của mô hình ngôn ngữ lớn)")
            with gr.Group():
                gr.Markdown("### 🧾 Cơ sở pháp lý (Top-K hiển thị để kiểm tra)", elem_classes=["card"])
                cites_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="cites_md")
                with gr.Row():
                    prev_page = gr.Button("⬅️ Trang trước")
                    next_page = gr.Button("Trang sau ➡️")
                with gr.Row():
                    page_info = gr.Markdown("Trang 0/0")
                    page_size = gr.Slider(3, 20, value=5, step=1, label="Số mục mỗi trang")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Nhập câu hỏi/tình huống của bạn...",
            scale=4,
            autofocus=True,
            container=True,
        )
        send = gr.Button("Gửi", variant="primary", scale=1)
        clear_btn = gr.Button("Xoá", variant="secondary", scale=1)

    # States
    state_session = gr.State(None)  # Sửa: None để sinh UUID per session
    state_last_answer = gr.State("")
    state_last_cites = gr.State("")
    state_docs = gr.State([])
    state_page = gr.State(1)

    # -------- Click/Submit bindings --------
    send.click(
        respond,
        inputs=[msg, chatbot, topk, temp, page_size, state_session],
        outputs=[msg, chatbot, cites_md, state_last_answer, state_last_cites, state_docs, state_page, page_info, state_session],  # Thêm state_session
        queue=True,
    )
    msg.submit(
        respond,
        inputs=[msg, chatbot, topk, temp, page_size, state_session],
        outputs=[msg, chatbot, cites_md, state_last_answer, state_last_cites, state_docs, state_page, page_info, state_session],  # Thêm state_session
        queue=True,
    )

    clear_btn.click(
        on_clear,
        inputs=[state_session],
        outputs=[chatbot, cites_md, state_last_answer, state_last_cites, state_docs, state_page, page_info, state_session],  # Thêm state_session
        queue=False
    )

    # -------- Like/Dislike --------
    def on_like(data: gr.LikeData):
        msg_like = data.value or {}
        role = msg_like.get("role", "assistant")
        text = msg_like.get("content", "")
        print(f"[VOTE] liked={data.liked} | role={role} | text={(text[:120]+'...') if len(text)>120 else text}")
        return None

    chatbot.like(on_like)

    # -------- Pagination Handlers --------
    def render_cites_for_page(docs, page, cur_page_size):
        md, label = docs_page_markdown(docs or [], int(page), int(cur_page_size))
        return gr.update(value=md), int(page), label

    def go_prev(docs, page, cur_page_size):
        if not docs:
            return render_cites_for_page([], 1, cur_page_size)
        _, _, _, _ = paginate_docs(docs, 1, int(cur_page_size))
        new_page = max(1, int(page) - 1)
        return render_cites_for_page(docs, new_page, cur_page_size)

    def go_next(docs, page, cur_page_size):
        if not docs:
            return render_cites_for_page([], 1, cur_page_size)
        _, total, total_pages, _ = paginate_docs(docs, 1, int(cur_page_size))
        new_page = min(total_pages if total_pages > 0 else 1, int(page) + 1)
        return render_cites_for_page(docs, new_page, cur_page_size)

    def on_change_page_size(docs, cur_page_size):
        return render_cites_for_page(docs, 1, cur_page_size)

    prev_page.click(
        go_prev,
        inputs=[state_docs, state_page, page_size],
        outputs=[cites_md, state_page, page_info],
        queue=False,
    )
    next_page.click(
        go_next,
        inputs=[state_docs, state_page, page_size],
        outputs=[cites_md, state_page, page_info],
        queue=False,
    )
    page_size.release(
        on_change_page_size,
        inputs=[state_docs, page_size],
        outputs=[cites_md, state_page, page_info],
        queue=False,
    )

    # Footer
    gr.HTML(f"""
    <div class="footer">
      © {datetime.now().year} — Trợ lý tư vấn dựa trên Luật Hôn Nhân & Gia Đình 2014.
      Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.
    </div>
    """)


if __name__ == "__main__":
    demo.launch()