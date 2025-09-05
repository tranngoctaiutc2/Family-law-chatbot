# botchat_honnhan.py
import os
from datetime import datetime
from textwrap import dedent

import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ================== ENV ==================
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "luat_hon_nhan_va_gia_dinh_2014")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

if not (QDRANT_URL and QDRANT_API_KEY and GEMINI_API_KEY):
    raise RuntimeError("Thiếu QDRANT_URL / QDRANT_API_KEY / GEMINI_API_KEY trong .env")

# ================== INIT ==================
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
embedder = SentenceTransformer(EMBEDDING_MODEL)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)

# ================== SEARCH HELPERS ==================
def search_law(query: str, top_k: int = 15):
    """
    Tìm kiếm trong Qdrant và trả về danh sách điều luật + điểm tương đồng.
    Với BAAI/bge-m3 nên normalize để kết quả ổn định.
    """
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=max(1, min(int(top_k), 50)),
        with_payload=True
    )

    docs = []
    for r in results:
        p = r.payload or {}
        docs.append({
            "citation": p.get("exact_citation", ""),     # ví dụ: "Điều 56, Khoản 1, Luật HN&GĐ 2014"
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
def build_prompt(query: str, docs, history_msgs):
    # Lịch sử gọn 6 lượt gần nhất
    history_block = ""
    if history_msgs:
        lines = []
        for i, m in enumerate(history_msgs[-6:], 1):
            role = m.get("role", "")
            content = m.get("content", "")
            lines.append(f"- {i}. {role}: {content}")
        history_block = "\nLịch sử hội thoại gần đây:\n" + "\n".join(lines)

    # Danh sách Top-K điều luật cho mô hình tự chọn và viện dẫn
    context_lines = []
    for idx, d in enumerate(docs, 1):
        cited, chapter, title = law_line(d)
        content = (d.get("content") or "").strip()
        context_lines.append(f"{idx}) {cited}{chapter}{title}: {content}")
    context = "\n".join(context_lines) if context_lines else "❌ Không có điều luật nào."

    prompt = dedent(f"""
    Bạn là **trợ lý pháp lý** chuyên về **Luật Hôn nhân và Gia đình Việt Nam**, trả lời với **phong thái của một luật sư**.
    Chức năng chính của bạn:
    - Giải đáp thắc mắc về pháp luật hôn nhân và gia đình.
    - Tìm kiếm các điều luật được yêu cầu (đã cung cấp bên dưới) và trả lời người dùng.
    - Khi có danh sách Top-K điều luật, **tự chọn các điều phù hợp nhất** để trả lời và **phải trích dẫn chi tiết** (Điều/Khoản/Điểm, nguyên văn nội dung), kèm **giải thích rõ ràng**.
    - Nếu **không có điều luật phù hợp** trong danh sách, nói rõ **không có**; **tuyệt đối không được bịa**.

    Yêu cầu trả lời đầy đủ, chi tiết, dễ hiểu, dễ áp dụng đối với những câu hỏi phức tạp, còn những câu hỏi đơn giản thì có thể trả lời ngắn gọn nhưng vẫn đầy đủ ý:
    - Luôn đặt câu trả lời trong bối cảnh pháp luật Việt Nam hiện hành.
    - Văn phong chuẩn mực, mạch lạc, lập luận theo logic pháp lý.
    - Cấu trúc đề xuất:
      1) **Tóm tắt câu hỏi/tình huống** (nếu phù hợp)
      2) **Cơ sở pháp lý được trích dẫn** (chỉ từ các điều luật dưới đây; ghi rõ Điều/Khoản/Điểm; trích NGUYÊN VĂN)
      3) **Phân tích** (giải thích và áp dụng vào tình huống; nêu điều kiện áp dụng/ngoại lệ nếu có)
      4) **Kết luận/Hướng xử lý**
      5) **Lưu ý**: "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức."
    - Nếu câu hỏi **không thuộc phạm vi** Luật HN&GĐ 2014: lịch sự từ chối và nêu phạm vi bạn hỗ trợ.

    Câu hỏi hiện tại của người dùng:
    \"\"\"{query}\"\"\"{history_block}

    Các điều luật Top-K (để bạn lựa chọn khi lập luận, KHÔNG được viện dẫn ngoài danh sách này):
    {context}
    """).strip()

    return prompt

# ================== LLM STREAM ==================
def stream_answer(prompt, temperature=0.2):
    try:
        cfg = genai.types.GenerationConfig(
            temperature=float(temperature),
        )
        resp = gemini_model.generate_content(prompt, generation_config=cfg, stream=True)
        for ch in resp:
            if getattr(ch, "text", None):
                yield ch.text
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
#chatbot { height: 560px !important; }
.card {
  border: 1px solid #e5e7eb; border-radius: 16px; padding: 12px; background: #ffffffaa;
  backdrop-filter: blur(8px);
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
                type="messages",          # schema messages của Gradio 5
                bubble_full_width=False,
                show_copy_button=True,
                elem_id="chatbot",
            )
            gr.Markdown(
                "> 💡 Mẹo: Mô tả tình huống (mốc thời gian, tài sản, con chung, thỏa thuận...) để phân tích chính xác hơn.",
                elem_classes=["card"]
            )
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### ⚙️ Tuỳ chọn", elem_classes=["card"])
                with gr.Row():
                    topk = gr.Slider(5, 30, value=15, step=1, label="Số điều luật lấy (Top-K)")
                    temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature (Độ sáng tạo của mô hình ngôn ngữ lớn)")
            with gr.Group():
                gr.Markdown("### 🧾 Cơ sở pháp lý (Top-K hiển thị để kiểm tra)", elem_classes=["card"])
                # Khung Markdown có chiều cao cố định và scroll
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
        clear = gr.Button("Xoá", variant="secondary", scale=1)

    # States
    state_history = gr.State([])      # lịch sử chat theo schema messages
    state_last_answer = gr.State("")  # vẫn giữ để tiện debug/nội bộ nếu cần
    state_last_cites = gr.State("")   # markdown đã render
    state_docs = gr.State([])         # lưu full docs của lần tra cứu hiện tại
    state_page = gr.State(1)          # trang hiện tại

    # -------- Core Handler (Streaming) --------
    def respond(message, history_msgs, k, temperature, cur_page_size):
        if not (message and message.strip()):
            gr.Info("Vui lòng nhập câu hỏi.")
            return gr.update(), history_msgs, gr.update(), "", "", [], 1, "Trang 0/0"

        # 0) Phân loại: có liên quan pháp lý HN&GĐ 2014?
        legal = is_legal_query(message)
        if not legal:
            # Không tra cứu Qdrant; trả lời ngắn gọn + gợi ý
            reply = (
                "Mình chủ yếu hỗ trợ **các vấn đề pháp lý theo Luật Hôn nhân & Gia đình 2014**.\n\n"
                "Bạn có thể cho mình biết tình huống pháp lý cụ thể (ví dụ: *thủ tục ly hôn, quyền nuôi con, chia tài sản, cấp dưỡng...*)? "
                "Nếu câu hỏi không thuộc phạm vi này, mình xin phép không tra cứu để tiết kiệm tài nguyên."
            )
            upd = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            # Reset khu cơ sở pháp lý
            return gr.update(value=""), upd, gr.update(value="(Chưa có dữ liệu)"), reply, "", [], 1, "Trang 0/0"

        # 1) Tìm điều luật (chỉ chạy khi legal=True)
        try:
            docs = search_law(message, top_k=int(k))
        except Exception as e:
            err = f"Lỗi tìm kiếm Qdrant: {e}"
            upd = history_msgs + [
                {"role":"user","content":message},
                {"role":"assistant","content":err},
            ]
            return gr.update(value=""), upd, gr.update(value="(Lỗi tra cứu)"), "", "(Lỗi tra cứu)", [], 1, "Trang 0/0"

        # 2) Render trang 1 cho Cơ sở pháp lý
        first_page = 1
        cites_markdown, page_label = docs_page_markdown(docs, first_page, int(cur_page_size))

        # 3) Tạo prompt theo yêu cầu
        prompt = build_prompt(message, docs, history_msgs)

        # 4) Đẩy user + placeholder assistant (ĐÚNG SCHEMA V5)
        history_msgs = history_msgs + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},   # stream đổ vào đây
        ]

        # 5) Stream kết quả vào message cuối
        acc = ""
        for chunk in stream_answer(prompt, temperature=float(temperature)):
            acc += chunk
            history_msgs[-1]["content"] = acc
            yield (
                gr.update(value=""),                 # clear ô nhập
                history_msgs,                        # cập nhật Chatbot
                gr.update(value=cites_markdown),     # Markdown cơ sở pháp lý (trang 1)
                acc,                                 # lưu để debug/nội bộ
                cites_markdown,                      # lưu markdown hiển thị
                docs,                                # state_docs
                first_page,                          # state_page
                page_label                           # page_info
            )

    send.click(
        respond,
        inputs=[msg, state_history, topk, temp, page_size],
        outputs=[msg, chatbot, cites_md, state_last_answer, state_last_cites, state_docs, state_page, page_info],
        queue=True,
    )
    msg.submit(
        respond,
        inputs=[msg, state_history, topk, temp, page_size],
        outputs=[msg, chatbot, cites_md, state_last_answer, state_last_cites, state_docs, state_page, page_info],
        queue=True,
    )

    # -------- Like/Dislike (Gradio 5) --------
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
        # Khi đổi page_size, quay về trang 1
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

    # -------- Clear --------
    def on_clear():
        return [], "(Chưa có dữ liệu)", "", "", [], 1, "Trang 0/0"

    clear.click(
        on_clear,
        None,
        [chatbot, cites_md, state_last_answer, state_last_cites, state_docs, state_page, page_info],
        queue=False
    )

    # Footer
    gr.HTML(f"""
    <div class="footer">
      © {datetime.now().year} — Trợ lý tư vấn dựa trên Luật Hôn Nhân & Gia Đình 2014.
      Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.
    </div>
    """)

if __name__ == "__main__":
    demo.launch(show_error=True)
