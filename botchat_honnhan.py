import os
from datetime import datetime
from textwrap import dedent

import logging
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from tenacity import retry, stop_after_attempt, wait_exponential

# ================== ENV ==================
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY =  os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
EMBEDDING_MODEL =  os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "models/gemini-1.5-flash")

INTENT_DEBUG = os.getenv("INTENT_DEBUG", "0").strip() in {"1", "true", "TRUE", "yes", "on"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CASUAL_MAX_WORDS = int(os.getenv("CASUAL_MAX_WORDS", "0").strip() or 0)  # 0 = không giới hạn
INTENT_RAW_PREVIEW_LIMIT = int(os.getenv("INTENT_RAW_PREVIEW_LIMIT", "240").strip() or 240) #log ngắn gọn hơn
INTENT_FALLBACK_CASUAL = os.getenv(
    "INTENT_FALLBACK_CASUAL",
    "Chào bạn, mình có thể hỗ trợ câu hỏi về Luật Hôn nhân & Gia đình. Bạn muốn hỏi nội dung gì?",# câu hỏi mặc định khi 0 hiểu input
).strip()

if not (QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("Thiếu QDRANT_URL / QDRANT_API_KEY trong .env")

# ================== LOGGING (rõ ràng) ==================
class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        extras_dict = getattr(record, "__kv__", {})
        if isinstance(extras_dict, dict):
            for k, v in extras_dict.items():
                try:
                    extras.append(f"{k}={v}")
                except Exception:
                    continue
        return base + (" | " + ",".join(extras) if extras else "")

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
handlers = [logging.StreamHandler(), logging.FileHandler("botchat_honnhan.log", encoding="utf-8")]
for h in handlers:
    h.setFormatter(KVFormatter(LOG_FORMAT))

root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
for h in handlers:
    root_logger.addHandler(h)

app_log = logging.getLogger("app")
metrics_logger = logging.getLogger("metrics")
if not metrics_logger.handlers:
    fh = logging.FileHandler("metrics.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    metrics_logger.addHandler(fh)
    metrics_logger.setLevel(logging.INFO)


def log_step(event: str, **kv):
    kvpairs = ",".join([f"{k}={v}" for k, v in kv.items()])
    metrics_logger.info(f"ts={int(time.time())},evt={event},{kvpairs}")




# ================== INIT ==================
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
embedder = SentenceTransformer(EMBEDDING_MODEL)

genai.configure()
INTENT_SYSTEM_PROMPT = dedent("""
Bạn là trợ lý về Luật HN&GĐ VN.
Hãy trả về **JSON THUẦN** (không markdown, không lời dẫn).

Schema một trong các dạng:
1) {"intent":"casual","answer":"..."}
2) {"intent":"legal_answer","normalized_query":"...","original_query":"..."}
3) {"intent":"law_search","filters":{"article_no":int?,"clause_no":int?,"point_letter":str?,"chapter_number":int?}}

Quy tắc xác định intent:
- Câu hỏi nhắm tới nội dung điều/khoản/chương/mục cụ thể → law_search.
- Xã giao/chào hỏi → casual.
- Nhắc số điều/khoản nhưng hỏi tình huống thực tế, áp dụng, thủ tục → legal_answer.
- Luôn dựa vào **mục đích câu hỏi**, không chỉ số điều/khoản.

Nếu intent = casual thì bắt buộc có answer (tiếng Việt, lịch sự).
""")
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_ID,
    system_instruction=INTENT_SYSTEM_PROMPT,
)
answer_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_ID,
)
# ================== HELPERS ==================
def _safe_truncate(text: str, limit: int = 800) -> str:
    return text if text and len(text) <= limit else (text[:limit] + "…(cắt)") if text else ""

LEGAL_HINTS = re.compile(
    r"(?i)\b(điều|khoản|điểm|chương|hôn nhân|ly hôn|ly thân|nuôi con|tài sản|"
    r"quan hệ vợ chồng|kết hôn|hủy kết hôn|chung sống như vợ chồng|cấp dưỡng|giám hộ)\b"
)

def looks_like_legal(query: str) -> bool:
    return bool(LEGAL_HINTS.search(query or ""))

class SimpleTTLCache:
    def __init__(self, ttl_seconds: int = 1800, max_items: int = 512):
        self.ttl = ttl_seconds
        self.max = max_items
        self.store: Dict[str, Tuple[float, Any]] = {}

    def _evict_if_needed(self):
        if len(self.store) <= self.max:
            return
        oldest_key = min(self.store, key=lambda k: self.store[k][0])
        self.store.pop(oldest_key, None)

    def get(self, key: str):
        item = self.store.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any):
        self.store[key] = (time.time(), value)
        self._evict_if_needed()

embed_cache = SimpleTTLCache(ttl_seconds=3600, max_items=1024)
search_cache = SimpleTTLCache(ttl_seconds=900, max_items=1024)


def encode_query(text: str):
    key = f"{EMBEDDING_MODEL}|query|{text}"
    v = embed_cache.get(key)
    if v is not None:
        return v
    vec = embedder.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()
    embed_cache.set(key, vec)
    return vec


# ================== INTENT (xử lý và phân loại câu hỏi) ==================

def _intent_via_gemini(query: str) -> Dict[str, Any]:#AI phân loại intent.
    try:
        cfg = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=192,
            response_mime_type="application/json",
        )
        resp = gemini_model.generate_content(
            [
                {
                    "role": "user",
                    "parts": [f"Câu hỏi: {query}\nHãy trả JSON thuần phù hợp schema đã nêu."],
                }
            ],
            generation_config=cfg,
        )

        # Thu thập thông tin chi tiết về candidates / finish_reason / safety
        candidates = getattr(resp, "candidates", None) or []
        first_cand = candidates[0] if candidates else None
        finish_reason = getattr(first_cand, "finish_reason", None)
        safety = []
        try:
            if first_cand and getattr(first_cand, "safety_ratings", None):
                for s in first_cand.safety_ratings:
                    cat = getattr(s, "category", "")
                    prob = getattr(s, "probability", "")
                    safety.append(f"{cat}:{prob}")
        except Exception:
            pass

        raw = getattr(resp, "text", "") or ""
        app_log.info(
            "INTENT_RAW",
            extra={
                "__kv__": {
                    "len": len(raw),
                    "preview": _safe_truncate(raw, INTENT_RAW_PREVIEW_LIMIT),
                    "candidates": len(candidates),
                    "finish_reason": finish_reason,
                    "safety": ";".join(safety[:6]),
                }
            },
        )

        # Nếu bị block (finish_reason == 2) hoặc không có text => fallback casual trực tiếp
        if finish_reason == 2 or not raw:
            log_step("intent_block", reason=str(finish_reason))
            app_log.warning(
                "INTENT_BLOCKED", extra={"__kv__": {"finish_reason": finish_reason, "raw_len": len(raw)}}
            )
            return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}

        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            app_log.warning("INTENT_NON_DICT")
            return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}
        out: Dict[str, Any] = {}
        for k in ("intent", "answer", "normalized_query", "filters", "original_query"):
            if k in data and data[k] not in (None, ""):
                out[k] = data[k]
        app_log.info(
            "INTENT_PARSED",
            extra={
                "__kv__": {
                    "intent": out.get("intent", ""),
                    "has_answer": int("answer" in out and bool(out.get("answer"))),
                    "has_filters": int("filters" in out and bool(out.get("filters"))),
                    "ans_len": len(out.get("answer", "") or ""),
                    "finish_reason": finish_reason,
                }
            },
        )
        return out
    except Exception as e:
        app_log.warning("INTENT_ERR", extra={"__kv__": {"err": str(e)}})
        # Fallback cuối cùng khi exception bất thường
        return {"intent": "casual", "answer": INTENT_FALLBACK_CASUAL}


def analyze_intent(query: str) -> Dict[str, Any]:#bộ lọc + fallback thủ công để chắc chắn lúc nào cũng có intent hợp lệ.
    data = _intent_via_gemini(query)
    intent = data.get("intent")
    answer = data.get("answer", "")
    normalized_query = data.get("normalized_query", "") or query
    original_query = data.get("original_query", "") 
    filters = data.get("filters", {}) or {}

    if intent not in {"casual", "legal_answer", "law_search"}:
        if re.search(r"(?i)\bđiều\s*\d+", query) or re.search(r"(?i)\bkhoản\s*\d+", query):
            intent = "law_search"
        elif looks_like_legal(query):
            intent = "legal_answer"
        else:
            intent = "casual"
        log_step("intent_fallback", qlen=len(query))

    log_step("intent", loai=intent, co_legal=str(looks_like_legal(query)))
    app_log.info("INTENT_DECISION", extra={"__kv__": {"intent": intent}})
    return {"intent": intent, "answer": answer, "normalized_query": normalized_query, "original_query": original_query, "filters": filters}


# ================== HIBRID SEARCH ==================


def _build_filter(query_text: str) -> Optional[Filter]:
    conds: List[FieldCondition] = []
    m = re.search(r"(?i)\bđiều\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="article_no", match=MatchValue(value=int(m.group(1)))))
    m = re.search(r"(?i)\bkhoản\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="clause_no", match=MatchValue(value=int(m.group(1)))))
    m = re.search(r"(?i)\bđiểm\s*([a-z])\b", query_text)
    if m:
        conds.append(FieldCondition(key="point_letter", match=MatchValue(value=m.group(1).lower())))
    m = re.search(r"(?i)\bchương\s*(\d+)\b", query_text)
    if m:
        conds.append(FieldCondition(key="chapter_number", match=MatchValue(value=int(m.group(1)))))
    return Filter(must=conds) if conds else None


def search_law(query: str, top_k: int = 15, score_threshold: float = 0.42):
    t0 = time.perf_counter()
    app_log.info("SEARCH_BEGIN", extra={"__kv__": {"q": _safe_truncate(query, 80), "k": top_k, "thr": score_threshold}})

    cache_key = f"search|{COLLECTION_NAME}|{top_k}|{score_threshold}|{query}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        app_log.info("SEARCH_CACHE_HIT")
        return cached

    try:
        t_embed0 = time.perf_counter()
        vec = encode_query(query)
        t_embed = time.perf_counter() - t_embed0

        flt = _build_filter(query)

        t_q0 = time.perf_counter()
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            with_payload=True,
            limit=max(1, min(int(top_k) * 3, 80)),
            query_filter=flt
        )
        t_qdrant = time.perf_counter() - t_q0

        raw_docs = []
        for r in results:
            p = r.payload or {}
            raw_docs.append({
                "citation": p.get("exact_citation", ""),
                "chapter_number": p.get("chapter_number", ""),
                "article_no": p.get("article_no", ""),
                "article_title": p.get("article_title", ""),
                "clause_no": p.get("clause_no", ""),
                "point_letter": p.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
                "score": float(r.score or 0.0),
            })

        t_filter0 = time.perf_counter()
        filtered = [d for d in raw_docs if d.get("score", 0.0) >= score_threshold] or raw_docs[:max(1, top_k)]
        t_filter = time.perf_counter() - t_filter0

        selected = filtered[:top_k]
        search_cache.set(cache_key, selected)

        sk_top1 = selected[0]['score'] if selected else 0.0
        log_step(
            "tim_kiem",
            k_yeu_cau=top_k,
            k_tra=f"{len(selected)}",
            sk_top1=f"{sk_top1:.4f}",
            t_embed=f"{t_embed:.4f}",
            t_qdrant=f"{t_qdrant:.4f}",
            t_loc=f"{t_filter:.4f}",
            t_tong=f"{time.perf_counter()-t0:.4f}"
        )
        app_log.info("SEARCH_DONE", extra={"__kv__": {"count": len(selected), "top1": f"{sk_top1:.4f}"}})
        return selected
    except Exception as e:
        app_log.error("SEARCH_ERR", extra={"__kv__": {"err": str(e)}})
        log_step("tim_kiem_loi", msg=str(e))
        raise

# ================== RENDER UTILS ==================

def law_line(d: Dict[str, Any]) -> Tuple[str, str, str]:
    art = d.get("article_no"); cls = d.get("clause_no"); pt  = d.get("point_letter")
    parts = []
    if pt: parts.append(f"Điểm {pt}")
    if cls: parts.append(f"khoản {cls}")
    if art: parts.append(f"điều {art}")
    cited = " ".join(parts)
    chapter = f" (Chương {d.get('chapter_number')})" if d.get("chapter_number") else ""
    title = f" — {d.get('article_title')}" if d.get("article_title") else ""
    return cited, chapter, title


def docs_to_markdown(docs: List[Dict[str, Any]]):
    if not docs:
        return "❌ Không tìm thấy điều luật nào."
    lines = []
    for i, d in enumerate(docs, 1):
        cited, chapter, title = law_line(d)
        content = (d.get("content") or "").strip()
        score = round(d.get("score", 0.0), 4)
        lines.append(
            f"**{i}.  {cited}{chapter}{title}**  \n" +
            f"{content}  \n" +
            f"<sub>Độ liên quan: {score}</sub>\n"
        )
    return "\n".join(lines)

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


# ================== PROMPT ==================
def build_prompt(query: str, docs: List[Dict[str, Any]], history_msgs=None, ):
    
    history_block = ""
    if history_msgs:
        lines = []
        for i, m in enumerate(history_msgs[-5:], 1):
            role = m.get("role", "")
            content = m.get("content", "")
            role_label = "Người dùng" if role == "user" else "Trợ lý"
            lines.append(f"- {i}. {role_label}: {content}")
        history_block = "\nLịch sử hội thoại gần đây:\n" + "\n".join(lines)

    docs_sorted = sorted(
        docs,
        key=lambda d: (
            int(d.get("article_no") or 9999),
            int(d.get("clause_no") or 9999),
            str(d.get("point_letter") or "")
        )
    )

    context_lines = []
    for idx, d in enumerate(docs_sorted, 1):
        art = d.get("article_no"); cls = d.get("clause_no"); pt  = d.get("point_letter")
        parts = []
        if pt: parts.append(f"Điểm {pt}")
        if cls: parts.append(f"khoản {cls}")
        if art: parts.append(f"điều {art}")
        cited = " ".join(parts)
        chapter = f" (Chương {d.get('chapter_number')})" if d.get("chapter_number") else ""
        title = f" — {d.get('article_title')}" if d.get("article_title") else ""
        content = (d.get("content") or "").strip()
        context_lines.append(f"{idx}) {cited}{chapter}{title}: {content}")

    context = "\n".join(context_lines) if context_lines else "❌ Không có điều luật nào."

    prompt = dedent(f"""
    Bạn là luật sư tư vấn về Luật Hôn nhân & Gia Đình. Chỉ dùng các trích đoạn trong danh sách dưới đây 
    Quy tắc:
    - Nếu câu hỏi là nhận định Đúng/Sai → trả lời **Kết luận: Đúng/Sai** + lý do.
    - Nếu câu hỏi thường → trả lời **ngắn gọn 1–3 câu**, bám sát câu hỏi.
    - **Trích dẫn nguyên văn** các điều luật liên quan trong danh sách (không bỏ sót), theo thứ tự: Điểm – Khoản – Điều + nội dung.
    - Nếu thiếu căn cứ, trả lời: **“Không đủ căn cứ.”**
    - Nếu câu hỏi không liên quan đến luật → trả lời lịch sự, ngắn gọn, không viện dẫn luật.
    ĐỊNH DẠNG TRẢ LỜI:
    - Trích dẫn: <liệt kê toàn bộ Điểm–Khoản–Điều + nội dung nguyên văn>
    - Giải thích: <1–3 câu, áp dụng vào tình huống>
    - Kết luận: <kết luận ngắn gọn dựa vào câu hỏi và giải thích>


    Câu hỏi hiện tại:
    \"\"\"{query}\"\"\"{history_block}

    Danh sách điều luật (top_k):
    {context}
    """).strip()

    return prompt

# ================== LLM STREAM ==================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gemini_stream(prompt, temperature: float):
    cfg = genai.types.GenerationConfig(temperature=float(temperature))
    return answer_model.generate_content(prompt, generation_config=cfg, stream=True)


def stream_answer(prompt, temperature=0.2):
    t0 = time.perf_counter(); t_first0 = time.perf_counter(); first_token_emitted = False
    try:
        resp = _gemini_stream(prompt, temperature)
        for ch in resp:
            if getattr(ch, "text", None):
                if not first_token_emitted:
                    log_step("llm_first_token", do_truoc=f"{time.perf_counter()-t_first0:.4f}")
                    first_token_emitted = True
                yield ch.text
    except Exception as e:
        app_log.error("LLM_ERR", extra={"__kv__": {"err": str(e)}})
        yield f"\n\nLỗi gọi mô hình: {e}"
    finally:
        log_step("llm_tong", t=f"{time.perf_counter()-t0:.4f}")

# -------- Qdrant Fetch Helper --------
def _fetch(filters: Dict[str, Any], limit: int = 10):
    must = []
    mapping = {"article_no": int, "clause_no": int, "point_letter": str, "chapter_number": int}
    for k, caster in mapping.items():
        if k in filters and filters[k] not in (None, ""):
            try:
                val = caster(filters[k])
                must.append(FieldCondition(key=k, match=MatchValue(value=val)))
            except Exception as e:
                app_log.warning("FETCH_CAST_ERROR", extra={"__kv__": {"key": k, "value": filters[k], "error": str(e)}})
                try:
                    val = str(filters[k])
                    must.append(FieldCondition(key=k, match=MatchValue(value=val)))
                except:
                    pass
    app_log.info("FETCH_FILTERS", extra={"__kv__": {"raw_filters": str(filters), "must_conditions": str(must)}})
    if not must:
        app_log.warning("FETCH_NO_MUST")
        return []
    flt = Filter(must=must)
    out = []
    try:
        scroll_res, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            limit=min(64, max(5, limit)),
            with_payload=True
        )
        app_log.info("FETCH_SCROLL_RESULTS", extra={"__kv__": {"count": len(scroll_res), "collection": COLLECTION_NAME}})
        for r in scroll_res:
            p = r.payload or {}
            out.append({
                "chapter": p.get("chapter", ""),
                "article_no": p.get("article_no", ""),
                "article_title": p.get("article_title", ""),
                "clause_no": p.get("clause_no", ""),
                "point_letter": p.get("point_letter", ""),
                "content": (p.get("content") or "").strip(),
                "score": 1.0,
            })
            if len(out) >= limit:
                break
    except Exception as e:
        app_log.error("FETCH_SCROLL_ERROR", extra={"__kv__": {"error": str(e), "collection": COLLECTION_NAME}})
        return []
    if not out:
        app_log.warning("FETCH_NO_OUT", extra={"__kv__": {"filters": str(filters)}})
    return out

# ================== UI (tối giản, đúng state) ==================
# CSS
CSS = """
#chatbot { height: 540px !important; }
label { font-size:12px !important; opacity:.9 }
#cites-box { 
    max-height: 360px; 
    overflow-y: auto; 
    border: 1px solid #ddd; 
    padding: 6px; 
    border-radius: 6px;
    background-color: #fafafa;
}
#history_box { 
    max-height: 200px;
}
"""

with gr.Blocks(
    title="⚖️ Trợ lý Luật HN&GĐ 2014",
    css=CSS,
) as demo:
    gr.Markdown("""
    ### ⚖️ Trợ lý Luật Hôn Nhân & Gia Đình 2014
    *Tham chiếu chính xác • Hạn chế suy diễn • Không thay thế tư vấn pháp lý*
    """)

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                value=[], type="messages", show_copy_button=True, elem_id="chatbot"
            )
            with gr.Row():
                ex1 = gr.Button("Chào bạn")
                ex2 = gr.Button("Điều 81 quy định gì về việc nuôi con sau ly hôn")
                ex3 = gr.Button("Khoản 2 Điều 56 nói gì")
        with gr.Column(scale=5):
            history_box = gr.Chatbot(
                value=[],
                type="messages",
                show_copy_button=False,
                label="📜 Lịch sử chat",
                elem_id="history_box"
            )
            gr.Markdown("**Cơ sở pháp lý**")
            cites_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="cites-box")
            with gr.Row():
                prev_page = gr.Button("⬅️")
                next_page = gr.Button("➡️")
            with gr.Row():
                page_info = gr.Markdown("Trang 0/0")
                page_size = gr.Slider(3, 20, value=5, step=1, label="Mỗi trang")

    with gr.Row():
        msg = gr.Textbox(placeholder="Nhập câu hỏi...", scale=5, autofocus=True)
        send = gr.Button("Gửi", variant="primary", scale=1)
        clear = gr.Button("Làm mới", scale=1)

    # Prefill examples
    def _fill(text):
        return text
    ex1.click(lambda: _fill("Chào bạn"), outputs=msg)
    ex2.click(lambda: _fill("Điều 81 quy định gì về việc nuôi con sau ly hôn"), outputs=msg)
    ex3.click(lambda: _fill("Khoản 2 Điều 56 nói gì"), outputs=msg)

    # States
    state_history = gr.State([])
    state_last_answer = gr.State("")
    state_docs = gr.State([])
    state_page = gr.State(1)

    # Helper để đảm bảo đúng thứ tự/đủ outputs
    def ui_return(msg_val, chatbot_val, cites_val, last_answer_val, docs_val, page_val, page_label_val, history_val):
        return (
            msg_val, chatbot_val, gr.update(value=cites_val), last_answer_val,
            docs_val, page_val, page_label_val, history_val
        )

    
        # -------- Core Handler (Streaming) --------
    def respond(message, history_msgs, cur_page_size, k=15 , temperature=0.2, threshold=0.42):
        if not (message and message.strip()):
            gr.Info("Vui lòng nhập câu hỏi.")
            return ui_return(gr.update(), history_msgs, "", "", [], 1, "Trang 0/0", history_msgs)

        t_overall0 = time.perf_counter()
        try:
            # ---------- intent detection (Gemini lần 1, chỉ nội bộ) ----------
            intent_info = analyze_intent(message)
            intent = intent_info["intent"]
            intent_answer = intent_info.get("answer", "")
            normalized_query = intent_info.get("normalized_query", message)
            original_query = intent_info.get("original_query", message)
            intent_filters = intent_info.get("filters", {})

            # ===== CASUAL =====
            if intent == "casual":
                final_answer = (intent_answer or "").replace("\u200b", "").strip()
                app_log.info("CASUAL_BRANCH", extra={"__kv__": {"ans_len": len(final_answer)}})

                # Giới hạn số từ nếu cấu hình
                if final_answer and CASUAL_MAX_WORDS > 0:
                    words = final_answer.split()
                    if len(words) > CASUAL_MAX_WORDS:
                        truncated = " ".join(words[:CASUAL_MAX_WORDS])
                        app_log.info(
                            "CASUAL_TRUNCATE",
                            extra={"__kv__": {"orig_words": len(words), "kept": CASUAL_MAX_WORDS, "orig_len": len(final_answer)}},
                        )
                        final_answer = truncated

                # Nếu Gemini đã trả sẵn answer → dùng luôn
                if len(final_answer) >= 1:
                    history_msgs = history_msgs + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": final_answer},
                    ]
                    yield ui_return(gr.update(value=""), history_msgs, "(Không có trích dẫn)", final_answer, [], 1, "Trang 0/0", history_msgs)
                    log_step("hoan_tat", t_tong=f"{time.perf_counter()-t_overall0:.4f}", t_llm="casual_direct")
                    return

                # Nếu không có answer → fallback gọi Gemini stream ngắn
                simple_prompt = "Trả lời thân thiện ngắn gọn (<=2 câu) tiếng Việt cho câu: " + message
                history_msgs = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": ""},
                ]
                acc = ""
                for chunk in stream_answer(simple_prompt, temperature=float(temperature)):
                    acc += chunk
                    history_msgs[-1]["content"] = acc
                    yield ui_return(gr.update(value=""), history_msgs, "(Không có trích dẫn)", acc, [], 1, "Trang 0/0", history_msgs)
                log_step("hoan_tat", t_tong=f"{time.perf_counter()-t_overall0:.4f}", t_llm="casual_stream")
                return

            # ===== LAW_SEARCH & LEGAL_ANSWER =====
            docs: List[Dict[str, Any]] = []
            source = None

            if intent == "law_search":
                docs = _fetch(intent_filters, limit=int(k)) if intent_filters else []
                source = "law_search"

            elif intent == "legal_answer":
                docs = search_law(normalized_query, top_k=int(k), score_threshold=float(threshold))
                source = "legal_answer"

            else:
                # fallback: intent không xác định
                reply = INTENT_FALLBACK_CASUAL
                history_msgs = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                ]
                yield ui_return(gr.update(value=""), history_msgs, "(Không có trích dẫn)", reply, [], 1, "Trang 0/0", history_msgs)
                return

            # Nếu không tìm thấy docs
            if not docs:
                reply = "Chưa tìm thấy cơ sở pháp lý phù hợp. Bạn có thể bổ sung Điều/Khoản hoặc thêm bối cảnh."
                upd = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                ]
                yield ui_return(gr.update(value=""), upd, "(Chưa có dữ liệu)", reply, [], 1, "Trang 0/0", upd)
                return

            # Chuẩn bị citations + prompt (cho Gemini lần 2)
            if intent == "legal_answer":
                user_query = original_query or message
            elif intent == "law_search":
                user_query = message
            else:
                user_query = message
            cites_markdown, page_label = docs_page_markdown(docs, 1, int(cur_page_size))
            prompt = build_prompt(user_query, docs, history_msgs)
            
            log_step("llm_chuanbi", k_docs=len(docs), source=source)

            # Stream Gemini answer (Gemini lần 2, trả ra UI)
            history_msgs = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            acc = ""
            t_llm0 = time.perf_counter()
            for chunk in stream_answer(prompt, temperature=float(temperature)):
                acc += chunk
                history_msgs[-1]["content"] = acc
                yield ui_return(gr.update(value=""), history_msgs, cites_markdown, acc, docs, 1, page_label, history_msgs)

            log_step("hoan_tat", source=source, t_tong=f"{time.perf_counter()-t_overall0:.4f}", t_llm=f"{time.perf_counter()-t_llm0:.4f}")
            return 

        except Exception as e:
            app_log.error("RESPOND_ERR", extra={"__kv__": {"err": str(e)}})
            return ui_return(gr.update(value=""), history_msgs, "(Lỗi hệ thống)", f"Lỗi: {e}", [], 1, "Trang 0/0", history_msgs)


    # Wiring outputs
    outputs = [msg, chatbot, cites_md, state_last_answer, state_docs, state_page, page_info, state_history]
    send.click(respond, inputs=[msg, state_history,page_size], outputs=outputs, queue=True)
    msg.submit(respond, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)


    # Like/Dislike
    def on_like(data: gr.LikeData):
        msg_like = data.value or {}
        role = msg_like.get("role", "assistant"); text = msg_like.get("content", "")
        app_log.info("FEEDBACK", extra={"__kv__": {"liked": data.liked, "role": role, "len": len(text or "")}})
        return None
    chatbot.like(on_like)

    # Pagination
    def render_cites_for_page(docs, page, cur_page_size):
        md, label = docs_page_markdown(docs or [], int(page), int(cur_page_size))
        return gr.update(value=md), int(page), label

    def go_prev(docs, page, cur_page_size):
        if not docs:
            return render_cites_for_page([], 1, cur_page_size)
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

    prev_page.click(go_prev, inputs=[state_docs, state_page, page_size], outputs=[cites_md, state_page, page_info], queue=False)
    next_page.click(go_next, inputs=[state_docs, state_page, page_size], outputs=[cites_md, state_page, page_info], queue=False)
    page_size.release(on_change_page_size, inputs=[state_docs, page_size], outputs=[cites_md, state_page, page_info], queue=False)

    gr.Markdown(f"""
    <sub>© {datetime.now().year} — Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.</sub>
    """)

if __name__ == "__main__":
    demo.launch(show_error=True)