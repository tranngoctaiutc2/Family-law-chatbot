#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
law_chunk_agent.py
------------------
Flow:
1) Nhận file (.docx/.txt) -> đọc & chuẩn hoá dòng
2) Chunking strict (Chương/Điều/Khoản/Điểm; chuỗi điểm phải bắt đầu a); tiêm intro khoản vào mọi điểm)
3) (Tuỳ chọn) Khi có cờ --AI: gọi Gemini (model cố định: gemini-2.5-flash) để RÀ SOÁT chunking.
   - Nếu phát hiện vấn đề → in báo cáo + ghi <output>.issues.json
   - Nếu OK → ghi <output>.json (mảng chunks)


# Chỉ chunking, KHÔNG gọi Gemini:
python AI_Agent_Chunking.py --input "luat_hon_nhan_va_gia_dinh.docx" --output "hn2014_chunks.json" --law-no "52/2014/QH13" --law-title "Luật Hôn nhân và Gia đình" --law-id "HN2014"

# Chunking + gọi Gemini đánh giá (--AI):
python AI_Agent_Chunking.py --input "data/luat_hon_nhan_va_gia_dinh.docx" --output "hn2014_chunks.json" --AI --law-no "52/2014/QH13" --law-title "Luật Hôn nhân và Gia đình" --law-id "HN2014" --issued-date "2014-06-19" --effective-date "2014-06-19" --signer "Chủ tịch Quốc hội Nguyễn Sinh Hùng"
python AI_Agent_Chunking.py --input "data/luat_xay_dung.docx" --output "xd2014_chunks.json" --AI --law-no "50/2014/QH13" --law-title "Luật Xây dựng" --law-id "XD2014"
python AI_Agent_Chunking.py --input "data/luat_dau_tu.docx" --output "dt2020_chunks.json" --AI --law-no "61/2020/QH14" --law-title "Luật Đầu tư" --law-id "DT2020"

"""
import argparse
import json
import os
import pathlib
import re
import sys
import unicodedata
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv  # pip install python-dotenv

# ====== CẤU HÌNH AGENT ======
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # cố định theo yêu cầu

# ===== Regex (đầu dòng) =====
ARTICLE_RE = re.compile(r'^Điều\s+(\d+)\s*[\.:]?\s*(.*)$', re.UNICODE)
CHAPTER_RE = re.compile(r'^Chương\s+([IVXLCDM]+)\s*(.*)$', re.UNICODE | re.IGNORECASE)
CLAUSE_RE = re.compile(r'^\s*(\d+)\.\s*(.*)$', re.UNICODE)  # Khoản: PHẢI "1."
POINT_RE  = re.compile(r'^\s*([a-zA-ZđĐ])[\)\.]\s+(.*)$', re.UNICODE)  # Điểm: "a)" hoặc "a."

# ===== .docx reader =====
try:
    from docx import Document  # pip install python-docx
except ImportError:
    Document = None

def read_text(input_path: pathlib.Path) -> str:
    suf = input_path.suffix.lower()
    if suf == ".docx":
        if Document is None:
            raise RuntimeError("Thiếu python-docx. Cài: pip install python-docx")
        doc = Document(str(input_path))
        return "\n".join((p.text or "").strip() for p in doc.paragraphs)
    elif suf == ".txt":
        return input_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise RuntimeError("Chỉ hỗ trợ .docx và .txt (hãy Save As .doc → .docx).")

def normalize_lines(text: str) -> List[str]:
    """Normalize lines for more robust header matching:
    - NFC unicode normalization (fix decomposed diacritics like `ề`)
    - replace common no-break spaces with normal spaces
    - strip trailing whitespace and remove BOM
    """
    if text is None:
        return []
    # Normalize unicode to composed form so regex matches decorated characters
    text = unicodedata.normalize("NFC", text)
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        if ln is None:
            continue
        # Replace common non-breaking / narrow no-break spaces
        ln = ln.replace('\u00A0', ' ').replace('\u202F', ' ').replace('\u2009', ' ')
        # Remove BOM if present
        ln = ln.replace('\ufeff', '')
        # Trim trailing whitespace
        ln = re.sub(r'\s+$', '', ln)
        out.append(ln)
    return out

# ===== Roman → int =====
ROMAN_MAP = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
def roman_to_int(s: str) -> Optional[int]:
    s = s.upper().strip()
    if not s or any(ch not in ROMAN_MAP for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = ROMAN_MAP[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total

# ===== PASS 1: Pre-scan CHƯƠNG/ĐIỀU (đầu dòng) =====
def prescan(lines: List[str]) -> Tuple[List[int], List[int], List[str], List[str]]:
    chapters_nums, articles_nums = [], []
    chapters_labels, article_titles_seen = [], []
    expecting_chapter_title = False
    roman_current = None
    for raw in lines:
        line = raw
        if not line:
            continue
        if expecting_chapter_title:
            if not (CHAPTER_RE.match(line) or CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                ch_title = line.strip()
                lbl = f"Chương {roman_current} – {ch_title}"
                chapters_labels[-1] = lbl  # cập nhật label cuối
                expecting_chapter_title = False
                continue
            else:
                expecting_chapter_title = False
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            n = roman_to_int(m_ch.group(1))
            if n:
                chapters_nums.append(n)
                title = (m_ch.group(2) or "").strip()
                lbl = f"Chương {m_ch.group(1).strip()}" + (f" – {title}" if title else "")
                chapters_labels.append(lbl)
                if not title:
                    expecting_chapter_title = True
                    roman_current = m_ch.group(1).strip()
                else:
                    expecting_chapter_title = False
            continue
        m_art = ARTICLE_RE.match(line)
        if m_art:
            num = int(m_art.group(1))
            articles_nums.append(num)
            article_titles_seen.append((m_art.group(2) or "").strip())
            continue
    return chapters_nums, articles_nums, chapters_labels, article_titles_seen

def build_article_header(article_no: int, article_title: str) -> str:
    t = (article_title or "").strip()
    return f"Điều {article_no}." + (f" {t}" if t else "")

# ===== Heuristic intro Điều dành cho các khoản =====
INTRO_CUE_PAT = re.compile(r'(sau đây|bao gồm|gồm các|quy định như sau)\s*:\s*$', re.IGNORECASE | re.UNICODE)
def is_intro_text_for_clauses(text: str) -> bool:
    """
    Heuristic: intro Điều áp dụng cho các khoản nếu:
      - Kết thúc bằng ':' HOẶC
      - Chứa các cụm phổ biến ('sau đây:', 'bao gồm:', 'quy định như sau:'...)
    """
    if not text:
        return False
    t = text.strip()
    if t.endswith(':'):
        return True
    if INTRO_CUE_PAT.search(t):
        return True
    return False

# ===== Flush helpers =====
def flush_article_intro(chunks, base, stats, article_no, article_title, article_intro_buf, chapter, citations, chapter_number):
    content = (article_intro_buf or "").strip()
    if not content:
        return
    cid = f"{base['law_id']}-D{article_no}"
    exact = f"Điều {article_no}"
    meta = {
        **base, "chapter": chapter,
        "article_no": article_no, "article_title": article_title,
        "exact_citation": exact,
        "chapter_number": chapter_number
    }
    title_line = f"Điều {article_no}. {article_title}".strip() if article_title else f"Điều {article_no}"
    chunks.append({"id": cid, "content": f"{title_line}\n{content}", "metadata": meta})
    stats["article_intro"] += 1
    citations.append(exact)

def flush_clause(chunks, base, stats, article_no, article_title, clause_no, content,
                 chapter, citations, clause_intro: Optional[str] = None, chapter_number=None):
    content = (content or "").strip()
    if not content:
        return
    cid = f"{base['law_id']}-D{article_no}-K{clause_no}"
    exact = f"Điều {article_no} khoản {clause_no}"
    meta = {
        **base, "chapter": chapter,
        "article_no": article_no, "article_title": article_title,
        "clause_no": clause_no, "exact_citation": exact,
        "chapter_number": chapter_number
    }
    art_hdr = build_article_header(article_no, article_title)

    # TIÊM intro Điều vào khoản (nếu có)
    if clause_intro:
        intro = clause_intro.rstrip().rstrip(':') + ':'
        full_content = f"{art_hdr} Khoản {clause_no}. {intro}\n{content}"
        meta["clause_intro"] = clause_intro
    else:
        full_content = f"{art_hdr} Khoản {clause_no}. {content}"

    chunks.append({"id": cid, "content": full_content, "metadata": meta})
    stats["clauses"] += 1
    citations.append(exact)

def flush_point(chunks, base, stats, article_no, article_title, clause_no, letter, content, chapter, citations, clause_intro: Optional[str] = None, chapter_number=None):
    content = (content or "").strip()
    if not content:
        return
    if clause_intro:
        intro = clause_intro.rstrip().rstrip(':') + ':'
        content = f"{intro}\n{content}"
    letter = letter.lower()
    cid = f"{base['law_id']}-D{article_no}-K{clause_no}-{letter}"
    exact = f"Điều {article_no} khoản {clause_no} điểm {letter}."
    meta = {
        **base, "chapter": chapter,
        "article_no": article_no, "article_title": article_title,
        "clause_no": clause_no, "point_letter": letter,
        "exact_citation": exact,
        "chapter_number": chapter_number
    }
    if clause_intro:
        meta["clause_intro"] = clause_intro
    art_hdr = build_article_header(article_no, article_title)

    if clause_intro:
        intro = clause_intro.rstrip().rstrip(':')
        full_content = f"{art_hdr} Khoản {clause_no}. {intro}, điểm {letter}.\n{content}"
    else:
        full_content = f"{art_hdr} Khoản {clause_no}, điểm {letter}. {content}"

    chunks.append({"id": cid, "content": full_content, "metadata": meta})
    stats["points"] += 1
    citations.append(exact)

# ===== PASS 2: Chunk strict =====
def chunk_strict(lines: List[str], base: Dict, chapters_set: set, articles_set: set):
    chunks, citations = [], []
    stats = {"articles": 0, "article_intro": 0, "clauses": 0, "points": 0}

    chapters_seen_labels, warnings = [], []
    halted_reason: Optional[str] = None

    chapter_label: Optional[str] = None
    expecting_chapter_title: bool = False
    roman_current: Optional[str] = None
    chapter_number: Optional[int] = None

    article_no: Optional[int] = None
    article_title: str = ""
    expecting_article_title: bool = False

    article_intro_buf: str = ""
    article_has_any_chunk: bool = False

    clause_no: Optional[int] = None
    clause_buf: str = ""
    # intro của khoản → tiêm vào mọi điểm của khoản
    clause_intro_current: Optional[str] = None
    # intro của điều → tiêm vào mọi khoản của điều
    article_clause_intro_current: Optional[str] = None

    in_points: bool = False
    point_letter: Optional[str] = None
    point_buf: str = ""

    expected_chapter: Optional[int] = None
    expected_article: Optional[int] = None
    seeking_article: bool = False

    def close_clause():
        nonlocal clause_no, clause_buf, in_points, point_letter, point_buf, article_has_any_chunk, clause_intro_current
        if clause_no is None:
            return
        if in_points and point_letter:
            flush_point(chunks, base, stats, article_no, article_title, clause_no, point_letter,
                        point_buf, chapter_label, citations, clause_intro_current, chapter_number)
        elif clause_buf.strip():
            flush_clause(chunks, base, stats, article_no, article_title, clause_no, clause_buf,
                         chapter_label, citations, article_clause_intro_current, chapter_number)
        article_has_any_chunk = True
        clause_no, clause_buf, in_points, point_letter, point_buf = None, "", False, None, ""
        clause_intro_current = None

    def close_article_if_needed():
        nonlocal article_intro_buf, article_has_any_chunk
        if (not article_has_any_chunk) and article_intro_buf.strip():
            flush_article_intro(chunks, base, stats, article_no, article_title, article_intro_buf,
                                chapter_label, citations, chapter_number)
        article_intro_buf = ""
        article_has_any_chunk = False

    for ln_idx, line in enumerate(lines, start=1):
        if not line:
            continue

        if seeking_article:
            m_art_seek = ARTICLE_RE.match(line)
            if m_art_seek:
                a_no = int(m_art_seek.group(1))
                if a_no == expected_article:
                    seeking_article = False
                    close_clause()
                    if article_no is not None:
                        close_article_if_needed()
                    article_no = a_no
                    article_title = (m_art_seek.group(2) or "").strip()
                    stats["articles"] += 1
                    if not article_title:
                        expecting_article_title = True
                    expected_article = a_no + 1
                    clause_no = None
                    clause_buf = ""
                    in_points = False
                    point_letter = None
                    point_buf = ""
                    clause_intro_current = None
                    article_clause_intro_current = None
                    continue
                else:
                    continue
            m_ch_seek = CHAPTER_RE.match(line)
            if m_ch_seek:
                halted_reason = f"Thiếu Điều {expected_article} trước khi chuyển sang {line}"
                break
            continue

        if expecting_article_title:
            if not (CHAPTER_RE.match(line) or CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                article_title = line
                expecting_article_title = False
                continue
            else:
                expecting_article_title = False

        if expecting_chapter_title:
            if not (CHAPTER_RE.match(line) or CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                ch_title = line.strip()
                lbl = f"Chương {roman_current} – {ch_title}"
                chapter_label = lbl
                if lbl not in chapters_seen_labels:
                    chapters_seen_labels.append(lbl)
                base["chapter"] = chapter_label
                expecting_chapter_title = False
                continue
            else:
                expecting_chapter_title = False

        # CHƯƠNG
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            close_clause()
            if article_no is not None:
                close_article_if_needed()
            article_no = None
            article_title = ""
            article_intro_buf = ""
            expecting_article_title = False
            article_clause_intro_current = None

            roman = m_ch.group(1).strip()
            ch_num = roman_to_int(roman) or 0
            ch_title = (m_ch.group(2) or "").strip()
            lbl = f"Chương {roman}" + (f" – {ch_title}" if ch_title else "")

            if not ch_title:
                expecting_chapter_title = True
                roman_current = roman  # lưu lại để dùng sau
                chapter_number = ch_num  # lưu chapter_number
            else:
                chapter_label = lbl
                chapter_number = ch_num  # lưu chapter_number
                if lbl not in chapters_seen_labels:
                    chapters_seen_labels.append(lbl)
                base["chapter"] = chapter_label
                expecting_chapter_title = False

            if expected_chapter is None:
                expected_chapter = ch_num + 1
            else:
                if ch_num == expected_chapter:
                    expected_chapter = ch_num + 1
                elif ch_num > expected_chapter:
                    if expected_chapter not in chapters_set:
                        halted_reason = f"Thiếu Chương {expected_chapter} (không có trong tài liệu). Dừng tại {lbl}."
                        break
                    else:
                        warnings.append(f"Bỏ qua {lbl} vì đang chờ Chương {expected_chapter}.")
                        continue
                else:
                    warnings.append(f"Bỏ qua {lbl} (Chương lùi số).")
                    continue

            continue

        # ĐIỀU
        m_art = ARTICLE_RE.match(line)
        if m_art:
            a_no = int(m_art.group(1))
            a_title = (m_art.group(2) or "").strip()

            if expected_article is None:
                expected_article = a_no + 1
                close_clause()
                if article_no is not None:
                    close_article_if_needed()
                article_no = a_no
                article_title = a_title
                stats["articles"] += 1
                if not article_title:
                    expecting_article_title = True
                clause_no = None
                clause_buf = ""
                in_points = False
                point_letter = None
                point_buf = ""
                clause_intro_current = None
                article_clause_intro_current = None
                continue
            else:
                if a_no == expected_article:
                    expected_article = a_no + 1
                    close_clause()
                    if article_no is not None:
                        close_article_if_needed()
                    article_no = a_no
                    article_title = a_title
                    stats["articles"] += 1
                    if not article_title:
                        expecting_article_title = True
                    clause_no = None
                    clause_buf = ""
                    in_points = False
                    point_letter = None
                    point_buf = ""
                    clause_intro_current = None
                    article_clause_intro_current = None
                    continue
                elif a_no > expected_article:
                    if expected_article not in articles_set:
                        halted_reason = f"Thiếu Điều {expected_article} (không có trong tài liệu). Dừng tại Điều {a_no}."
                        break
                    else:
                        warnings.append(f"Bỏ qua Điều {a_no} vì đang chờ Điều {expected_article}.")
                        seeking_article = True
                        continue
                else:
                    warnings.append(f"Bỏ qua Điều {a_no} (Điều lùi số).")
                    continue

        if article_no is None:
            continue

        # KHOẢN
        m_k = CLAUSE_RE.match(line)
        if m_k and m_k.group(1).isdigit():
            # Nếu Điều có intro 'giới thiệu' → lưu lại để tiêm cho mọi khoản, KHÔNG flush chunk intro Điều
            if article_intro_buf.strip() and is_intro_text_for_clauses(article_intro_buf):
                article_clause_intro_current = article_intro_buf.strip()
                article_intro_buf = ""
            elif article_intro_buf.strip():
                # Intro không phải dạng 'giới thiệu' → tạo chunk Điều intro như cũ
                flush_article_intro(chunks, base, stats, article_no, article_title, article_intro_buf,
                                    chapter_label, citations)
                article_intro_buf = ""
                article_has_any_chunk = True

            close_clause()
            clause_no = int(m_k.group(1))
            clause_buf = (m_k.group(2) or "").strip()
            in_points = False
            point_letter = None
            point_buf = ""
            clause_intro_current = None
            continue

        # ĐIỂM — chuỗi điểm chỉ bắt đầu nếu mở bằng a)
        m_p = POINT_RE.match(line)
        if m_p and clause_no is not None:
            letter = m_p.group(1).lower()
            text = (m_p.group(2) or "").strip()

            if not in_points:
                if letter != 'a':
                    # không coi là điểm → nhập vào nội dung khoản
                    clause_buf += ("\n" if clause_buf else "") + f"{letter}. {text}"
                    continue
                # bắt đầu chuỗi điểm với 'a)' — KHÔNG flush chunk "Khoản X"
                # lưu intro khoản để tiêm vào MỌI điểm
                clause_intro_current = clause_buf.strip() if clause_buf.strip() else None
                clause_buf = ""
                in_points = True
                point_letter = letter
                point_buf = text
                continue

            # đang trong chuỗi điểm: flush điểm trước, mở điểm mới
            if point_letter:
                flush_point(chunks, base, stats, article_no, article_title, clause_no, point_letter,
                            point_buf, chapter_label, citations, clause_intro_current)
            in_points = True
            point_letter = letter
            point_buf = text
            continue

        # Nội dung kéo dài
        if clause_no is not None:
            if in_points and point_letter:
                point_buf += ("\n" if point_buf else "") + line
            else:
                clause_buf += ("\n" if clause_buf else "") + line
        else:
            # intro Điều (trước khi có khoản đầu tiên)
            article_intro_buf += ("\n" if article_intro_buf else "") + line

    # Kết thúc file
    close_clause()
    if article_no is not None:
        close_article_if_needed()

    summary = {
        "chapters_seen": chapters_seen_labels,
        "articles": stats["articles"],
        "article_intro": stats["article_intro"],
        "clauses": stats["clauses"],
        "points": stats["points"],
        "citations": citations,
        "warnings": warnings,
        "halted_reason": halted_reason,
        "total_chunks": len(chunks)
    }
    return chunks, summary

# ===================== GEMINI EVALUATION AGENT =====================

GEMINI_PROMPT = """Bạn là chuyên gia pháp điển hoá & kiểm thử dữ liệu luật.
Tôi gửi cho bạn:
1) Summary thống kê & cảnh báo từ bộ chunking.
2) Một số "excerpts" (trích đoạn nguyên văn) đại diện.
3) Danh mục chunks (id + metadata + content rút gọn).

Nhiệm vụ:
- PHÁT HIỆN BẤT THƯỜNG (anomalies) trong chunking, ví dụ:
  * Sai thứ tự/chưa “strict” (nhảy cóc Chương/Điều/Khoản/Điểm).
  * Nhận diện nhầm “Khoản” (không phải dạng `1.`) hoặc “Điểm” (không phải `a)`).
  * Không tiêm intro khoản vào điểm khi đã có chuỗi điểm.
  * Thiếu/bỏ sót nội dung so với excerpts.
  * Metadata không khớp: article_no/clause_no/point_letter/exact_citation.
  * Đóng mở chuỗi điểm sai (bắt đầu không phải “a)”, chèn nội dung thường vào giữa).
  * Nội dung “Điều” không có khoản nhưng không sinh chunk intro.
- GỢI Ý SỬA: chỉ rõ vị trí (id / exact_citation), mô tả vấn đề, cách khắc phục.
- Nếu KHÔNG thấy vấn đề, xác nhận “ok” và nêu ngắn gọn cơ sở kết luận.

Hãy TRẢ LỜI **CHỈ** ở dạng JSON theo schema:
{
  "status": "ok" | "issues_found",
  "confidence": 0.0-1.0,
  "issues": [
    {
      "id": "chuỗi id chunk hoặc mô tả vị trí",
      "citation": "Điều ... khoản ... điểm ...",
      "severity": "low|medium|high",
      "category": "ordering|regex|metadata|omission|points_chain|format|other",
      "message": "Mô tả ngắn gọn vấn đề",
      "suggestion": "Cách sửa ngắn gọn"
    }
  ],
  "notes": "Ghi chú ngắn (tuỳ chọn)"
}
Trả JSON hợp lệ. Không giải thích ngoài JSON.
"""

def _shorten_text(s: str, max_len: int = 500) -> str:
    if len(s) <= max_len:
        return s
    head = s[: max_len // 2].rstrip()
    tail = s[- max_len // 2 :].lstrip()
    return head + "\n...\n" + tail

def build_review_payload(chunks: List[Dict], summary: Dict, raw_text: str, sample_excerpts_chars: int = 8000):
    n = len(raw_text)
    if n <= sample_excerpts_chars:
        excerpts = raw_text
    else:
        k = sample_excerpts_chars // 3
        excerpts = raw_text[:k] + "\n...\n" + raw_text[n//2 - k//2 : n//2 + k//2] + "\n...\n" + raw_text[-k:]

    def lite(c):
        return {
            "id": c.get("id"),
            "metadata": c.get("metadata"),
            "content_preview": _shorten_text(c.get("content",""), 700)
        }

    chunks_lite = [lite(c) for c in chunks]
    return {
        "summary": summary,
        "excerpts": excerpts,
        "chunks_preview": chunks_lite[:1200]
    }

def call_gemini_review(payload: Dict, api_key: Optional[str] = None) -> Dict:
    """
    Gọi Gemini (model cố định GEMINI_MODEL_NAME) và ép trả JSON.
    API key lấy từ .env (GEMINI_API_KEY).
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY trong môi trường (.env).")

    # import tại chỗ, chỉ khi --AI bật
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(GEMINI_MODEL_NAME, generation_config=generation_config)
    prompt_parts = [
        {"role": "user", "parts": [{"text": GEMINI_PROMPT}]},
        {"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]},
    ]
    resp = model.generate_content(prompt_parts)
    raw = getattr(resp, "text", None) or (
        resp.candidates and resp.candidates[0].content.parts[0].text
    )
    if not raw:
        raise RuntimeError("Gemini không trả ra nội dung.")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON không phải object")
        data.setdefault("status", "issues_found")
        data.setdefault("issues", [])
        data.setdefault("confidence", 0.0)
        return data
    except Exception as e:
        return {
            "status": "issues_found",
            "confidence": 0.0,
            "issues": [{
                "id": "PARSER",
                "citation": "",
                "severity": "high",
                "category": "other",
                "message": f"Không parse được JSON từ Gemini: {e}",
                "suggestion": "Chạy lại hoặc giảm excerpts."
            }],
            "notes": raw[:2000]
        }

# ===================== CLI MAIN =====================

def main():
    load_dotenv()  # nạp .env (lấy GEMINI_API_KEY)

    ap = argparse.ArgumentParser(
        description="Chunk luật (strict) + (tuỳ chọn) gọi Gemini để thẩm định (--AI)."
    )
    ap.add_argument("--input", required=True, help="Đường dẫn .docx hoặc .txt (UTF-8)")
    ap.add_argument("--output", required=True, help="Đường dẫn file .json để xuất chunks (khi OK)")
    ap.add_argument("--law-no", default="52/2014/QH13")
    ap.add_argument("--law-title", default="Luật Hôn nhân và Gia đình")
    ap.add_argument("--law-id", default="LAW")
    ap.add_argument("--issued-date", help="Ngày ban hành luật (YYYY-MM-DD)")
    ap.add_argument("--effective-date", help="Ngày có hiệu lực (YYYY-MM-DD)")
    ap.add_argument("--signer", help="Người ký luật")

    # Chỉ bật AI khi có cờ --AI
    ap.add_argument("--AI", action="store_true", help="BẬT gọi Gemini (mặc định KHÔNG gọi).")
    ap.add_argument("--sample-excerpts", type=int, default=8000, help="Ký tự excerpts gửi Gemini (khi --AI).")
    ap.add_argument("--strict-ok-only", action="store_true", help="Chỉ ghi chunks nếu Gemini trả 'ok' (chỉ khi --AI).")

    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"❌ Không tìm thấy file: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Đọc toàn văn
    try:
        raw_text = read_text(in_path)
        # Thay thế "Luật này" (với L viết hoa) bằng law-title
        raw_text = re.sub(r'\bLuật này\b', args.law_title, raw_text)
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}", file=sys.stderr)
        sys.exit(1)
    lines = normalize_lines(raw_text)

    # PASS 1
    chapters_nums, articles_nums, chapters_labels, _ = prescan(lines)
    chapters_set, articles_set = set(chapters_nums), set(articles_nums)
    print("===== PRE-SCAN =====")
    print(f"Tổng CHƯƠNG (đầu dòng): {len(chapters_nums)}")
    print(f"Tổng ĐIỀU   (đầu dòng): {len(articles_nums)}")

    # PASS 2 (chunk strict)
    base = {
        "law_no": args.law_no,
        "law_title": args.law_title,
        "law_id": args.law_id,
        "issued_date": args.issued_date,
        "effective_date": args.effective_date,
        "expiry_date": None,  # mặc định null
        "signer": args.signer
    }
    chunks, summary = chunk_strict(lines, base, chapters_set, articles_set)

    # In tóm tắt
    print("\n===== TÓM TẮT CHUNKING =====")
    print(f"Số CHƯƠNG gặp (strict): {len(summary['chapters_seen'])}")
    print(f"Số ĐIỀU:               {summary['articles']}")
    print(f"Số ĐIỀU (intro):       {summary['article_intro']}")
    print(f"Số KHOẢN:              {summary['clauses']}")
    print(f"Số ĐIỂM:               {summary['points']}")
    print(f"Tổng chunk:            {summary['total_chunks']}")
    if summary["warnings"]:
        print(f"- Cảnh báo: {len(summary['warnings'])}")
        for i, warning in enumerate(summary["warnings"], 1):
            print(f"  {i}. {warning}")
    if summary["halted_reason"]:
        print(f"- Dừng strict tại: {summary['halted_reason']}")

    # Nếu KHÔNG bật --AI: ghi luôn chunks và kết thúc
    if not args.AI:
        out_path = pathlib.Path(args.output)
        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ (No-AI) Đã ghi {len(chunks)} chunks vào: {out_path}\n")
        sys.exit(0)

    # ===== Khi --AI bật: gọi Gemini để thẩm định =====
    payload = build_review_payload(
        chunks=chunks,
        summary=summary,
        raw_text=raw_text,
        sample_excerpts_chars=max(2000, args.sample_excerpts)
    )

    print("\n===== GỌI GEMINI (gemini-2.5-flash) ĐÁNH GIÁ =====")
    try:
        review = call_gemini_review(payload)
    except Exception as e:
        print(f"❌ Lỗi gọi Gemini: {e}", file=sys.stderr)
        if args.strict_ok_only:
            sys.exit(2)
        else:
            out_path = pathlib.Path(args.output)
            out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"⚠️ Không thẩm định được. Vẫn ghi chunks vào: {out_path}")
            sys.exit(0)

    status = review.get("status", "issues_found")
    confidence = review.get("confidence", 0.0)
    issues = review.get("issues", []) or []
    notes = review.get("notes", "")

    print(f"- Trạng thái: {status} | Confidence: {confidence:.2f}")
    if notes:
        print(f"- Ghi chú: {notes[:4000]}")

    if issues:
        print("\n===== BÁO CÁO VẤN ĐỀ =====")
        for i, it in enumerate(issues, 1):
            print(f"{i:02d}. [{it.get('severity','?')}] ({it.get('category','other')}) "
                  f"{it.get('citation') or it.get('id') or ''}")
            print(f"    - {it.get('message','(no message)')}")
            if it.get('suggestion'):
                print(f"    → Gợi ý: {it['suggestion']}")
        issues_path = pathlib.Path(args.output).with_suffix(".issues.json")
        issues_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n⚠️ Đã ghi báo cáo vấn đề: {issues_path}")

        if args.strict_ok_only:
            print("❌ --strict-ok-only: Không ghi chunks vì Gemini chưa xác nhận 'ok'.")
            sys.exit(3)

    # Ghi chunks (khi ok hoặc không bật strict-ok-only)
    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Đã ghi {len(chunks)} chunks vào: {out_path}")
    print("\n✓ Hoàn tất.\n")

if __name__ == "__main__":
    main()
