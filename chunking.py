#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chunking.py
-----------
Pass 1 (pre-scan): Đếm CHƯƠNG & ĐIỀU ở đầu dòng.
Pass 2 (strict): Chương N -> chỉ chấp nhận Chương N+1; Điều k -> chỉ chấp nhận Điều k+1.
- Nếu thiếu N+1/k+1 trong pre-scan => dừng, báo rõ.
- Nếu gặp số nhảy cóc và N+1/k+1 có trong file => bỏ qua cho đến khi gặp đúng số.

Micro-chunk:
- Điều (intro nếu không có khoản) -> Khoản -> Điểm
- Khoản: PHẢI dạng "1." (số + dấu chấm)
- Điểm: PHẢI dạng "a)" (có dấu )). Chỉ bắt chuỗi điểm nếu mở đầu là a).
- Nếu khoản có điểm: KHÔNG sinh chunk "Khoản X"; intro khoản được lưu và
  được TIÊM vào content của mọi điểm (a/b/c...), ví dụ:
  "Khoản 2, điểm b) Cấm các hành vi sau đây:\nTảo hôn, …"

Chạy:
python chunking.py --input "luat_hon_nhan_va_gia_dinh.docx" --output "hn2014_chunks.json" --law-no "52/2014/QH13" --law-title "Luật Hôn nhân và Gia đình" --law-id "HN2014"
"""

import argparse
import json
import pathlib
import re
import sys
from typing import List, Dict, Optional, Tuple

# ===== Regex (đầu dòng) =====
ARTICLE_RE = re.compile(r'^Điều\s+(\d+)\s*[\.:]?\s*(.*)$', re.UNICODE)
CHAPTER_RE = re.compile(r'^Chương\s+([IVXLCDM]+)\s*(.*)$', re.UNICODE|re.IGNORECASE)
SECTION_RE = re.compile(r'^Mục\s+(\d+)\s*[:\-]?\s*(.*)$', re.UNICODE|re.IGNORECASE)
# Khoản: PHẢI "1." (số + dấu chấm)
CLAUSE_RE  = re.compile(r'^\s*(\d+)\.\s*(.*)$', re.UNICODE)
# Điểm: PHẢI "a)" (bắt buộc có ')')
POINT_RE   = re.compile(r'^\s*([a-zA-ZđĐ])\)\s+(.*)$', re.UNICODE)

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
    return [re.sub(r'\s+$', '', ln) for ln in text.splitlines()]

# ===== Roman → int =====
ROMAN_MAP = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
def roman_to_int(s: str) -> Optional[int]:
    s = s.upper().strip()
    if not s or any(ch not in ROMAN_MAP for ch in s):
        return None
    total = 0; prev = 0
    for ch in reversed(s):
        val = ROMAN_MAP[ch]
        if val < prev: total -= val
        else: total += val; prev = val
    return total

# ===== PASS 1: Pre-scan CHƯƠNG/ĐIỀU (đầu dòng) =====
def prescan(lines: List[str]) -> Tuple[List[int], List[int], List[str], List[str]]:
    chapters_nums, articles_nums = [], []
    chapters_labels, article_titles_seen = [], []
    for raw in lines:
        line = raw
        if not line: continue
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            n = roman_to_int(m_ch.group(1))
            if n:
                chapters_nums.append(n)
                title = (m_ch.group(2) or "").strip()
                chapters_labels.append(f"Chương {m_ch.group(1).strip()}" + (f" – {title}" if title else ""))
            continue
        m_art = ARTICLE_RE.match(line)
        if m_art:
            num = int(m_art.group(1)); articles_nums.append(num)
            article_titles_seen.append((m_art.group(2) or "").strip())
            continue
    return chapters_nums, articles_nums, chapters_labels, article_titles_seen

# ===== Flush helpers =====
def flush_article_intro(chunks, base, stats, article_no, article_title, article_intro_buf, chapter, section, citations):
    content = (article_intro_buf or "").strip()
    if not content: return
    cid = f"{base['law_id']}-D{article_no}"
    exact = f"Điều {article_no}"
    meta = {**base, "chapter": chapter, "section": section,
            "article_no": article_no, "article_title": article_title,
            "exact_citation": exact}
    title_line = f"Điều {article_no}. {article_title}".strip() if article_title else f"Điều {article_no}"
    chunks.append({"id": cid, "content": f"{title_line}\n{content}", "metadata": meta})
    stats["article_intro"] += 1
    citations.append(exact)

def flush_clause(chunks, base, stats, article_no, article_title, clause_no, content, chapter, section, citations):
    content = (content or "").strip()
    if not content: return
    cid = f"{base['law_id']}-D{article_no}-K{clause_no}"
    exact = f"Điều {article_no} khoản {clause_no}"
    meta = {**base, "chapter": chapter, "section": section,
            "article_no": article_no, "article_title": article_title,
            "clause_no": clause_no, "exact_citation": exact}
    chunks.append({"id": cid, "content": f"Khoản {clause_no}. {content}", "metadata": meta})
    stats["clauses"] += 1
    citations.append(exact)

def flush_point(chunks, base, stats, article_no, article_title, clause_no, letter, content, chapter, section, citations, clause_intro: Optional[str] = None):
    content = (content or "").strip()
    if not content: return
    # TIÊM intro khoản vào đầu nội dung điểm (nếu có)
    if clause_intro:
        intro = clause_intro.rstrip().rstrip(':') + ':'
        content = f"{intro}\n{content}"
    letter = letter.lower()
    cid = f"{base['law_id']}-D{article_no}-K{clause_no}-{letter}"
    exact = f"Điều {article_no} khoản {clause_no} điểm {letter})"
    meta = {**base, "chapter": chapter, "section": section,
            "article_no": article_no, "article_title": article_title,
            "clause_no": clause_no, "point_letter": letter,
            "exact_citation": exact}
    # lưu cả clause_intro vào metadata để tra cứu/hiển thị
    if clause_intro:
        meta["clause_intro"] = clause_intro
    chunks.append({"id": cid, "content": f"Khoản {clause_no}, điểm {letter}) {content}", "metadata": meta})
    stats["points"] += 1
    citations.append(exact)

# ===== PASS 2: Chunk strict =====
def chunk_strict(lines: List[str], base: Dict, chapters_set: set, articles_set: set):
    chunks, citations = [], []
    stats = {"articles": 0, "article_intro": 0, "clauses": 0, "points": 0}

    chapters_seen_labels, warnings = [], []
    halted_reason: Optional[str] = None

    chapter_label: Optional[str] = None
    section_label: Optional[str] = None

    article_no: Optional[int] = None
    article_title: str = ""
    expecting_article_title: bool = False

    article_intro_buf: str = ""
    article_has_any_chunk: bool = False

    clause_no: Optional[int] = None
    clause_buf: str = ""
    # intro của khoản sẽ được tiêm vào mọi điểm
    clause_intro_current: Optional[str] = None
    in_points: bool = False
    point_letter: Optional[str] = None
    point_buf: str = ""

    expected_chapter: Optional[int] = None
    expected_article: Optional[int] = None
    seeking_article: bool = False

    def close_clause():
        nonlocal clause_no, clause_buf, in_points, point_letter, point_buf, article_has_any_chunk, clause_intro_current
        if clause_no is None: return
        if in_points and point_letter:
            flush_point(chunks, base, stats, article_no, article_title, clause_no, point_letter,
                        point_buf, chapter_label, section_label, citations, clause_intro_current)
        elif clause_buf.strip():
            flush_clause(chunks, base, stats, article_no, article_title, clause_no, clause_buf,
                         chapter_label, section_label, citations)
        article_has_any_chunk = True
        clause_no, clause_buf, in_points, point_letter, point_buf = None, "", False, None, ""
        clause_intro_current = None

    def close_article_if_needed():
        nonlocal article_intro_buf, article_has_any_chunk
        if (not article_has_any_chunk) and article_intro_buf.strip():
            flush_article_intro(chunks, base, stats, article_no, article_title, article_intro_buf,
                                chapter_label, section_label, citations)
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
                    clause_no = None; clause_buf = ""; in_points = False; point_letter = None; point_buf = ""
                    clause_intro_current = None
                    continue
                else:
                    continue
            m_ch_seek = CHAPTER_RE.match(line)
            if m_ch_seek:
                halted_reason = f"Thiếu Điều {expected_article} trước khi chuyển sang {line}"
                break
            continue

        # bổ sung tiêu đề điều ở dòng sau nếu cần
        if expecting_article_title:
            if not (CHAPTER_RE.match(line) or SECTION_RE.match(line) or CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                article_title = line; expecting_article_title = False; continue
            else:
                expecting_article_title = False

        # CHƯƠNG
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            # ĐÓNG hoàn toàn điều trước khi sang chương mới
            close_clause()
            if article_no is not None:
                close_article_if_needed()
            article_no = None
            article_title = ""
            article_intro_buf = ""
            expecting_article_title = False

            roman = m_ch.group(1).strip()
            ch_num = roman_to_int(roman) or 0
            ch_title = (m_ch.group(2) or "").strip()
            lbl = f"Chương {roman}" + (f" – {ch_title}" if ch_title else "")

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
                        warnings.append(f"Bỏ qua {lbl} vì đang chờ Chương {expected_chapter}."); continue
                else:
                    warnings.append(f"Bỏ qua {lbl} (Chương lùi số)."); continue

            chapter_label = lbl
            if lbl not in chapters_seen_labels: chapters_seen_labels.append(lbl)
            base["chapter"] = chapter_label
            section_label = None
            continue

        # MỤC
        m_sec = SECTION_RE.match(line)
        if m_sec:
            # Ngắt điều tương tự khi sang mục mới
            close_clause()
            if article_no is not None:
                close_article_if_needed()
            article_no = None
            article_title = ""
            article_intro_buf = ""
            expecting_article_title = False

            sec_no = m_sec.group(1).strip()
            sec_title = (m_sec.group(2) or "").strip()
            section_label = f"Mục {sec_no}" + (f" – {sec_title}" if sec_title else "")
            base["section"] = section_label
            continue

        # ĐIỀU
        m_art = ARTICLE_RE.match(line)
        if m_art:
            a_no = int(m_art.group(1))
            a_title = (m_art.group(2) or "").strip()

            if expected_article is None:
                expected_article = a_no + 1
                close_clause()
                if article_no is not None: close_article_if_needed()
                article_no = a_no; article_title = a_title; stats["articles"] += 1
                if not article_title: expecting_article_title = True
                clause_no = None; clause_buf = ""; in_points = False; point_letter = None; point_buf = ""
                clause_intro_current = None
                continue
            else:
                if a_no == expected_article:
                    expected_article = a_no + 1
                    close_clause()
                    if article_no is not None: close_article_if_needed()
                    article_no = a_no; article_title = a_title; stats["articles"] += 1
                    if not article_title: expecting_article_title = True
                    clause_no = None; clause_buf = ""; in_points = False; point_letter = None; point_buf = ""
                    clause_intro_current = None
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

        # nếu chưa vào điều nào thì bỏ qua
        if article_no is None:
            continue

        # KHOẢN — PHẢI "1." (số + dấu chấm)
        m_k = CLAUSE_RE.match(line)
        if m_k and m_k.group(1).isdigit():
            # Nếu đang có intro điều và chuẩn bị vào khoản đầu tiên → flush intro như "Điều X"
            if article_intro_buf.strip():
                flush_article_intro(chunks, base, stats, article_no, article_title, article_intro_buf,
                                    chapter_label, section_label, citations)
                article_intro_buf = ""; article_has_any_chunk = True
            close_clause()
            clause_no = int(m_k.group(1))
            clause_buf = (m_k.group(2) or "").strip()
            in_points = False; point_letter = None; point_buf = ""
            clause_intro_current = None
            continue

        # ĐIỂM — chỉ bắt đầu chuỗi điểm nếu mở đầu là a)
        m_p = POINT_RE.match(line)
        if m_p and clause_no is not None:
            letter = m_p.group(1).lower()
            text = (m_p.group(2) or "").strip()

            if not in_points:
                if letter != 'a':
                    # không coi là điểm -> gộp vào nội dung khoản
                    clause_buf += ("\n" if clause_buf else "") + f"{letter}) {text}"
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
                            point_buf, chapter_label, section_label, citations, clause_intro_current)
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
            # intro điều (chỉ tồn tại trước khi có khoản đầu tiên)
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

def main():
    ap = argparse.ArgumentParser(description="Chunk luật (strict Chương/Điều; Khoản '1.'; Điểm 'a)'; tiêm intro khoản vào mọi điểm)")
    ap.add_argument("--input", required=True, help="Đường dẫn .docx hoặc .txt (UTF-8)")
    ap.add_argument("--output", required=True, help="Đường dẫn file .json để xuất chunks")
    ap.add_argument("--law-no", default="52/2014/QH13")
    ap.add_argument("--law-title", default="Luật Hôn nhân và Gia đình")
    ap.add_argument("--law-id", default="LAW")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"❌ Không tìm thấy file: {in_path}", file=sys.stderr); sys.exit(1)

    try:
        text = read_text(in_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}", file=sys.stderr); sys.exit(1)

    lines = normalize_lines(text)

    # PASS 1
    chapters_nums, articles_nums, chapters_labels, _ = prescan(lines)
    chapters_set, articles_set = set(chapters_nums), set(articles_nums)
    print("===== PRE-SCAN =====")
    print(f"Tổng CHƯƠNG (đầu dòng): {len(chapters_nums)}  --> {', '.join(map(str, chapters_nums[:20]))}{' ...' if len(chapters_nums)>20 else ''}")
    print(f"Tổng ĐIỀU   (đầu dòng): {len(articles_nums)}  --> {', '.join(map(str, articles_nums[:20]))}{' ...' if len(articles_nums)>20 else ''}")

    # PASS 2
    base = {"law_no": args.law_no, "law_title": args.law_title, "law_id": args.law_id}
    chunks, summary = chunk_strict(lines, base, chapters_set, articles_set)

    # Xuất JSON
    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Đã ghi {len(chunks)} chunks vào: {out_path}")

    # TÓM TẮT
    print("\n===== TÓM TẮT =====")
    print(f"Số CHƯƠNG gặp (strict): {len(summary['chapters_seen'])}")
    for ch in summary["chapters_seen"]:
        print(f"  - {ch}")
    print(f"Số ĐIỀU:             {summary['articles']}")
    print(f"Số ĐIỀU (intro):     {summary['article_intro']}   # Điều không có Khoản → chunk 'Điều X'")
    print(f"Số KHOẢN:            {summary['clauses']}")
    print(f"Số ĐIỂM:             {summary['points']}")
    print(f"Tổng chunk:          {summary['total_chunks']}")

    # EXACT CITATIONS
    print("\n===== EXACT CITATIONS (theo thứ tự xuất hiện) =====")
    for i, cit in enumerate(summary["citations"], 1):
        print(f"{i:04d}. {cit}")

    # WARNINGS & HALT
    if summary["warnings"]:
        print("\n===== CẢNH BÁO =====")
        for w in summary["warnings"]:
            print("- " + w)
    if summary["halted_reason"]:
        print("\n===== DỪNG DO STRICT =====")
        print(summary["halted_reason"])

    print("\n✓ Hoàn tất.\n")

if __name__ == "__main__":
    main()
