"""투자설명서·간이투자설명서 마크다운을 조판된 PDF 로 변환한다.

외부 마크다운 라이브러리에 의존하지 않는다 — 입력 문서를 우리가 직접 쓰기 때문에
필요한 문법(제목·표·인용·목록·강조·수평선)만 처리하는 최소 변환기를 둔다.

파이프라인: Markdown → HTML(인쇄용 CSS) → Chrome headless → PDF → 페이지번호 스탬프

실행: python build_prospectus_pdf.py
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path

import fitz  # 페이지 번호 스탬프용

DOC = Path(r"C:/EST/indi/aq2_qrisproject/doc")
WORK = Path(r"C:/Users/User/AppData/Local/Temp/claude/C--EST-indi-aq2-qrisproject/c254a5b5-164d-4180-a794-0338d65ac3eb/scratchpad/prospectus")
CHROME = r"C:/Program Files/Google/Chrome/Application/chrome.exe"

TARGETS = [
    ("투자설명서_헬름IRP세븐서티액티브.md", "투자설명서_헬름IRP세븐서티액티브.pdf"),
    ("간이투자설명서_헬름IRP세븐서티액티브.md", "간이투자설명서_헬름IRP세븐서티액티브.pdf"),
]

# ── 인라인 변환 ──────────────────────────────────────────────────
def inline(s: str) -> str:
    """굵게·기울임·코드·이스케이프. 순서 주의(코드 먼저 보호)."""
    out, codes = [], []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes)-1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
    s = s.replace("&lt;br&gt;", "<br>")   # 표 헤더 줄바꿈만 허용(그 외 태그는 이스케이프 유지)
    for i, c in enumerate(codes):
        s = s.replace(f"\x00{i}\x00", f"<code>{html.escape(c)}</code>")
    return s


def render_table(rows: list[str]) -> str:
    """| a | b | 형태의 표. 두 번째 행이 구분선이면 첫 행을 헤더로 쓴다."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    has_head = len(cells) > 1 and all(set(c) <= set("-: ") and c for c in cells[1])
    body_start = 2 if has_head else 0
    out = ['<table>']
    # 헤더 칸이 전부 비어 있으면(| | | 형태) 빈 색띠만 남으므로 헤더를 그리지 않는다
    if has_head and any(c for c in cells[0]):
        out.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells[0]) + "</tr></thead>")
    out.append("<tbody>")
    for row in cells[body_start:]:
        # 첫 칸만 채워진 행은 소제목 행으로 처리(구분 강조)
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render_blocks(lines: list[str]) -> str:
    """블록 단위 렌더링. 인용 안에서도 제목·목록·표가 나올 수 있어 재귀 처리한다."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue

        # 수평선
        if re.fullmatch(r"-{3,}", s):
            out.append('<hr>')
            i += 1
            continue

        # 인용 블록 (연속 '>' 라인)
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner = render_blocks(buf)
            cls = "notice" if "⚠" in "".join(buf) else "quote"
            out.append(f'<div class="{cls}">{inner}</div>')
            continue

        # 표
        if s.startswith("|"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            out.append(render_table(buf))
            continue

        # 제목
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        # 목록 (연속된 -/숫자. 라인, 들여쓴 이어짐 포함)
        if re.match(r"^(-|\d+\.)\s+", s):
            ordered = bool(re.match(r"^\d+\.", s))
            items: list[str] = []
            while i < len(lines):
                t = lines[i]
                st = t.strip()
                if re.match(r"^(-|\d+\.)\s+", st):
                    items.append(re.sub(r"^(-|\d+\.)\s+", "", st))
                elif st and t.startswith(("  ", "\t")) and items:
                    items[-1] += " " + st           # 다음 줄 이어붙이기
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue

        # 문단 (빈 줄 전까지 합침)
        buf = []
        while i < len(lines):
            t = lines[i].strip()
            if not t or t.startswith(("|", ">", "#", "---")) or re.match(r"^(-|\d+\.)\s+", t):
                break
            buf.append(t)
            i += 1
        if buf:
            out.append(f"<p>{inline(' '.join(buf))}</p>")
    return "".join(out)


CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Malgun Gothic", "맑은 고딕", sans-serif;
  font-size: 9.6pt; line-height: 1.65; color: #1d232b; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 {
  font-size: 19pt; color: #12294a; margin: 0 0 14px; padding-bottom: 10px;
  border-bottom: 3px solid #12294a; line-height: 1.35; letter-spacing: -0.4px;
}
h1 + .notice { margin-top: 14px; }
h2 {
  font-size: 13.5pt; color: #12294a; margin: 26px 0 10px; padding: 7px 10px;
  background: #eef2f7; border-left: 4px solid #b98f34; letter-spacing: -0.3px;
}
/* 부(部) 제목은 새 페이지에서 시작 */
h1:not(:first-of-type) { page-break-before: always; }
h3 { font-size: 11pt; color: #1b3a63; margin: 18px 0 7px; letter-spacing: -0.2px; }
h4 { font-size: 10pt; color: #33445c; margin: 13px 0 5px; }
p { margin: 7px 0; }
ul, ol { margin: 7px 0 7px 0; padding-left: 19px; }
li { margin: 3px 0; }
strong { color: #0f2340; }
code {
  font-family: Consolas, monospace; font-size: 8.8pt;
  background: #eef1f5; padding: 1px 4px; border-radius: 3px;
}
hr { border: 0; border-top: 1px solid #d5dbe3; margin: 20px 0; }
table {
  width: 100%; border-collapse: collapse; margin: 10px 0 14px;
  font-size: 8.9pt; page-break-inside: avoid;
}
th {
  background: #12294a; color: #fff; font-weight: 700; text-align: left;
  padding: 6px 8px; border: 1px solid #12294a; letter-spacing: -0.2px;
}
/* 헤더 셀 안의 강조는 본문 색(진남색)을 물려받으면 배경에 묻힌다 */
th strong, th em, th code { color: #ffd479; background: none; }
td { padding: 5px 8px; border: 1px solid #ccd4de; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f6f8fb; }
/* 경고 박스 — 모의 문서 고지 */
.notice {
  background: #fdf6ec; border: 1.6px solid #d9a441; border-radius: 5px;
  padding: 12px 15px; margin: 14px 0; page-break-inside: avoid;
}
.notice h2 {
  background: none; border: 0; padding: 0; margin: 0 0 8px;
  color: #93610d; font-size: 12pt;
}
.notice p, .notice li { color: #4a3a1c; font-size: 9.2pt; }
.notice strong { color: #7a4a06; }
/* 일반 인용 — 주석·유의사항 */
.quote {
  background: #f5f7fa; border-left: 3px solid #9fb0c6; padding: 9px 13px;
  margin: 11px 0; font-size: 9.1pt; color: #3c4a5c; page-break-inside: avoid;
}
.quote p { margin: 4px 0; }
.quote table { margin: 6px 0; }
"""


def convert(md_path: Path, pdf_path: Path) -> None:
    md = md_path.read_text(encoding="utf-8")
    body = render_blocks(md.split("\n"))
    title = md_path.stem
    doc = (f"<!DOCTYPE html><html lang=ko><head><meta charset=utf-8>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body>{body}</body></html>")

    WORK.mkdir(parents=True, exist_ok=True)
    htm = WORK / (md_path.stem + ".html")
    htm.write_text(doc, encoding="utf-8")

    raw = WORK / (md_path.stem + "_raw.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", f"--print-to-pdf={raw}",
                    htm.as_uri()], check=True, capture_output=True, timeout=180)

    stamp(raw, pdf_path, title)
    print(f"· {pdf_path.name}  ({fitz.open(pdf_path).page_count}p)")


def stamp(src: Path, dst: Path, title: str) -> None:
    """Chrome 은 CSS 페이지번호를 지원하지 않으므로 생성 후 직접 새긴다."""
    d = fitz.open(src)
    n = d.page_count
    note = "모의 문서 · 투자 권유 불가"
    for i, page in enumerate(d):
        w, h = page.rect.width, page.rect.height
        page.draw_line(fitz.Point(42, h - 34), fitz.Point(w - 42, h - 34),
                       color=(0.84, 0.87, 0.9), width=0.6)
        # 한글은 base-14 폰트로 못 그린다 — 내장 CJK 폰트(korea)를 쓴다.
        page.insert_text(fitz.Point(42, h - 22), note, fontname="korea",
                         fontsize=7.2, color=(0.55, 0.58, 0.63))
        label = f"{i + 1} / {n}"
        page.insert_text(fitz.Point(w - 42 - fitz.get_text_length(label, "helv", 8), h - 22),
                         label, fontname="helv", fontsize=8, color=(0.35, 0.4, 0.47))
    d.save(dst, deflate=True)
    d.close()


if __name__ == "__main__":
    if not Path(CHROME).exists():
        sys.exit(f"Chrome 을 찾을 수 없습니다: {CHROME}")
    for md_name, pdf_name in TARGETS:
        convert(DOC / md_name, DOC / pdf_name)
    print("완료")
