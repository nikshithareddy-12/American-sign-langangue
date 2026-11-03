"""
create_docx.py
--------------
Script to convert PROJECT_DOCUMENTATION.md into a beautifully formatted Microsoft Word (.docx) document.
Generates: C:\\Users\\niksh\\OneDrive\\Desktop\\American sign lang\\PROJECT_DOCUMENTATION.docx
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

MD_PATH = os.path.join(os.path.dirname(__file__), "PROJECT_DOCUMENTATION.md")
DOCX_PATH = os.path.join(os.path.dirname(__file__), "PROJECT_DOCUMENTATION.docx")

def set_cell_background(cell, fill_hex):
    """Set cell background color in docx table."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner cell margins (padding)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_formatted_heading(doc, text, level):
    """Add styled headings."""
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    
    # Custom font colors for headings
    run = h.runs[0] if h.runs else h.add_run(text)
    if level == 1:
        run.font.size = Pt(18)
        run.font.color.rgb = RGBColor(26, 54, 93) # Navy Blue
        run.font.bold = True
    elif level == 2:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(43, 108, 176) # Medium Blue
        run.font.bold = True
    elif level == 3:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(45, 55, 72) # Slate Dark
        run.font.bold = True
    return h

def parse_markdown_to_docx():
    print(f"Reading {MD_PATH}...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = docx.Document()

    # Set document margins (1 inch all around)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Base Normal style setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(45, 55, 72)

    in_code_block = False
    code_block_lines = []
    
    in_table = False
    table_headers = []
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Code block handling (```)
        if line.strip().startswith("```"):
            if in_code_block:
                # Flush code block
                in_code_block = False
                code_text = "\n".join(code_block_lines)
                
                # Add table block for code formatting
                tbl = doc.add_table(rows=1, cols=1)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                cell = tbl.cell(0, 0)
                set_cell_background(cell, "F7FAFC") # Light gray-blue
                set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
                
                cp = cell.paragraphs[0]
                cp.paragraph_format.space_before = Pt(2)
                cp.paragraph_format.space_after = Pt(2)
                
                c_run = cp.add_run(code_text)
                c_run.font.name = "Consolas"
                c_run.font.size = Pt(9.5)
                c_run.font.color.rgb = RGBColor(45, 55, 72)
                
                # Empty paragraph after code box
                p_space = doc.add_paragraph()
                p_space.paragraph_format.space_before = Pt(0)
                p_space.paragraph_format.space_after = Pt(4)
                
                code_block_lines = []
            else:
                in_code_block = True
                code_block_lines = []
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Table handling (| ... |)
        if line.strip().startswith("|") and line.strip().endswith("|"):
            # Check if separator row (| :--- | ---: |)
            if re.match(r"^\|[\s:\-|\+]+\|$", line.strip()):
                i += 1
                continue
            
            row_cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            if not in_table:
                in_table = True
                table_headers = row_cells
                table_rows = []
            else:
                table_rows.append(row_cells)
            i += 1
            continue
        else:
            if in_table:
                # Flush table
                in_table = False
                num_cols = len(table_headers)
                tbl = doc.add_table(rows=len(table_rows) + 1, cols=num_cols)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                
                # Header row formatting
                hdr_cells = tbl.rows[0].cells
                for idx, h_text in enumerate(table_headers):
                    if idx < len(hdr_cells):
                        hdr_cells[idx].text = h_text
                        set_cell_background(hdr_cells[idx], "2B6CB0") # Medium Blue
                        set_cell_margins(hdr_cells[idx], top=100, bottom=100, left=120, right=120)
                        hp = hdr_cells[idx].paragraphs[0]
                        hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        for r in hp.runs:
                            r.font.bold = True
                            r.font.color.rgb = RGBColor(255, 255, 255)
                            r.font.size = Pt(10)
                
                # Body rows formatting
                for r_idx, r_data in enumerate(table_rows):
                    row_cells = tbl.rows[r_idx + 1].cells
                    bg_color = "EDF2F7" if r_idx % 2 == 1 else "FFFFFF"
                    for c_idx, val in enumerate(r_data):
                        if c_idx < len(row_cells):
                            row_cells[c_idx].text = val
                            set_cell_background(row_cells[c_idx], bg_color)
                            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
                            bp = row_cells[c_idx].paragraphs[0]
                            for r in bp.runs:
                                r.font.size = Pt(9.5)
                                r.font.color.rgb = RGBColor(45, 55, 72)

                p_space = doc.add_paragraph()
                p_space.paragraph_format.space_after = Pt(6)

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Horizontal rule (---)
        if line.strip() == "---":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            p_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="CBD5E0"/></w:pBdr>')
            p._p.get_or_add_pPr().append(p_border)
            i += 1
            continue

        # Document Title (# Title)
        if line.startswith("# "):
            title_text = line[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(title_text)
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = RGBColor(26, 54, 93)
            i += 1
            continue

        # Headings (##, ###, ####)
        if line.startswith("## "):
            add_formatted_heading(doc, line[3:].strip(), level=1)
            i += 1
            continue

        if line.startswith("### "):
            add_formatted_heading(doc, line[4:].strip(), level=2)
            i += 1
            continue

        if line.startswith("#### "):
            add_formatted_heading(doc, line[5:].strip(), level=3)
            i += 1
            continue

        # Bullet lists (- or *)
        if re.match(r"^[\s]*[\-\*]\s+", line):
            item_text = re.sub(r"^[\s]*[\-\*]\s+", "", line).strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            
            # Format bold inline Markdown (**bold**)
            parts = re.split(r"(\*\*.*?\*\*)", item_text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2])
                    r.font.bold = True
                else:
                    p.add_run(part)
            i += 1
            continue

        # Numbered lists (1., 2.)
        if re.match(r"^\d+\.\s+", line):
            item_text = re.sub(r"^\d+\.\s+", "", line).strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            
            parts = re.split(r"(\*\*.*?\*\*)", item_text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2])
                    r.font.bold = True
                else:
                    p.add_run(part)
            i += 1
            continue

        # Blockquote / Alert (> [!NOTE] or > text)
        if line.startswith(">"):
            quote_text = line.lstrip("> ").strip()
            tbl = doc.add_table(rows=1, cols=1)
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            cell = tbl.cell(0, 0)
            set_cell_background(cell, "EBF8FF") # Light blue tint
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            
            # Left border styling
            tcPr = cell._tc.get_or_add_tcPr()
            borders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'<w:left w:val="single" w:sz="24" w:space="0" w:color="3182CE"/>'
                f'<w:top w:val="none"/>'
                f'<w:right w:val="none"/>'
                f'<w:bottom w:val="none"/>'
                f'</w:tcBorders>'
            )
            tcPr.append(borders)
            
            qp = cell.paragraphs[0]
            qp.paragraph_format.space_before = Pt(2)
            qp.paragraph_format.space_after = Pt(2)
            q_run = qp.add_run(quote_text)
            q_run.font.italic = True
            q_run.font.color.rgb = RGBColor(43, 108, 176)
            
            p_space = doc.add_paragraph()
            p_space.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # Standard Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        
        parts = re.split(r"(\*\*.*?\*\*|\`.*?\`|\$.*?\$)", line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                r = p.add_run(part[2:-2])
                r.font.bold = True
            elif part.startswith("`") and part.endswith("`"):
                r = p.add_run(part[1:-1])
                r.font.name = "Consolas"
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(197, 48, 48) # Dark Red for inline code
            elif part.startswith("$") and part.endswith("$"):
                r = p.add_run(part[1:-1])
                r.font.italic = True
            else:
                p.add_run(part)

        i += 1

    doc.save(DOCX_PATH)
    print(f"[SUCCESS] Document generated successfully at: {DOCX_PATH}")

if __name__ == "__main__":
    parse_markdown_to_docx()
