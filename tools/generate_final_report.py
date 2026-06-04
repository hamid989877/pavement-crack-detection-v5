from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
OUTPUT_PDF = REPORT_DIR / "Pavement_Crack_Detection_Final_Project_Report.pdf"

ASPHALT = colors.HexColor("#333333")
ASPHALT_DARK = colors.HexColor("#252728")
ASPHALT_LIGHT = colors.HexColor("#53565A")
AMBER = colors.HexColor("#FFC107")
YELLOW = colors.HexColor("#FFD700")
RED = colors.HexColor("#DC3545")
LIGHT = colors.HexColor("#F0F0F0")
INK = colors.HexColor("#1D1F20")
BLUE_LINK = colors.HexColor("#1F5FBF")

PROJECT_URL = "https://hamid989877.github.io/pavement-crack-detection-v5/"
IMAGE_TEST_URL = "https://hamid989877.github.io/pavement-crack-detection-v5/image.html"
MODEL_REPO_URL = "https://huggingface.co/hamid989877/pavement-crack-detection-v5-model"


MODEL_ROWS = [
    ("version-1", "run-1", "yolo11m.pt", "RDD-1", 40.52, 0.8940, 0.5254, 0.8331, 0.8283),
    ("version-1", "run-2", "yolo26m.pt", "RDD-1", 44.03, 0.8813, 0.5123, 0.8201, 0.8337),
    ("version-1", "run-3", "yolo26x.pt", "RDD-5", 118.33, 0.8804, 0.5151, 0.8401, 0.8104),
    ("version-1", "run-4", "yolo11x.pt", "RDD-5", 114.40, 0.8821, 0.5157, 0.8215, 0.8311),
    ("version-1", "run-5", "yolo11n.pt", "RDD-5", 5.48, 0.8574, 0.4923, 0.7851, 0.8069),
    ("version-1", "run-6", "yolov8m.pt", "RDD-5", 52.04, 0.8844, 0.5164, 0.8294, 0.8111),
    ("version-2", "run-1", "yolo26m.pt", "RDD-2", 44.03, 0.8453, 0.4952, 0.8461, 0.8014),
    ("version-3", "run-1", "yolo26x.pt", "RDD-4", 118.33, 0.8695, 0.5075, 0.8317, 0.8286),
]


REFERENCES = [
    (
        "Nie, M. and Wang, C. Pavement Crack Detection based on yolo v3. "
        "2019 2nd International Conference on Safety Produce Informatization, "
        "doi:10.1109/IICSPI48186.2019.9095956."
    ),
    (
        "Zhang, S., Bei, Z., Ling, T., Chen, Q., and Zhang, L. Research on "
        "high-precision recognition model for multi-scene asphalt pavement "
        "distresses based on deep learning. Scientific Reports, 2024."
    ),
    (
        "Antariksa, G. et al. Comparative Analysis of Advanced AI-based Object "
        "Detection Models for Pavement Marking Quality Assessment during Daytime. 2025."
    ),
    (
        "Cubero-Fernandez, A., Rodriguez-Lozano, F. J., Villatoro, R., Olivares, J., "
        "and Palomares, J. M. Efficient pavement crack detection and classification. "
        "EURASIP Journal on Image and Video Processing, 2017, doi:10.1186/s13640-017-0187-0."
    ),
    "Ultralytics. YOLOv8 model documentation. https://docs.ultralytics.com/models/yolov8/",
    "Ultralytics. YOLO11 model documentation. https://docs.ultralytics.com/models/yolo11/",
]

NEW_PAGE_SECTIONS = {
    "Technical Summary",
    "Classical Crack Detection Experiments",
    "Modern YOLOv8 Literature Experiments",
    "Synthesis Of Reference Experiments",
    "YOLO Algorithm Family",
    "YOLOv8 Architecture Details",
    "Loss Functions And Optimization",
    "Bounding Boxes, IoU, And NMS",
    "Precision-Recall Evidence",
    "Confidence Threshold Behavior",
    "Confusion Matrix And Error Analysis",
    "Validation Prediction Examples",
    "More Visual Evidence From The Selected Run",
    "Cross-Version Training Observations",
    "Qualitative Comparison Across Versions",
    "From Detection Output To Maintenance Insight",
    "Reference Lessons Applied To This Project",
    "Future Technical Improvements",
    "References",
}


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=draw_page)])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "Heading1":
            text = flowable.getPlainText()
            key = f"section-{self.seq.nextf('section')}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (0, text, self.page, key))


def draw_page(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:
        width, height = A4
        canvas.setStrokeColor(AMBER)
        canvas.setLineWidth(1)
        canvas.line(doc.leftMargin, height - 0.42 * inch, width - doc.rightMargin, height - 0.42 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(ASPHALT_LIGHT)
        canvas.drawString(doc.leftMargin, 0.35 * inch, "Pavement Crack Detection - Final Project Report")
        canvas.drawRightString(width - doc.rightMargin, 0.35 * inch, f"Page {page}")
    canvas.restoreState()


def stylesheet():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=33,
            alignment=TA_CENTER,
            textColor=ASPHALT,
            spaceAfter=18,
        )
    )
    base.add(
        ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=19,
            alignment=TA_CENTER,
            textColor=ASPHALT_DARK,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            "TOCTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=ASPHALT,
            spaceAfter=10,
        )
    )
    base["Heading1"].fontName = "Helvetica-Bold"
    base["Heading1"].fontSize = 17
    base["Heading1"].leading = 21
    base["Heading1"].textColor = ASPHALT
    base["Heading1"].spaceBefore = 6
    base["Heading1"].spaceAfter = 8
    base["Heading2"].fontName = "Helvetica-Bold"
    base["Heading2"].fontSize = 12.5
    base["Heading2"].leading = 15
    base["Heading2"].textColor = ASPHALT_DARK
    base["Heading2"].spaceBefore = 7
    base["Heading2"].spaceAfter = 5
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 9.4
    base["Normal"].leading = 13.4
    base["Normal"].alignment = TA_JUSTIFY
    base["Normal"].spaceAfter = 6
    base.add(ParagraphStyle("Body", parent=base["Normal"], alignment=TA_JUSTIFY))
    base.add(
        ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontSize=7.7,
            leading=9.7,
            textColor=colors.HexColor("#444444"),
        )
    )
    base.add(
        ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=8.2,
            leading=10.2,
            alignment=TA_CENTER,
            textColor=ASPHALT_LIGHT,
            spaceBefore=3,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            "ProjectBullet",
            parent=base["Normal"],
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=0,
            alignment=TA_LEFT,
        )
    )
    base.add(
        ParagraphStyle(
            "Link",
            parent=base["Normal"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            textColor=BLUE_LINK,
            fontSize=10,
            leading=14,
        )
    )
    return base


STYLES = stylesheet()


def para(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def heading(text: str, level: int = 1) -> Paragraph:
    return Paragraph(text, STYLES["Heading1" if level == 1 else "Heading2"])


def bullet(text: str) -> Paragraph:
    return Paragraph(text, STYLES["ProjectBullet"], bulletText="-")


def image_flowable(path: Path, caption: str, max_width=6.5 * inch, max_height=3.65 * inch):
    if not path.exists():
        return KeepTogether([para(f"Missing figure: {path.name}", "Small")])
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    return KeepTogether(
        [
            Image(str(path), width=width * scale, height=height * scale),
            Paragraph(caption, STYLES["Caption"]),
        ]
    )


def styled_table(rows, col_widths=None, font_size=7.4, repeat_rows=1, highlight_row=None):
    processed = [[Paragraph(str(cell), STYLES["Small"]) for cell in row] for row in rows]
    table = Table(processed, colWidths=col_widths, repeatRows=repeat_rows)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), ASPHALT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8C8C8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_row is not None:
        commands.extend(
            [
                ("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#FFF4C2")),
                ("LINEBEFORE", (0, highlight_row), (0, highlight_row), 3, AMBER),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def section(story, title: str):
    if title in NEW_PAGE_SECTIONS and story:
        story.append(PageBreak())
    story.append(heading(title))


def detector_process_drawing():
    drawing = Drawing(470, 145)
    drawing.add(String(10, 130, "YOLO-style detection pipeline", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    boxes = [
        (10, 78, "Input image", "resize to 640"),
        (112, 78, "Backbone", "feature extraction"),
        (214, 78, "Neck", "multi-scale fusion"),
        (316, 78, "Head", "boxes and classes"),
        (214, 18, "Post-processing", "NMS and threshold"),
        (316, 18, "Detection result", "label, box, confidence"),
    ]
    for x, y, title, sub in boxes:
        drawing.add(Rect(x, y, 90, 42, fillColor=colors.HexColor("#F7F7F7"), strokeColor=ASPHALT_LIGHT))
        drawing.add(String(x + 7, y + 26, title, fontName="Helvetica-Bold", fontSize=8.2, fillColor=ASPHALT))
        drawing.add(String(x + 7, y + 12, sub, fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    for x1, y1, x2, y2 in [(100, 99, 112, 99), (202, 99, 214, 99), (304, 99, 316, 99), (361, 78, 361, 60), (259, 39, 316, 39)]:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=AMBER, strokeWidth=2))
    drawing.add(Rect(420, 78, 28, 42, fillColor=AMBER, strokeColor=AMBER))
    drawing.add(String(426, 96, "IoU", fontName="Helvetica-Bold", fontSize=7, fillColor=ASPHALT))
    drawing.add(String(421, 84, "score", fontName="Helvetica-Bold", fontSize=7, fillColor=ASPHALT))
    return drawing


def training_process_drawing():
    drawing = Drawing(470, 145)
    drawing.add(String(10, 130, "Project training and model-selection process", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    boxes = [
        (10, 78, "Images", "road distress samples"),
        (110, 78, "Labels", "four crack classes"),
        (210, 78, "Train", "50 epochs, 640 px"),
        (310, 78, "Validate", "mAP, precision, recall"),
        (210, 18, "Compare", "versions and runs"),
        (310, 18, "Select", "version-1/run-6"),
    ]
    for x, y, title, sub in boxes:
        drawing.add(Rect(x, y, 88, 42, fillColor=colors.HexColor("#F7F7F7"), strokeColor=ASPHALT_LIGHT))
        drawing.add(String(x + 7, y + 26, title, fontName="Helvetica-Bold", fontSize=8.2, fillColor=ASPHALT))
        drawing.add(String(x + 7, y + 12, sub, fontName="Helvetica", fontSize=6.8, fillColor=ASPHALT_LIGHT))
    for x1, y1, x2, y2 in [(98, 99, 110, 99), (198, 99, 210, 99), (298, 99, 310, 99), (254, 78, 254, 60), (254, 39, 310, 39)]:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=AMBER, strokeWidth=2))
    return drawing


def model_bar_chart():
    width = 470
    height = 170
    drawing = Drawing(width, height)
    drawing.add(String(10, 155, "Final validation mAP by training run", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    chart_x = 55
    chart_y = 30
    chart_w = 390
    chart_h = 110
    drawing.add(Line(chart_x, chart_y, chart_x + chart_w, chart_y, strokeColor=ASPHALT_LIGHT))
    drawing.add(Line(chart_x, chart_y, chart_x, chart_y + chart_h, strokeColor=ASPHALT_LIGHT))
    group_w = chart_w / len(MODEL_ROWS)
    for idx, row in enumerate(MODEL_ROWS):
        _, run, _, _, _, map50, map5095, _, _ = row
        x = chart_x + idx * group_w + 7
        h1 = map50 * chart_h
        h2 = map5095 * chart_h
        fill = AMBER if run == "run-6" and idx == 5 else colors.HexColor("#D4A800")
        drawing.add(Rect(x, chart_y, 8, h1, fillColor=fill, strokeColor=fill))
        drawing.add(Rect(x + 10, chart_y, 8, h2, fillColor=ASPHALT_LIGHT, strokeColor=ASPHALT_LIGHT))
        label = f"v{row[0].split('-')[-1]}-{run.replace('run-', 'r')}"
        drawing.add(String(x - 4, chart_y - 12, label, fontName="Helvetica", fontSize=6.2, fillColor=ASPHALT_LIGHT))
    drawing.add(String(12, chart_y + chart_h - 2, "1.0", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(14, chart_y + chart_h * 0.5 - 2, "0.5", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(Rect(300, 150, 8, 8, fillColor=AMBER, strokeColor=AMBER))
    drawing.add(String(313, 151, "mAP50", fontName="Helvetica", fontSize=7, fillColor=ASPHALT))
    drawing.add(Rect(360, 150, 8, 8, fillColor=ASPHALT_LIGHT, strokeColor=ASPHALT_LIGHT))
    drawing.add(String(373, 151, "mAP50-95", fontName="Helvetica", fontSize=7, fillColor=ASPHALT))
    return drawing


def yolo_v8_complexity_chart():
    data = [
        ("n", 3.2, 8.7),
        ("s", 11.2, 28.6),
        ("m", 25.9, 78.9),
        ("l", 43.7, 165.2),
        ("x", 68.2, 257.8),
    ]
    width = 470
    height = 180
    drawing = Drawing(width, height)
    drawing.add(String(10, 165, "YOLOv8 model complexity from Zhang et al. [2]", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    chart_x = 55
    chart_y = 32
    chart_w = 370
    chart_h = 120
    drawing.add(Line(chart_x, chart_y, chart_x + chart_w, chart_y, strokeColor=ASPHALT_LIGHT))
    drawing.add(Line(chart_x, chart_y, chart_x, chart_y + chart_h, strokeColor=ASPHALT_LIGHT))
    group_w = chart_w / len(data)
    max_params = max(row[1] for row in data)
    max_flops = max(row[2] for row in data)
    for idx, (name, params, flops) in enumerate(data):
        x = chart_x + idx * group_w + 16
        h1 = params / max_params * chart_h
        h2 = flops / max_flops * chart_h
        drawing.add(Rect(x, chart_y, 13, h1, fillColor=AMBER, strokeColor=AMBER))
        drawing.add(Rect(x + 18, chart_y, 13, h2, fillColor=ASPHALT_LIGHT, strokeColor=ASPHALT_LIGHT))
        drawing.add(String(x + 4, chart_y - 13, name, fontName="Helvetica-Bold", fontSize=7, fillColor=ASPHALT))
    drawing.add(Rect(285, 160, 8, 8, fillColor=AMBER, strokeColor=AMBER))
    drawing.add(String(298, 161, "Parameters", fontName="Helvetica", fontSize=7, fillColor=ASPHALT))
    drawing.add(Rect(365, 160, 8, 8, fillColor=ASPHALT_LIGHT, strokeColor=ASPHALT_LIGHT))
    drawing.add(String(378, 161, "FLOPs", fontName="Helvetica", fontSize=7, fillColor=ASPHALT))
    return drawing


def literature_result_chart():
    rows = [
        ("Classic crack detection [4]", 88.0, "Detection success"),
        ("YOLOv3 crack detection [1]", 88.0, "Accuracy"),
        ("SMG-YOLOv8 [2]", 79.4, "mAP50"),
        ("Selected project model", 88.44, "mAP50"),
    ]
    drawing = Drawing(470, 175)
    drawing.add(String(10, 160, "Reference and project result anchors", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    chart_x = 160
    chart_y = 30
    bar_h = 18
    max_value = 100
    for idx, (name, value, metric) in enumerate(rows):
        y = chart_y + (len(rows) - idx - 1) * 30
        drawing.add(String(10, y + 4, name, fontName="Helvetica", fontSize=7.2, fillColor=ASPHALT))
        drawing.add(Rect(chart_x, y, value / max_value * 280, bar_h, fillColor=AMBER if "project" in name.lower() else ASPHALT_LIGHT, strokeColor=None))
        drawing.add(String(chart_x + value / max_value * 280 + 5, y + 4, f"{value:.1f}% {metric}", fontName="Helvetica", fontSize=7, fillColor=ASPHALT))
    drawing.add(Line(chart_x, chart_y - 6, chart_x + 280, chart_y - 6, strokeColor=ASPHALT_LIGHT))
    drawing.add(String(chart_x, chart_y - 20, "0", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(chart_x + 135, chart_y - 20, "50", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(chart_x + 268, chart_y - 20, "100", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    return drawing


def model_tradeoff_chart():
    drawing = Drawing(470, 185)
    drawing.add(String(10, 170, "Project model trade-off: checkpoint size versus mAP50", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    chart_x = 55
    chart_y = 35
    chart_w = 370
    chart_h = 120
    drawing.add(Line(chart_x, chart_y, chart_x + chart_w, chart_y, strokeColor=ASPHALT_LIGHT))
    drawing.add(Line(chart_x, chart_y, chart_x, chart_y + chart_h, strokeColor=ASPHALT_LIGHT))
    min_size, max_size = 0, 125
    min_map, max_map = 0.84, 0.90
    for version, run, base_model, _data, size_mb, map50, _map5095, _precision, _recall in MODEL_ROWS:
        x = chart_x + (size_mb - min_size) / (max_size - min_size) * chart_w
        y = chart_y + (map50 - min_map) / (max_map - min_map) * chart_h
        fill = AMBER if version == "version-1" and run == "run-6" else ASPHALT_LIGHT
        drawing.add(Rect(x - 4, y - 4, 8, 8, fillColor=fill, strokeColor=ASPHALT))
        drawing.add(String(x + 5, y + 2, run.replace("run-", "r"), fontName="Helvetica", fontSize=6.2, fillColor=ASPHALT))
    drawing.add(String(chart_x, chart_y - 18, "0 MB", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(chart_x + chart_w - 24, chart_y - 18, "125 MB", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(12, chart_y + chart_h - 2, "0.90", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(12, chart_y - 1, "0.84", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    return drawing


def precision_recall_chart():
    drawing = Drawing(470, 185)
    drawing.add(String(10, 170, "Project model trade-off: precision versus recall", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    chart_x = 55
    chart_y = 35
    chart_w = 370
    chart_h = 120
    drawing.add(Line(chart_x, chart_y, chart_x + chart_w, chart_y, strokeColor=ASPHALT_LIGHT))
    drawing.add(Line(chart_x, chart_y, chart_x, chart_y + chart_h, strokeColor=ASPHALT_LIGHT))
    min_p, max_p = 0.76, 0.86
    min_r, max_r = 0.78, 0.85
    for version, run, _base_model, _data, _size_mb, _map50, _map5095, precision, recall in MODEL_ROWS:
        x = chart_x + (precision - min_p) / (max_p - min_p) * chart_w
        y = chart_y + (recall - min_r) / (max_r - min_r) * chart_h
        fill = AMBER if version == "version-1" and run == "run-6" else ASPHALT_LIGHT
        drawing.add(Rect(x - 4, y - 4, 8, 8, fillColor=fill, strokeColor=ASPHALT))
        drawing.add(String(x + 5, y + 2, run.replace("run-", "r"), fontName="Helvetica", fontSize=6.2, fillColor=ASPHALT))
    drawing.add(String(chart_x, chart_y - 18, "Precision 0.76", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(chart_x + chart_w - 56, chart_y - 18, "Precision 0.86", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(12, chart_y + chart_h - 2, "Recall 0.85", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(12, chart_y - 1, "0.78", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    return drawing


def image_grid(items, max_width=3.05 * inch, max_height=2.1 * inch):
    rows = []
    row = []
    for path, caption in items:
        if path.exists():
            with PILImage.open(path) as img:
                width, height = img.size
            scale = min(max_width / width, max_height / height)
            cell = [
                Image(str(path), width=width * scale, height=height * scale),
                Paragraph(caption, STYLES["Caption"]),
            ]
        else:
            cell = [Paragraph(f"Missing figure: {path.name}", STYLES["Small"])]
        row.append(cell)
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        row.append("")
        rows.append(row)
    table = Table(rows, colWidths=[3.2 * inch, 3.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def add_reference_list(story):
    for idx, ref in enumerate(REFERENCES, start=1):
        story.append(Paragraph(f"[{idx}] {ref}", STYLES["Small"]))
        story.append(Spacer(1, 3))


def build_story():
    story = []
    cover_img = ROOT / "static" / "assets" / "hero-crack-background.jpg"
    selected_dir = ROOT / "versions" / "version-1" / "run-6" / "detect" / "train"

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Pavement Crack Detection", STYLES["CoverTitle"]))
    story.append(Paragraph("Final Project Report", STYLES["CoverSubtitle"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Hamidreza Khalaj Zahraei", STYLES["CoverSubtitle"]))
    story.append(Paragraph("Student ID: 300204093", STYLES["CoverSubtitle"]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(image_flowable(cover_img, "Cover image: asphalt pavement crack used as the project visual theme.", max_width=6.4 * inch, max_height=3.0 * inch))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        para(
            "This report documents the technical process for a YOLO-based pavement crack detection project. "
            "It explains the detection problem, related algorithms, reference experiments, training versions, "
            "metric interpretation, and the reason version-1/run-6 was selected for the public demonstration."
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Table of Contents", STYLES["TOCTitle"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOCHeading1", fontName="Helvetica", fontSize=9.5, leftIndent=0, firstLineIndent=0, leading=12.2),
    ]
    story.append(toc)

    section(story, "Technical Summary")
    story.append(
        para(
            "The project goal is to detect and classify visible pavement distresses from road images using a YOLO object detector. "
            "The deployed class set is Alligator Crack, Longitudinal Crack, Pothole, and Transverse Crack. These classes were chosen "
            "because the literature repeatedly separates road-surface detection from road-surface classification: detecting a crack "
            "is useful, but knowing the crack type helps decide the maintenance response [4]."
        )
    )
    story.append(
        para(
            "The final selected model is version-1/run-6, trained from a YOLOv8m checkpoint. Its final validation metrics were "
            "mAP50 = 0.8844, mAP50-95 = 0.5164, precision = 0.8294, and recall = 0.8111. This model was chosen because it gave a "
            "strong accuracy and reliability balance without requiring the much larger X-size checkpoints."
        )
    )
    story.append(
        para(
            "The public project page is listed below. The image testing page is the main place to upload pavement images and inspect "
            "the predicted crack labels, confidence scores, and bounding boxes."
        )
    )
    story.append(Paragraph(f"Project URL: {PROJECT_URL}", STYLES["Link"]))
    story.append(Paragraph(f"Image testing URL: {IMAGE_TEST_URL}", STYLES["Link"]))
    story.append(Spacer(1, 8))
    story.append(
        para(
            "This revised report intentionally avoids website implementation details and instead focuses on the computer-vision method, "
            "training evidence, algorithm comparison, and lessons from the supplied research papers."
        )
    )

    section(story, "Road Distress Problem")
    story.append(
        para(
            "Pavement distress appears in many visual forms. A longitudinal crack runs mainly along the road direction, a transverse "
            "crack crosses the lane, an alligator crack forms a connected pattern caused by repeated stress, and a pothole is a local "
            "surface failure where material has been lost. From an image-analysis perspective, these defects are difficult because the "
            "foreground object is often thin, irregular, dark, and textured like the background."
        )
    )
    story.append(
        para(
            "The practical value of automated detection is not just faster labeling. It also helps make inspections more consistent. "
            "Manual inspection depends on human attention, camera viewpoint, lighting, road dirt, and subjective judgment. A trained "
            "detector can apply the same decision rules over a large number of frames, which is useful for road-network screening and "
            "maintenance prioritization."
        )
    )
    story.append(
        styled_table(
            [
                ["Distress class", "Visual pattern", "Why detection is difficult", "Maintenance relevance"],
                ["Alligator Crack", "Connected crack network resembling broken asphalt blocks.", "Many small segments can be confused with texture or shadows.", "Often indicates structural fatigue and may require stronger repair."],
                ["Longitudinal Crack", "Crack line following the driving direction.", "Can be thin, low contrast, or partly hidden by lane texture.", "May indicate joint failure, wheel-path stress, or drainage issues."],
                ["Pothole", "Localized missing pavement material.", "Shape and color vary with water, dirt, and lighting.", "Usually needs direct repair because it affects safety and ride quality."],
                ["Transverse Crack", "Crack line crossing the road direction.", "Can be short or broken into disconnected fragments.", "Can allow water intrusion and become wider over time."],
            ],
            [1.05 * inch, 1.65 * inch, 2.15 * inch, 1.65 * inch],
        )
    )

    section(story, "Classical Crack Detection Experiments")
    story.append(
        para(
            "Before modern deep learning became dominant, many pavement-crack systems used image-processing and handcrafted-feature "
            "pipelines. Cubero-Fernandez et al. used logarithmic transformation, bilateral filtering, Canny edge detection, morphological "
            "filtering, and a decision-tree classifier [4]. This is a useful reference because it shows the full chain that YOLO largely "
            "compresses into learned features."
        )
    )
    story.append(
        para(
            "The classical pipeline has an important strength: each stage is interpretable. The filter smooths noise, Canny highlights "
            "edges, morphology joins useful crack structures, and the classifier makes the final type decision. The weakness is that "
            "each stage has parameters that may fail when the road texture, camera angle, light, or crack thickness changes."
        )
    )
    story.append(
        styled_table(
            [
                ["Step in [4]", "Technical role", "Limitation for field images"],
                ["Logarithmic transform", "Enhances dark crack regions against pavement texture.", "Can also emphasize shadows and stains."],
                ["Bilateral filter", "Reduces noise while preserving edge information.", "Parameter choices depend on image quality."],
                ["Canny edge detection", "Finds abrupt intensity transitions that may correspond to cracks.", "Edges from aggregate texture can become false positives."],
                ["Morphological filter", "Cleans and connects candidate crack structures.", "Small cracks may be removed if filtering is too aggressive."],
                ["Decision tree", "Classifies crack type from extracted features.", "Depends on handcrafted geometry rather than learned context."],
            ],
            [1.35 * inch, 2.35 * inch, 2.75 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        para(
            "The paper reports an average 88% success rate for crack detection and about 80% success for crack-type classification [4]. "
            "Those results are strong for a handcrafted pipeline, but the same study also shows why classification is harder than binary "
            "crack detection: many errors come from assigning one crack type to another rather than missing the defect completely."
        )
    )

    section(story, "YOLOv3 Reference Experiment")
    story.append(
        para(
            "Nie and Wang adapted YOLOv3 to pavement crack detection and compared it with other detection approaches [1]. Their reported "
            "comparison is important because it shows the typical one-stage detector trade-off: YOLOv3 can be slightly less accurate than "
            "a two-stage detector such as Faster R-CNN at some IoU thresholds, but it is much faster."
        )
    )
    story.append(
        styled_table(
            [
                ["Method in [1]", "Test images", "Reported mAP", "Reported speed"],
                ["RetinaNet", "400", "49.7", "19 fps"],
                ["Faster R-CNN", "400", "53.1", "12 fps"],
                ["YOLOv3", "400", "51.2", "60 fps"],
            ],
            [1.75 * inch, 1.1 * inch, 1.2 * inch, 1.2 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "The same reference also compares YOLOv3 with traditional crack-detection methods. YOLOv3 is reported at 88% accuracy and "
            "60 fps, while grayscale image plus SVM and structured light were reported near 83% and 80% accuracy without the same real-time "
            "speed profile [1]. This supports the central design choice of this project: use a one-stage detector for a practical image "
            "testing workflow rather than a multi-step handcrafted image-processing chain."
        )
    )

    section(story, "Modern YOLOv8 Literature Experiments")
    story.append(
        para(
            "The most relevant modern reference is the SMG-YOLOv8 study by Zhang et al. [2]. That paper modifies YOLOv8s using space-to-depth "
            "convolution, a multi-scale convolutional attention module, and a G-GhostC2f structure. The reported result is not simply a bigger "
            "network; it is a more carefully structured model that improves feature preservation and multi-scale distress recognition."
        )
    )
    story.append(
        styled_table(
            [
                ["Experiment from [2]", "Reported finding", "Lesson applied in this project"],
                ["YOLOv8s baseline improvement", "SMG-YOLOv8 reached Pmacro 81.1% and mAP50 79.4%, improving over baseline by 8.2% and 12.5%.", "Architecture and feature fusion matter, especially for small and irregular cracks."],
                ["Comparison with YOLOv5n, YOLOv5s, YOLOv6s, YOLOv8n, and SSD", "SMG-YOLOv8 improved mAP50 compared with several baselines.", "Do not assume all YOLO variants behave equally on pavement distress."],
                ["Real-world project images", "The model reached Pmacro 80.5% and Rmacro 86.2%.", "Generalization to field images is a separate check from validation performance."],
            ],
            [1.55 * inch, 2.65 * inch, 2.3 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "Antariksa et al. compare YOLOv8m, YOLOv8n, and YOLOv8x for pavement-marking assessment [3]. Although the object category is "
            "pavement marking quality rather than cracks, the experiment is still useful because it evaluates transportation imagery under "
            "a speed-and-accuracy lens. Their results show that the lightest model can sometimes be the best operational choice when fast "
            "inference matters."
        )
    )
    story.append(
        styled_table(
            [
                ["Variant in [3]", "Good-class mAP0.5", "mAP@0.5:0.95", "Inference time"],
                ["YOLOv8m", "0.790", "0.388", "8.7 ms"],
                ["YOLOv8n", "0.823", "0.399", "3.4 ms"],
                ["YOLOv8x", "0.788", "0.382", "11.3 ms"],
            ],
            [1.3 * inch, 1.45 * inch, 1.45 * inch, 1.25 * inch],
        )
    )

    section(story, "Synthesis Of Reference Experiments")
    story.append(literature_result_chart())
    story.append(Paragraph("Figure 1. Result anchors from the supplied references and the selected project model.", STYLES["Caption"]))
    story.append(
        para(
            "The four reference papers do not use identical datasets or identical metrics, so the chart above should not be read as a direct leaderboard. "
            "It is included to show the technical range of results that motivated the project. Classical image processing and YOLOv3 both reach strong "
            "reported crack-detection results, while modern YOLOv8 research focuses more on multi-class distress recognition and real-world generalization."
        )
    )
    story.append(
        para(
            "The main pattern is that the problem changes as the method becomes more practical. A binary crack detector can report a high accuracy number, "
            "but a road-maintenance model also needs class labels, box locations, robustness to pavement texture, and usable speed. This project therefore "
            "uses a multi-class YOLO model and evaluates it with mAP, precision, recall, checkpoint size, and visual validation examples."
        )
    )
    story.append(
        styled_table(
            [
                ["Reference", "Experiment type", "Most important experience for this project"],
                ["Cubero-Fernandez et al. [4]", "Handcrafted image-processing plus decision-tree classification.", "Crack-type classification is harder than binary detection and needs class-specific evidence."],
                ["Nie and Wang [1]", "YOLOv3 crack detection against RetinaNet, Faster R-CNN, SVM, and structured light.", "One-stage detectors are attractive when speed matters, even if another model has slightly higher mAP."],
                ["Zhang et al. [2]", "SMG-YOLOv8 for multi-scene asphalt distress.", "Feature fusion, small-object preservation, and attention are valuable for complex pavement images."],
                ["Antariksa et al. [3]", "YOLOv8n, YOLOv8m, and YOLOv8x for transportation imagery.", "The most useful model is often a speed-accuracy compromise, not simply the largest checkpoint."],
            ],
            [1.55 * inch, 2.25 * inch, 2.7 * inch],
        )
    )

    section(story, "YOLO Algorithm Family")
    story.append(
        para(
            "YOLO means You Only Look Once. In object detection, this family is built around a one-stage idea: the model predicts "
            "where objects are and what class they belong to in one forward pass. This differs from two-stage detectors, where a "
            "first stage proposes candidate regions and a second stage classifies them."
        )
    )
    story.append(
        para(
            "For pavement inspection, the one-stage property is valuable because road images and video frames can be numerous. A road "
            "survey vehicle or a student demonstration website cannot wait for a slow multi-stage process for every frame. YOLO models "
            "make the problem practical by combining localization and classification in a single network pass."
        )
    )
    story.append(
        styled_table(
            [
                ["YOLO family", "Main technical idea", "Relevance to pavement crack detection"],
                ["YOLOv3", "Anchor-based predictions at multiple feature scales.", "Important crack-detection baseline with strong speed in [1]."],
                ["YOLOv5", "Practical PyTorch training workflow and small-to-large model variants.", "Common benchmark family in pavement-distress comparisons [2]."],
                ["YOLOv6", "Efficiency-oriented detector with deployment-aware design.", "Useful comparison point in multi-model pavement distress work [2]."],
                ["YOLOv8", "Anchor-free detection head and modern Ultralytics training workflow [5].", "Selected run uses YOLOv8m because it balances accuracy and usability."],
                ["YOLO11", "Recent Ultralytics generation for detection and other vision tasks [6].", "Several project runs evaluated YOLO11 variants."],
            ],
            [1.1 * inch, 2.75 * inch, 2.65 * inch],
        )
    )

    section(story, "YOLOv8 Architecture Details")
    story.append(yolo_v8_complexity_chart())
    story.append(Paragraph("Figure 2. YOLOv8 model-size and FLOP progression reported by Zhang et al. [2].", STYLES["Caption"]))
    story.append(
        para(
            "YOLOv8 is useful for this project because it provides several model sizes. The small variants are easier to run quickly, while the larger "
            "variants can learn richer features at the cost of storage and computation. Zhang et al. report the expected progression: the n, s, m, l, "
            "and x variants grow from 3.2 million to 68.2 million parameters and from 8.7 to 257.8 GFLOPs [2]."
        )
    )
    story.append(
        para(
            "The selected project model uses the medium variant, YOLOv8m. That choice is technically consistent with the report's model-selection logic. "
            "A nano model can be too limited for subtle road-surface variation, while an X model can be large without producing enough extra practical value. "
            "The medium model gives a stronger feature extractor while keeping the checkpoint size manageable."
        )
    )
    story.append(
        styled_table(
            [
                ["Component", "What it learns", "Why it matters for cracks"],
                ["Backbone", "Low-level texture, edge, and deeper semantic features.", "Cracks are often thin dark structures that need both edge and context cues."],
                ["Neck", "Multi-scale feature fusion.", "Potholes, alligator patterns, and thin line cracks appear at different sizes."],
                ["Detection head", "Box coordinates, confidence, and class scores.", "Turns learned features into visible crack labels and rectangles."],
                ["Post-processing", "Thresholding and non-maximum suppression.", "Removes weak predictions and duplicate boxes around the same crack."],
            ],
            [1.35 * inch, 2.45 * inch, 2.7 * inch],
        )
    )

    section(story, "Loss Functions And Optimization")
    story.append(
        para(
            "Training a YOLO detector requires several losses at the same time. The model must learn where the box is, whether the object exists, and which "
            "class the object belongs to. During training, box loss improves localization, classification loss improves label assignment, and distribution "
            "or localization-related losses help make the predicted box more precise."
        )
    )
    story.append(
        para(
            "The archived project runs use 50 epochs, 640-pixel image size, batch size 32, and the MuSGD optimizer. Keeping these settings similar across "
            "runs makes the comparison more meaningful. If the training setup changed dramatically between runs, it would be harder to tell whether the "
            "difference came from the model family, the dataset version, or the optimization setup."
        )
    )
    story.append(
        styled_table(
            [
                ["Training element", "Role in learning", "Project interpretation"],
                ["Epochs = 50", "Number of passes over the training set.", "Enough to compare trends without making the report dependent on very long training."],
                ["Image size = 640", "Resolution used during training and validation.", "Balances detail for thin cracks with practical computation."],
                ["Batch size = 32", "Number of images processed before an update.", "Keeps training stable across all archived runs."],
                ["MuSGD optimizer", "Updates network weights from loss gradients.", "Consistent optimizer across versions supports fairer comparison."],
                ["Validation curves", "Track whether detection quality improves.", "Used to judge whether final metrics are supported by training behavior."],
            ],
            [1.35 * inch, 2.4 * inch, 2.75 * inch],
        )
    )

    section(story, "How YOLO Works")
    story.append(detector_process_drawing())
    story.append(Paragraph("Figure 1. Simplified YOLO-style detection pipeline.", STYLES["Caption"]))
    story.append(
        para(
            "A YOLO detector first resizes the input image and passes it through a backbone network. The backbone extracts edges, textures, "
            "surface patterns, and higher-level cues. For cracks, useful features include long dark lines, connected fracture patterns, "
            "missing-surface regions, and contrast changes between the defect and the asphalt."
        )
    )
    story.append(
        para(
            "The neck then combines features from different spatial scales. This is especially important for pavement distress because a "
            "pothole can be visually large while a longitudinal crack may be very thin. Multi-scale feature fusion helps the detector keep "
            "both small local details and broader context."
        )
    )
    story.append(
        para(
            "The detection head predicts bounding-box coordinates, a confidence score, and class probabilities. During inference, low-confidence "
            "candidates are filtered, and overlapping boxes are reduced using non-maximum suppression. The final output is a compact set of "
            "boxes with class names and confidence values."
        )
    )

    section(story, "Bounding Boxes, IoU, And NMS")
    story.append(
        para(
            "Object detection is different from image classification because the model must both recognize and locate the defect. The bounding "
            "box is the localization output. A correct prediction must have the right class and enough overlap with the ground-truth box."
        )
    )
    story.append(
        styled_table(
            [
                ["Concept", "Definition", "Why it matters"],
                ["Bounding box", "A rectangle around the predicted crack or pothole.", "Allows the user to see exactly where the distress was found."],
                ["Confidence", "The model score for a predicted object.", "Controls how strict or permissive the detection threshold is."],
                ["Intersection over Union", "Overlap area divided by total covered area for predicted and true boxes.", "Used to decide whether localization is close enough."],
                ["Non-maximum suppression", "Keeps the strongest box when several boxes overlap heavily.", "Prevents duplicate detections for one crack segment."],
                ["Class probability", "The model's class choice for the detected box.", "Separates alligator, longitudinal, pothole, and transverse damage."],
            ],
            [1.4 * inch, 2.45 * inch, 2.6 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "For crack detection, bounding boxes are an approximation. A thin curved crack does not fill a rectangle perfectly, so even a useful "
            "prediction may include pavement background inside the box. This is one reason mAP50-95 is harder than mAP50: tighter IoU thresholds "
            "punish imperfect box alignment more strongly."
        )
    )

    section(story, "Metrics Used In This Project")
    story.append(
        para(
            "The model comparison uses precision, recall, mAP50, and mAP50-95. These metrics answer different questions. Precision asks whether "
            "predicted crack boxes are trustworthy. Recall asks whether real crack objects are being found. mAP summarizes the precision-recall "
            "curve across confidence thresholds and object classes."
        )
    )
    story.append(
        styled_table(
            [
                ["Metric", "Plain-language meaning", "Interpretation for crack detection"],
                ["Precision", "Among predicted detections, the share that are correct.", "High precision means fewer false crack alerts."],
                ["Recall", "Among real objects, the share that are detected.", "High recall means fewer missed cracks."],
                ["mAP50", "Mean average precision when IoU threshold is 0.50.", "Good for checking whether the detector finds approximate crack locations."],
                ["mAP50-95", "Mean average precision averaged over stricter IoU thresholds.", "Better measure of localization quality and box tightness."],
                ["Model size", "Storage needed for the checkpoint.", "Important when the model must be shared or loaded quickly."],
            ],
            [1.25 * inch, 2.35 * inch, 2.9 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "The selected model's mAP50 of 0.8844 means it performs strongly when an approximate localization threshold is used. The lower "
            "mAP50-95 value of 0.5164 is normal for crack datasets because thin, irregular objects are harder to localize tightly than compact "
            "objects such as cars or signs."
        )
    )

    section(story, "Dataset Labels And Class Design")
    story.append(
        para(
            "The active class names are Alligator Crack, Longitudinal Crack, Pothole, and Transverse Crack. This class list aligns well with "
            "the supplied references. Cubero-Fernandez et al. discuss transverse, longitudinal, and mesh/alligator crack categories [4], while "
            "Zhang et al. explicitly evaluate longitudinal cracks, transverse cracks, mesh cracks, and potholes in asphalt-distress detection [2]."
        )
    )
    story.append(
        para(
            "Good object-detection labels need consistent box placement. For cracks, this is harder than for rigid objects because a crack can be "
            "curved, fragmented, or connected to other cracks. If two annotators draw boxes differently around the same crack, training noise increases. "
            "The validation curves and confusion matrices therefore need to be interpreted together, not as independent proof."
        )
    )
    story.append(image_flowable(selected_dir / "labels.jpg", "Figure 2. Label distribution and annotation overview for the selected training run.", max_height=3.35 * inch))

    section(story, "Training Versions And Experiment Design")
    story.append(training_process_drawing())
    story.append(Paragraph("Figure 3. Project training and model-selection process.", STYLES["Caption"]))
    story.append(
        para(
            "The project preserved three dataset/model versions. Version-1 includes six runs, while version-2 and version-3 include one run each. "
            "Every run was trained for 50 epochs with 640-pixel images and batch size 32. This makes the comparison more consistent because the "
            "main differences are the dataset version and the starting model family."
        )
    )
    story.append(
        styled_table(
            [
                ["Version", "Runs", "Model families evaluated", "Dataset reference", "Common training settings"],
                ["version-1", "6", "YOLO11m, YOLO26m, YOLO26x, YOLO11x, YOLO11n, YOLOv8m", "RDD-1 and RDD-5", "50 epochs, image size 640, batch 32, MuSGD"],
                ["version-2", "1", "YOLO26m", "RDD-2", "50 epochs, image size 640, batch 32, MuSGD"],
                ["version-3", "1", "YOLO26x", "RDD-4", "50 epochs, image size 640, batch 32, MuSGD"],
            ],
            [0.85 * inch, 0.55 * inch, 2.35 * inch, 1.15 * inch, 1.9 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "Large model checkpoints are kept in the Hugging Face model repository, while plots, CSVs, argument files, and validation images are "
            "kept in GitHub. The model repository is: "
        )
    )
    story.append(Paragraph(MODEL_REPO_URL, STYLES["Link"]))

    section(story, "Project Model Comparison")
    story.append(
        para(
            "The following table summarizes the final validation metrics from each run. The highlighted row is the selected model. A single metric "
            "does not fully determine the best model: this project considers mAP, precision, recall, size, family support, and behavior in practical "
            "image testing."
        )
    )
    rows = [["Version", "Run", "Base model", "Data", "Size MB", "mAP50", "mAP50-95", "Precision", "Recall"]]
    for row in MODEL_ROWS:
        version, run, base_model, data, size_mb, map50, map5095, precision, recall = row
        rows.append([version, run, base_model, data, f"{size_mb:.2f}", f"{map50:.4f}", f"{map5095:.4f}", f"{precision:.4f}", f"{recall:.4f}"])
    story.append(
        styled_table(
            rows,
            [0.72 * inch, 0.48 * inch, 0.88 * inch, 0.55 * inch, 0.58 * inch, 0.62 * inch, 0.72 * inch, 0.7 * inch, 0.6 * inch],
            font_size=6.7,
            highlight_row=6,
        )
    )
    story.append(Paragraph("Table 1. Final-epoch model comparison. Highlighted row: version-1/run-6.", STYLES["Caption"]))
    story.append(model_bar_chart())
    story.append(Paragraph("Figure 4. mAP50 and mAP50-95 comparison across project runs.", STYLES["Caption"]))

    section(story, "Accuracy, Size, Precision, And Recall Trade-Offs")
    story.append(
        para(
            "A deployment-oriented model choice should compare more than one axis. The first trade-off plot below places mAP50 against checkpoint size. "
            "The selected model is not the smallest and not the largest; it sits near the strong-accuracy group while avoiding the storage cost of the X-size "
            "runs. This is exactly the kind of balance recommended by the YOLOv8 comparison experience in [3]."
        )
    )
    story.append(model_tradeoff_chart())
    story.append(Paragraph("Figure 5. Checkpoint size versus mAP50 for the archived project runs.", STYLES["Caption"]))
    story.append(
        para(
            "The second trade-off plot places precision against recall. In pavement inspection, high recall reduces missed cracks, while high precision reduces "
            "unnecessary alerts. Version-1/run-6 stays in the balanced part of the plot: it is not the highest-recall model, but it avoids the weaker precision "
            "profile of the smallest run and remains suitable for the four-class image testing task."
        )
    )
    story.append(precision_recall_chart())
    story.append(Paragraph("Figure 6. Precision versus recall for the archived project runs.", STYLES["Caption"]))
    story.append(
        styled_table(
            [
                ["Decision question", "Metric evidence", "Interpretation"],
                ["Should the smallest model be used?", "version-1/run-5 has size 5.48 MB but mAP50 0.8574 and recall 0.8069.", "Small size is attractive, but this run gives up too much detection quality."],
                ["Should the largest model be used?", "version-1/run-3 and version-1/run-4 exceed 114 MB but do not dominate mAP50.", "Extra capacity does not automatically create better crack detection."],
                ["Is version-1/run-6 balanced?", "52.04 MB, mAP50 0.8844, precision 0.8294, recall 0.8111.", "This is a strong middle option for the report's public image-testing goal."],
            ],
            [1.65 * inch, 2.55 * inch, 2.3 * inch],
        )
    )

    section(story, "Why Version-1 Run-6 Was Chosen")
    story.append(
        para(
            "Version-1/run-6 was chosen because it offered the best practical balance rather than the single highest value in one column. "
            "Version-1/run-1 has the highest mAP50, but run-6 uses YOLOv8m, which is a stable and widely supported Ultralytics model family. "
            "It is also much smaller than the X-size checkpoints while staying close to the strongest validation results."
        )
    )
    story.append(
        styled_table(
            [
                ["Selection criterion", "Version-1/run-6 evidence", "Interpretation"],
                ["Accuracy", "mAP50 = 0.8844 and mAP50-95 = 0.5164.", "Strong detection performance with competitive tight-box performance."],
                ["Precision and recall", "Precision = 0.8294 and recall = 0.8111.", "Balanced behavior: not only finding cracks but keeping false detections controlled."],
                ["Model size", "About 52 MB.", "Practical compared with 114 MB and 118 MB X-size checkpoints."],
                ["Model family", "YOLOv8m.", "Well supported by the Ultralytics workflow and consistent with modern YOLO literature."],
                ["Class support", "Four pavement distress labels.", "Matches the intended image testing task."],
            ],
            [1.45 * inch, 2.3 * inch, 2.65 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "The choice is also consistent with the lesson from Antariksa et al. [3]: operational performance depends on a balance between accuracy, "
            "speed, and model complexity. A larger model can be attractive for maximum accuracy, but the middle model is often the better practical "
            "choice when a demonstration must stay responsive."
        )
    )

    section(story, "Selected Run Training Curves")
    story.append(
        para(
            "The training-curve figure for version-1/run-6 shows the relationship between training losses, validation losses, and detection metrics "
            "over 50 epochs. Loss curves indicate whether the model is learning stable representations, while mAP curves show whether those representations "
            "translate into useful detections."
        )
    )
    story.append(image_flowable(selected_dir / "results.png", "Figure 5. Training and validation curves for version-1/run-6.", max_height=5.2 * inch))

    section(story, "Precision-Recall Evidence")
    story.append(
        para(
            "Precision-recall curves are especially helpful for crack detection because users can adjust confidence thresholds. A lower threshold may "
            "find more cracks but also produce more false positives. A higher threshold may show fewer boxes but make each prediction more trustworthy. "
            "The selected run's curves show how the model behaves across that threshold range."
        )
    )
    story.append(image_flowable(selected_dir / "BoxPR_curve.png", "Figure 6. Precision-recall curve for version-1/run-6.", max_height=3.3 * inch))
    story.append(image_flowable(selected_dir / "BoxF1_curve.png", "Figure 7. F1-confidence curve for version-1/run-6.", max_height=3.05 * inch))

    section(story, "Confidence Threshold Behavior")
    story.append(
        para(
            "The precision-confidence and recall-confidence curves show how model behavior changes as the confidence cutoff changes. For a public image "
            "testing page, the default threshold should be low enough to show likely cracks but high enough to avoid drawing boxes on ordinary pavement "
            "texture. In this project, a confidence value around 0.25 is useful as a permissive starting point because users can still inspect and adjust "
            "the outputs visually."
        )
    )
    story.append(image_flowable(selected_dir / "BoxP_curve.png", "Figure 8. Precision-confidence curve for version-1/run-6.", max_height=3.25 * inch))
    story.append(image_flowable(selected_dir / "BoxR_curve.png", "Figure 9. Recall-confidence curve for version-1/run-6.", max_height=3.25 * inch))

    section(story, "Confusion Matrix And Error Analysis")
    story.append(
        para(
            "The confusion matrix shows which classes are separated cleanly and which are more likely to be confused. In pavement distress, confusion is "
            "expected between alligator and transverse or longitudinal cracks when the visible region is small or when a network of cracks is only partly "
            "captured inside one box. Potholes are usually more compact but may be confused with dark patches or surface stains."
        )
    )
    story.append(image_flowable(selected_dir / "confusion_matrix_normalized.png", "Figure 10. Normalized confusion matrix for version-1/run-6.", max_height=5.2 * inch))

    section(story, "Validation Prediction Examples")
    story.append(
        para(
            "Validation prediction images are useful because metrics alone cannot show whether detections look reasonable. The examples below show predicted "
            "boxes on held-out validation images. These figures helped confirm that the selected model produced interpretable boxes and labels rather than "
            "only strong aggregate metrics."
        )
    )
    story.append(
        styled_table(
            [
                ["Visual check", "What a good result looks like", "Why it matters"],
                ["Box placement", "Boxes cover the visible distress without swallowing too much clean pavement.", "Tighter boxes make the output easier for a user to trust."],
                ["Class label", "The predicted label matches the crack orientation or distress shape.", "Correct labels connect detection to maintenance meaning."],
                ["Duplicate boxes", "One visible defect is not repeated many times with overlapping boxes.", "Duplicate detections can exaggerate the amount of damage."],
                ["Background errors", "Shadows, lane markings, and stains are mostly ignored.", "Road imagery contains many crack-like distractors."],
            ],
            [1.35 * inch, 2.75 * inch, 2.4 * inch],
        )
    )
    story.append(image_flowable(selected_dir / "val_batch0_pred.jpg", "Figure 11. Validation prediction batch 0 for version-1/run-6.", max_height=5.25 * inch))

    section(story, "More Visual Evidence From The Selected Run")
    story.append(
        para(
            "The second validation batch gives another qualitative check. In technical evaluation, this type of visual review is important because it can "
            "reveal problems that metrics hide, such as duplicated boxes, boxes around shadows, or labels that are correct numerically but not useful for "
            "human inspection."
        )
    )
    story.append(
        styled_table(
            [
                ["Question for review", "How the validation image helps answer it"],
                ["Are all four classes visually plausible?", "The batch shows crack-like regions with class labels, allowing a human reader to compare label names against road orientation and shape."],
                ["Does the model react to real road context?", "The examples include lanes, vehicles, buildings, bright sky, and shadows rather than isolated crack crops."],
                ["Does confidence remain interpretable?", "Confidence values appear beside boxes, so the reader can see which predictions are strong and which are borderline."],
            ],
            [1.75 * inch, 4.75 * inch],
        )
    )
    story.append(image_flowable(selected_dir / "val_batch1_pred.jpg", "Figure 12. Validation prediction batch 1 for version-1/run-6.", max_height=5.25 * inch))

    section(story, "Cross-Version Training Observations")
    story.append(
        para(
            "The three versions provide a small but useful experiment archive. Version-1/run-6 is selected, but the other versions help explain the choice. "
            "Version-2/run-1 has strong precision but lower mAP50 and mAP50-95. Version-3/run-1 uses a larger model and has competitive recall, but its "
            "checkpoint size is more than twice the selected model size."
        )
    )
    story.append(
        styled_table(
            [
                ["Observation", "Evidence", "Meaning"],
                ["The smallest model is not the best here.", "version-1/run-5 is only 5.48 MB but has the lowest mAP50 and recall.", "Speed and size alone are not enough."],
                ["The largest model is not automatically best.", "X-size runs exceed 114 MB but do not dominate all metrics.", "Model capacity can overfit or become inefficient."],
                ["Dataset version changes matter.", "RDD-1, RDD-2, RDD-4, and RDD-5 runs have different metric profiles.", "Training data composition affects final behavior."],
                ["The selected model is a middle-size compromise.", "YOLOv8m, 52.04 MB, mAP50 0.8844.", "Good fit for a public demonstration model."],
            ],
            [1.55 * inch, 2.5 * inch, 2.45 * inch],
        )
    )
    story.append(Spacer(1, 6))
    story.append(image_flowable(ROOT / "versions" / "version-2" / "run-1" / "detect" / "train" / "results.png", "Figure 13. Training curves for version-2/run-1.", max_height=2.85 * inch))
    story.append(image_flowable(ROOT / "versions" / "version-3" / "run-1" / "detect" / "train" / "results.png", "Figure 14. Training curves for version-3/run-1.", max_height=2.85 * inch))

    section(story, "Qualitative Comparison Across Versions")
    story.append(
        para(
            "The version archive also includes validation prediction images. These images are important because two models can have similar metrics while "
            "producing different visual behavior. A model may draw boxes that are too loose, miss small cracks, duplicate detections, or label a borderline "
            "crack type differently. The examples below use held-out validation batches from different runs to support the metric-based discussion."
        )
    )
    story.append(
        image_grid(
            [
                (ROOT / "versions" / "version-1" / "run-1" / "detect" / "train" / "val_batch0_pred.jpg", "version-1/run-1 validation predictions"),
                (ROOT / "versions" / "version-1" / "run-6" / "detect" / "train" / "val_batch0_pred.jpg", "version-1/run-6 validation predictions"),
                (ROOT / "versions" / "version-2" / "run-1" / "detect" / "train" / "val_batch0_pred.jpg", "version-2/run-1 validation predictions"),
                (ROOT / "versions" / "version-3" / "run-1" / "detect" / "train" / "val_batch0_pred.jpg", "version-3/run-1 validation predictions"),
            ],
            max_height=1.95 * inch,
        )
    )
    story.append(
        para(
            "The comparison supports the final selection because version-1/run-6 produces readable boxes while keeping the model size moderate. The other "
            "runs remain valuable evidence: run-1 explains the high-mAP option, version-2 shows a different dataset split with strong precision, and "
            "version-3 shows how a larger model can remain competitive without becoming the best practical choice."
        )
    )
    story.append(
        image_grid(
            [
                (ROOT / "versions" / "version-1" / "run-6" / "detect" / "train" / "train_batch0.jpg", "Selected run training batch example"),
                (ROOT / "versions" / "version-1" / "run-6" / "detect" / "train" / "val_batch2_pred.jpg", "Selected run validation batch 2 predictions"),
            ],
            max_height=2.25 * inch,
        )
    )

    section(story, "From Detection Output To Maintenance Insight")
    story.append(
        para(
            "The model output is a set of labels, boxes, and confidence scores. For a road-maintenance workflow, those raw predictions become useful when "
            "they are interpreted as maintenance signals. A pothole with high confidence is usually more immediately actionable than a low-confidence thin "
            "crack. An alligator-crack pattern may suggest fatigue distress, while a transverse crack may suggest water-entry risk or thermal movement."
        )
    )
    story.append(
        styled_table(
            [
                ["Detected type", "Typical interpretation", "Useful follow-up measurement"],
                ["Alligator Crack", "Possible fatigue or structural distress.", "Crack density, affected area, and repeated occurrence along a segment."],
                ["Longitudinal Crack", "Possible joint, lane-edge, or wheel-path distress.", "Length, lane position, and whether the crack is widening."],
                ["Pothole", "Localized missing surface material with direct safety impact.", "Area, depth if available, and proximity to wheel path."],
                ["Transverse Crack", "Cross-road cracking that can admit water.", "Length, spacing between cracks, and edge deterioration."],
            ],
            [1.3 * inch, 2.55 * inch, 2.65 * inch],
        )
    )
    story.append(
        para(
            "This is also why confidence scores should not be treated as severity scores. Confidence means the model believes the object is present and "
            "classified correctly. Severity requires additional information such as crack width, depth, density, road location, traffic volume, and whether "
            "the same distress appears repeatedly over time."
        )
    )

    section(story, "Reference Lessons Applied To This Project")
    story.append(
        para(
            "The references shaped the interpretation of this project in four ways. First, [4] shows that crack-type classification is a separate problem "
            "from crack presence detection. Second, [1] shows why YOLO is attractive when speed matters. Third, [2] shows that multi-scale fusion and attention "
            "can improve asphalt distress recognition. Fourth, [3] shows that model selection should consider computational efficiency as well as accuracy."
        )
    )
    story.append(
        styled_table(
            [
                ["Reference lesson", "How it affects this project"],
                ["Traditional pipelines are interpretable but fragile [4].", "Use validation visuals and not only handcrafted thresholds."],
                ["YOLOv3 is fast enough for real-time crack detection [1].", "Keep one-stage detection as the main method family."],
                ["Multi-scale and attention modules improve complex distress detection [2].", "Prefer detectors that preserve small and large distress features."],
                ["Light or middle models can be more useful than the largest model [3].", "Select version-1/run-6 for balance rather than maximum size."],
                ["Field generalization is different from validation performance [2].", "Future testing should include weather, lighting, and camera-angle variation."],
            ],
            [2.65 * inch, 3.85 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        para(
            "Together, these lessons make the final model decision more defensible. The selected run is not just a number in a table; it is a model whose "
            "accuracy, size, training evidence, and literature-aligned behavior make sense for the intended image-testing use case."
        )
    )

    section(story, "Public Demonstration URL")
    story.append(
        para(
            "The website is referenced only as the public testing location for this project. Detailed implementation choices are intentionally not described "
            "in this report because the technical focus is the YOLO detection method, experimental comparison, and model-selection evidence."
        )
    )
    story.append(Paragraph(f"Public project page: {PROJECT_URL}", STYLES["Link"]))
    story.append(Paragraph(f"Image testing page: {IMAGE_TEST_URL}", STYLES["Link"]))
    story.append(Spacer(1, 10))
    story.append(
        para(
            "A user can open the image testing page, upload a pavement image, and review the detected crack class, confidence value, and bounding box. "
            "This is the simplest demonstration of the trained model because still images allow the result to be inspected carefully."
        )
    )

    section(story, "Limitations And Uncertainty")
    story.append(
        para(
            "The current results should be interpreted as validation evidence, not a guarantee of field performance under every road condition. Pavement "
            "images can vary by camera height, weather, sun angle, road material, shadows, lane markings, dirt, water, and motion blur. These conditions can "
            "change both the visibility of cracks and the model's confidence."
        )
    )
    story.append(
        para(
            "Another limitation is box annotation for thin objects. A bounding box around a crack can include a large amount of background pavement, so "
            "mAP50-95 is naturally difficult. Instance segmentation or crack-line segmentation may represent long thin cracks better than rectangular boxes, "
            "but object detection is easier to deploy and easier for users to understand."
        )
    )
    story.append(
        styled_table(
            [
                ["Risk", "Technical cause", "Mitigation"],
                ["False positives on shadows", "Dark shadows resemble cracks.", "Add more shadow-heavy negative images."],
                ["Missed thin cracks", "Small low-contrast objects have weak features.", "Add close-up examples and consider higher-resolution training."],
                ["Class confusion", "Crack types can overlap visually.", "Improve annotation rules and add more borderline examples."],
                ["Poor field generalization", "Validation set may not cover all road scenes.", "Test on new roads, lighting, weather, and camera positions."],
            ],
            [1.45 * inch, 2.25 * inch, 2.8 * inch],
        )
    )

    section(story, "Future Technical Improvements")
    story.append(
        para(
            "The next technical improvement would be a larger and more varied test set. The references show that generalization is a central issue: models "
            "can perform well in controlled datasets while struggling with multi-scene road images [2]. The project should therefore include additional "
            "images from different pavement materials, weather conditions, camera viewpoints, and road environments."
        )
    )
    story.append(
        para(
            "A second improvement is severity estimation. The current labels identify the type of distress, but maintenance decisions also depend on severity, "
            "width, length, density, and location. A future model could estimate severity from box size, crack density, or a segmentation mask. This would move "
            "the project from detection toward maintenance prioritization."
        )
    )
    story.append(
        para(
            "A third improvement is model-family experimentation. Based on [2], a future version could test a YOLOv8-derived architecture with additional "
            "multi-scale attention or a lightweight feature block. Based on [3], the experiment should record inference time and memory use alongside mAP so "
            "the selected model is justified by operational evidence, not accuracy alone."
        )
    )
    for item in [
        "Add a larger independent test split from roads not used during training.",
        "Record inference speed for every checkpoint and compare accuracy per megabyte.",
        "Evaluate segmentation models for long thin cracks.",
        "Add severity estimation for maintenance prioritization.",
        "Add more negative examples containing shadows, stains, lane markings, gravel, and repaired pavement.",
    ]:
        story.append(bullet(item))

    section(story, "Conclusion")
    story.append(
        para(
            "This project demonstrates a complete technical workflow for pavement crack detection with YOLO. The method is grounded in earlier work on "
            "traditional crack detection, YOLOv3 crack detection, modern YOLOv8 asphalt distress recognition, and YOLOv8 variant comparison for transportation "
            "imagery. The selected model, version-1/run-6, is a YOLOv8m run that balances accuracy, size, class support, and practical usability."
        )
    )
    story.append(
        para(
            "The final report evidence supports the choice through quantitative metrics, training curves, precision-recall curves, confusion-matrix review, "
            "validation prediction examples, and cross-version comparison. The main technical conclusion is that the best model for this project is not simply "
            "the largest model or the model with one isolated metric peak. It is the model that performs reliably enough while staying practical for public "
            "image testing."
        )
    )

    section(story, "References")
    add_reference_list(story)
    return story


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    doc = NumberedDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.62 * inch,
        title="Pavement Crack Detection Final Project Report",
        author="Hamidreza Khalaj Zahraei",
    )
    story = build_story()
    doc.multiBuild(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
