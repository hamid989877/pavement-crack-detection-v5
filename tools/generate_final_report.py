from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.graphics import renderPDF
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


MODEL_ROWS = [
    ("version-1", "run-1", "yolo11m.pt", 40.52, 0.8940, 0.5254, 0.8331, 0.8283),
    ("version-1", "run-2", "yolo26m.pt", 44.03, 0.8813, 0.5123, 0.8201, 0.8337),
    ("version-1", "run-3", "yolo26x.pt", 118.33, 0.8804, 0.5151, 0.8401, 0.8104),
    ("version-1", "run-4", "yolo11x.pt", 114.40, 0.8821, 0.5157, 0.8215, 0.8311),
    ("version-1", "run-5", "yolo11n.pt", 5.48, 0.8574, 0.4923, 0.7851, 0.8069),
    ("version-1", "run-6", "yolov8m.pt", 52.04, 0.8844, 0.5164, 0.8294, 0.8111),
    ("version-2", "run-1", "yolo26m.pt", 44.03, 0.8453, 0.4952, 0.8461, 0.8014),
    ("version-3", "run-1", "yolo26x.pt", 118.33, 0.8695, 0.5075, 0.8317, 0.8286),
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
        "distresses based on deep learning."
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
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in ("Heading1", "Heading2"):
                level = 0 if style_name == "Heading1" else 1
                text = flowable.getPlainText()
                key = f"section-{self.seq.nextf('section')}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, text, self.page, key))


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
        canvas.drawString(doc.leftMargin, 0.35 * inch, "Pavement Crack Detection V5 - Final Project Report")
        canvas.drawRightString(width - doc.rightMargin, 0.35 * inch, f"Page {page}")
    canvas.restoreState()


def stylesheet():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=35,
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
            fontSize=15,
            leading=20,
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
    base["Heading1"].fontSize = 18
    base["Heading1"].leading = 22
    base["Heading1"].textColor = ASPHALT
    base["Heading1"].spaceBefore = 12
    base["Heading1"].spaceAfter = 8
    base["Heading2"].fontName = "Helvetica-Bold"
    base["Heading2"].fontSize = 13
    base["Heading2"].leading = 16
    base["Heading2"].textColor = ASPHALT_DARK
    base["Heading2"].spaceBefore = 8
    base["Heading2"].spaceAfter = 6
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 9.7
    base["Normal"].leading = 13.5
    base["Normal"].alignment = TA_JUSTIFY
    base["Normal"].spaceAfter = 6
    base.add(
        ParagraphStyle(
            "Body",
            parent=base["Normal"],
            alignment=TA_JUSTIFY,
        )
    )
    base.add(
        ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#444444"),
        )
    )
    base.add(
        ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontSize=8.5,
            leading=10.5,
            alignment=TA_CENTER,
            textColor=ASPHALT_LIGHT,
            spaceBefore=3,
            spaceAfter=9,
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
    return base


STYLES = stylesheet()


def para(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def heading(text: str, level: int = 1) -> Paragraph:
    return Paragraph(text, STYLES["Heading1" if level == 1 else "Heading2"])


def bullet(text: str) -> Paragraph:
    return Paragraph(text, STYLES["ProjectBullet"], bulletText="-")


def image_flowable(path: Path, caption: str, max_width=6.5 * inch, max_height=3.9 * inch):
    if not path.exists():
        return KeepTogether([para(f"Missing figure: {path}", "Small")])
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    flow = [
        Image(str(path), width=width * scale, height=height * scale),
        Paragraph(caption, STYLES["Caption"]),
    ]
    return KeepTogether(flow)


def styled_table(rows, col_widths=None, font_size=8, repeat_rows=1, highlight_row=None):
    processed = []
    for row in rows:
        processed.append([Paragraph(str(cell), STYLES["Small"]) for cell in row])
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


def workflow_drawing():
    drawing = Drawing(470, 140)
    boxes = [
        (10, 80, "User Upload", "image or video"),
        (125, 80, "FastAPI Backend", "validation and job API"),
        (250, 80, "YOLO Model", "best.pt inference"),
        (365, 80, "Result", "boxes, video, log"),
        (125, 20, "Static UI", "home, image, video pages"),
        (250, 20, "Outputs", "annotated media"),
    ]
    for x, y, title, sub in boxes:
        drawing.add(Rect(x, y, 95, 44, fillColor=colors.HexColor("#F7F7F7"), strokeColor=ASPHALT_LIGHT))
        drawing.add(String(x + 8, y + 27, title, fontName="Helvetica-Bold", fontSize=8.5, fillColor=ASPHALT))
        drawing.add(String(x + 8, y + 12, sub, fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    for x1, y1, x2, y2 in [(105, 102, 125, 102), (220, 102, 250, 102), (345, 102, 365, 102), (172, 80, 172, 64), (297, 80, 297, 64)]:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=AMBER, strokeWidth=2))
    drawing.add(String(10, 132, "Application workflow", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    return drawing


def model_bar_chart():
    width = 470
    height = 170
    drawing = Drawing(width, height)
    drawing.add(String(10, 155, "mAP50 and mAP50-95 comparison by training run", fontName="Helvetica-Bold", fontSize=10, fillColor=ASPHALT))
    chart_x = 55
    chart_y = 30
    chart_w = 390
    chart_h = 110
    drawing.add(Line(chart_x, chart_y, chart_x + chart_w, chart_y, strokeColor=ASPHALT_LIGHT))
    drawing.add(Line(chart_x, chart_y, chart_x, chart_y + chart_h, strokeColor=ASPHALT_LIGHT))
    max_value = 1.0
    group_w = chart_w / len(MODEL_ROWS)
    for idx, row in enumerate(MODEL_ROWS):
        _, run, _, _, map50, map5095, _, _ = row
        x = chart_x + idx * group_w + 7
        h1 = map50 / max_value * chart_h
        h2 = map5095 / max_value * chart_h
        drawing.add(Rect(x, chart_y, 8, h1, fillColor=AMBER, strokeColor=AMBER))
        drawing.add(Rect(x + 10, chart_y, 8, h2, fillColor=ASPHALT_LIGHT, strokeColor=ASPHALT_LIGHT))
        label = run.replace("run-", "r")
        drawing.add(String(x - 2, chart_y - 12, label, fontName="Helvetica", fontSize=6.5, fillColor=ASPHALT_LIGHT))
    drawing.add(String(12, chart_y + chart_h - 2, "1.0", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(String(14, chart_y + chart_h * 0.5 - 2, "0.5", fontName="Helvetica", fontSize=7, fillColor=ASPHALT_LIGHT))
    drawing.add(Rect(310, 150, 8, 8, fillColor=AMBER, strokeColor=AMBER))
    drawing.add(String(323, 151, "mAP50", fontName="Helvetica", fontSize=7, fillColor=ASPHALT))
    drawing.add(Rect(365, 150, 8, 8, fillColor=ASPHALT_LIGHT, strokeColor=ASPHALT_LIGHT))
    drawing.add(String(378, 151, "mAP50-95", fontName="Helvetica", fontSize=7, fillColor=ASPHALT))
    return drawing


def add_reference_list(story):
    for idx, ref in enumerate(REFERENCES, start=1):
        story.append(Paragraph(f"[{idx}] {ref}", STYLES["Small"]))
        story.append(Spacer(1, 3))


def build_story():
    story = []

    cover_img = ROOT / "static" / "assets" / "hero-crack-background.jpg"
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Pavement Crack Detection V5", STYLES["CoverTitle"]))
    story.append(Paragraph("Final Project Report", STYLES["CoverSubtitle"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Hamidreza Khalaj Zahraei", STYLES["CoverSubtitle"]))
    story.append(Paragraph("Student ID: 300204093", STYLES["CoverSubtitle"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(image_flowable(cover_img, "Cover image: asphalt pavement crack used as the project hero background.", max_width=6.3 * inch, max_height=3.0 * inch))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        para(
            "This report documents the development of a YOLO-based pavement crack detection system, "
            "including the training-version archive, model comparison, deployment-oriented model selection, "
            "and website implementation for image and video inference."
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Table of Contents", STYLES["TOCTitle"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOCHeading1", fontName="Helvetica", fontSize=10, leftIndent=0, firstLineIndent=0, leading=13),
        ParagraphStyle("TOCHeading2", fontName="Helvetica", fontSize=9, leftIndent=18, firstLineIndent=0, leading=12),
    ]
    story.append(toc)
    story.append(PageBreak())

    story.append(heading("Executive Summary"))
    story.append(
        para(
            "The project addresses automated pavement distress detection using object detection. "
            "Manual road inspection is slow, subjective, and difficult to scale across large road networks. "
            "The reviewed literature shows the same motivation: early crack detection reduces maintenance cost, "
            "supports safer roads, and enables more consistent infrastructure monitoring [1], [4]."
        )
    )
    story.append(
        para(
            "The delivered system combines a trained YOLO model with a FastAPI backend and a browser interface. "
            "Users can upload an image for immediate detection, upload a video for frame-by-frame inference, "
            "watch live annotated frames during processing, and review a structured detection log after completion."
        )
    )
    story.append(
        para(
            "Three training dataset/model versions were archived. Non-model artifacts are stored in GitHub under "
            "<b>versions/</b>, while all large checkpoints are stored in Hugging Face. Among the evaluated runs, "
            "<b>version-1/run-6</b> was chosen for the website because it provided a strong deployment balance: "
            "YOLOv8m compatibility, moderate file size, high validation performance, and stable behavior in the web workflow."
        )
    )

    story.append(heading("Introduction"))
    story.append(
        para(
            "Pavement cracks are visible indicators of road deterioration. If cracks are detected early, maintenance teams "
            "can intervene before surface damage expands into more expensive structural repair. Earlier traditional systems "
            "used image processing pipelines such as filtering, edge detection, morphology, and heuristic classification [4]. "
            "These approaches can work, but they often need careful preprocessing and can be sensitive to lighting, shadows, "
            "camera angle, and pavement texture."
        )
    )
    story.append(
        para(
            "Modern object detection models provide a more flexible approach. YOLO-style detectors are especially attractive "
            "because they perform localization and classification in a single forward pass. Nie and Wang applied YOLOv3 to "
            "pavement crack detection and reported improved speed and an 88% crack-detection accuracy compared with traditional "
            "methods [1]. More recent work on multi-scene asphalt distress detection shows that improved YOLOv8-based designs "
            "can identify longitudinal cracks, transverse cracks, mesh cracks, and potholes under practical road-scene variation [2]."
        )
    )
    story.append(
        image_flowable(
            ROOT / "static" / "assets" / "pavement-crack-detection.png",
            "Figure 1. Visual project preview showing detected pavement crack boxes and confidence labels.",
            max_height=3.0 * inch,
        )
    )

    story.append(heading("Related Work And References"))
    story.append(
        para(
            "The four supplied references guided the project framing. Cubero-Fernandez et al. emphasize that detection alone "
            "is not enough; the crack type is important because different pavement damage categories require different repair "
            "strategies [4]. This directly supports the class design used in the website: alligator crack, longitudinal crack, "
            "pothole, and transverse crack."
        )
    )
    story.append(
        para(
            "Nie and Wang demonstrate how YOLOv3 can be adapted to pavement crack detection through manual annotation, training, "
            "and model verification [1]. Zhang et al. extend this direction by modifying YOLOv8s into SMG-YOLOv8, highlighting "
            "the importance of multi-scale features and attention mechanisms for asphalt distress recognition across multiple "
            "scenes [2]. Antariksa et al. compare YOLOv8 variants in a transportation context and show why model selection should "
            "consider both accuracy and computational efficiency rather than choosing only the largest model [3]."
        )
    )

    story.append(heading("YOLO Algorithm Background"))
    story.append(
        para(
            "YOLO, short for You Only Look Once, is a family of one-stage object detectors. Instead of using a separate proposal "
            "stage, a YOLO model predicts object locations and classes directly from image features. This is useful for pavement "
            "inspection because road images and videos may need near-real-time processing."
        )
    )
    yolo_table = [
        ["Algorithm family", "Key idea", "Relevance to this project"],
        ["YOLOv3", "Anchor-based, multi-scale detection with strong real-time performance.", "Important baseline in pavement crack literature [1]."],
        ["YOLOv5", "PyTorch-oriented training workflow and practical model-size variants.", "Useful reference point for efficient deployment models."],
        ["YOLOv6", "Efficiency-oriented detector family with deployment-focused design.", "Represents the speed/accuracy trade-off considered in model comparisons."],
        ["YOLOv8", "Anchor-free split head and modern Ultralytics training/deployment workflow [5].", "The selected website model uses a YOLOv8m checkpoint."],
        ["YOLO11", "Ultralytics generation designed for detection and other vision tasks [6].", "Several project runs tested YOLO11 variants."],
        ["YOLO26", "Experimental/training checkpoint naming used in this project archive.", "Evaluated empirically against YOLO11 and YOLOv8 runs."],
    ]
    story.append(styled_table(yolo_table, [1.25 * inch, 2.6 * inch, 2.55 * inch], font_size=7.4))
    story.append(Spacer(1, 8))

    story.append(heading("Project Process And Methodology"))
    story.append(workflow_drawing())
    story.append(Paragraph("Figure 2. Main web-application workflow.", STYLES["Caption"]))
    story.append(
        para(
            "The project workflow began with multiple dataset/training versions, then moved into model evaluation and deployment. "
            "Training artifacts were preserved so model decisions can be audited. The application layer was built separately from "
            "the model storage layer: GitHub contains source code and training artifacts, while Hugging Face stores the weights."
        )
    )
    for item in [
        "Dataset/version folders were reviewed and copied into a GitHub-safe archive without `.pt` files.",
        "All versioned `best.pt` and `last.pt` checkpoints were uploaded to Hugging Face under matching version paths.",
        "The backend loads `model/best.pt` with Ultralytics YOLO and exposes image and video detection endpoints.",
        "The video workflow runs inference in a background job, streams live frames, saves the processed video, and returns a detection log.",
    ]:
        story.append(bullet(item))

    story.append(heading("Dataset And Version Archive"))
    version_table = [
        ["Version", "Runs", "GitHub content", "Hugging Face content"],
        ["version-1", "6", "Training plots, CSV metrics, configs, validation images, version screenshot.", "12 checkpoints: best.pt and last.pt for runs 1-6."],
        ["version-2", "1", "Training plots, CSV metrics, configs, validation images, version screenshot.", "2 checkpoints: best.pt and last.pt for run 1."],
        ["version-3", "1", "Training plots, CSV metrics, configs, validation images, version screenshot.", "2 checkpoints: best.pt and last.pt for run 1."],
    ]
    story.append(styled_table(version_table, [0.85 * inch, 0.55 * inch, 3.0 * inch, 2.1 * inch], font_size=7.4))
    story.append(Spacer(1, 8))
    story.append(
        para(
            "This split avoids storing large binary files in GitHub while still preserving the complete experimental context. "
            "The version folders in GitHub can be inspected for training curves, confusion matrices, and validation predictions. "
            "The checkpoints can be downloaded from Hugging Face when they are needed for reproduction or deployment."
        )
    )

    story.append(heading("Model Comparison"))
    story.append(
        para(
            "The following table compares the final logged validation metrics for the eight training runs. The highlighted row "
            "is the model selected for website deployment. Values come from each run's `results.csv`; checkpoint sizes refer to "
            "the corresponding `best.pt` files."
        )
    )
    rows = [["Version", "Run", "Base model", "Size MB", "mAP50", "mAP50-95", "Precision", "Recall"]]
    for row in MODEL_ROWS:
        version, run, base_model, size_mb, map50, map5095, precision, recall = row
        rows.append([version, run, base_model, f"{size_mb:.2f}", f"{map50:.4f}", f"{map5095:.4f}", f"{precision:.4f}", f"{recall:.4f}"])
    story.append(styled_table(rows, [0.78 * inch, 0.55 * inch, 1.02 * inch, 0.7 * inch, 0.65 * inch, 0.75 * inch, 0.75 * inch, 0.65 * inch], font_size=6.9, highlight_row=6))
    story.append(Paragraph("Table 1. Final-epoch model comparison. Highlighted row: version-1/run-6.", STYLES["Caption"]))
    story.append(model_bar_chart())
    story.append(Paragraph("Figure 3. mAP comparison across training runs.", STYLES["Caption"]))

    story.append(heading("Why Version-1 Run-6 Was Selected", 2))
    story.append(
        para(
            "Version-1/run-6 was not selected only because of a single metric. Run-1 has the highest mAP50 in the table, but the "
            "final website model needed a practical balance between accuracy, model maturity, runtime compatibility, file size, "
            "and stable behavior in image/video inference. Run-6 uses YOLOv8m, a widely supported Ultralytics detector family, "
            "and achieved mAP50 = 0.8844 and mAP50-95 = 0.5164 while keeping the checkpoint near 52 MB. This is much lighter than "
            "the X-size runs and still close to the strongest validation results."
        )
    )
    for item in [
        "Strong accuracy: mAP50 remained above 0.88, with competitive mAP50-95 among version-1 runs.",
        "Deployment balance: the 52 MB checkpoint is practical for a FastAPI website and video processing workflow.",
        "Runtime stability: YOLOv8m integrates cleanly with the Ultralytics package used by the backend.",
        "Visual performance: the run produced usable validation predictions and stable annotated outputs during web testing.",
        "Operational fit: the model supports the four deployed road-distress labels used by the website.",
    ]:
        story.append(bullet(item))

    story.append(heading("Selected Training Figures"))
    story.append(
        para(
            "The following figures come from `versions/version-1/run-6/detect/train`. They document the selected model's training "
            "behavior and validation performance."
        )
    )
    selected_dir = ROOT / "versions" / "version-1" / "run-6" / "detect" / "train"
    story.append(image_flowable(selected_dir / "results.png", "Figure 4. Training and validation curves for version-1/run-6.", max_height=3.25 * inch))
    story.append(image_flowable(selected_dir / "confusion_matrix_normalized.png", "Figure 5. Normalized confusion matrix for version-1/run-6.", max_height=3.45 * inch))
    story.append(image_flowable(selected_dir / "BoxPR_curve.png", "Figure 6. Precision-recall curve for version-1/run-6.", max_height=3.45 * inch))
    story.append(image_flowable(selected_dir / "val_batch0_pred.jpg", "Figure 7. Example validation predictions for version-1/run-6.", max_height=3.45 * inch))

    story.append(heading("Website And Backend Implementation"))
    story.append(
        para(
            "The website is implemented as static HTML, CSS, and JavaScript served by FastAPI. The backend exposes routes for the "
            "home page, image detection, video detection, job status, and MJPEG frame streaming. For image uploads, the API decodes "
            "the file with OpenCV, runs YOLO prediction, returns a base64 annotated image, and sends a detection table to the UI."
        )
    )
    story.append(
        para(
            "For video uploads, a background thread processes frames and writes an annotated MP4. The frontend polls the job status "
            "and displays the live stream, progress count, total detections, output video, and a detailed detection log. This design "
            "prevents long videos from blocking the browser request."
        )
    )
    api_table = [
        ["Endpoint", "Method", "Purpose"],
        ["/health", "GET", "Checks backend status and model availability."],
        ["/api/detect/image", "POST", "Runs detection on a single uploaded image."],
        ["/api/detect/video", "POST", "Starts a background video detection job."],
        ["/api/detect/video/{job_id}", "GET", "Returns progress, output URL, and detection events."],
        ["/api/detect/video/{job_id}/stream", "GET", "Streams live annotated frames."],
    ]
    story.append(styled_table(api_table, [1.9 * inch, 0.65 * inch, 3.95 * inch], font_size=7.4))
    story.append(Spacer(1, 8))

    story.append(heading("Conclusion And Future Work"))
    story.append(
        para(
            "The project demonstrates a complete path from pavement-crack model training to a usable web interface. The report "
            "archive preserves evidence of multiple training versions, the model-comparison table explains the deployment choice, "
            "and the website provides practical image and video detection workflows. The separation of GitHub and Hugging Face "
            "keeps the repository readable while preserving all checkpoint files for reproduction."
        )
    )
    story.append(
        para(
            "Future work can improve the system by adding a hosted backend separate from GitHub Pages, adding severity grading for "
            "detected cracks, exporting detection logs as CSV, evaluating inference speed per model version, and expanding the "
            "dataset with more weather, lighting, camera-angle, and pavement-surface variation."
        )
    )

    story.append(heading("References"))
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
        title="Pavement Crack Detection V5 Final Project Report",
        author="Hamidreza Khalaj Zahraei",
    )
    story = build_story()
    doc.multiBuild(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
