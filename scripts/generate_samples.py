"""Regenerate seven wholly fictional sample reports and the 40-question evaluation set."""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "sample_docs"
EVAL = ROOT / "data" / "eval"
DOCS.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

REPORTS = {
    "discharge_dengue.pdf": {
        "title": "FICTIONAL DISCHARGE SUMMARY", "patient": "Demo Patient A", "id": "FAKE-DS-001",
        "sections": {
            "Diagnosis": ["Dengue fever."],
            "Investigations": ["Platelet count: 145,000 /uL on 04 April 2026.", "Hematocrit: 39% on 04 April 2026."],
            "Medications": ["Paracetamol 500 mg orally every 8 hours for 3 days."],
            "Follow-up": ["Review in medical clinic on 07 April 2026."],
            "Warning signs": ["Return to hospital for bleeding, persistent vomiting, or severe abdominal pain."],
            "Diet and activity": ["Drink fluids as advised. Rest at home for 3 days."],
        },
    },
    "discharge_diabetes_hypertension.pdf": {
        "title": "FICTIONAL DISCHARGE SUMMARY", "patient": "Demo Patient B", "id": "FAKE-DS-002",
        "sections": {
            "Diagnosis": ["Type 2 diabetes mellitus and hypertension."],
            "Investigations": ["Fasting blood glucose: 156 mg/dL on 08 May 2026.", "Blood pressure: 146/90 mmHg at discharge."],
            "Medications": ["Metformin 500 mg orally twice daily with food for 30 days.", "Amlodipine 5 mg orally once daily for 30 days."],
            "Follow-up": ["Review at clinic on 22 May 2026 with a fasting glucose result."],
            "Advice": ["Follow the diet plan given by the clinic. Walk for 20 minutes daily if comfortable."],
            "Warning signs": ["Return for fainting or severe weakness."],
        },
    },
    "discharge_appendectomy.pdf": {
        "title": "FICTIONAL DISCHARGE SUMMARY", "patient": "Demo Patient C", "id": "FAKE-DS-003",
        "sections": {
            "Diagnosis": ["Acute appendicitis; laparoscopic appendectomy performed on 10 June 2026."],
            "Medications": ["Paracetamol 650 mg orally every 6 hours as needed for pain for 5 days."],
            "Follow-up": ["Surgical clinic review on 17 June 2026 for wound check."],
            "Advice": ["Keep wound clean and dry. Avoid heavy lifting for 2 weeks."],
            "Warning signs": ["Return for fever, increasing wound redness, or discharge from wound."],
        },
    },
    "prescription_asthma.pdf": {
        "title": "FICTIONAL PRESCRIPTION", "patient": "Demo Patient D", "id": "FAKE-RX-004",
        "sections": {
            "Diagnosis": ["Asthma."],
            "Medications": ["Salbutamol inhaler 100 mcg: 2 puffs when needed for wheeze, up to 4 times daily for 14 days.",
                            "Budesonide inhaler 200 mcg: 1 puff twice daily for 30 days."],
            "Follow-up": ["Review at respiratory clinic on 01 July 2026."],
            "Warning signs": ["Seek urgent care for worsening breathlessness."],
        },
    },
    "prescription_uti_scanned.pdf": {
        "title": "FICTIONAL PRESCRIPTION - SCANNED STYLE", "patient": "Demo Patient E", "id": "FAKE-RX-005",
        "sections": {
            "Diagnosis": ["Urinary tract infection."],
            "Medications": ["Nitrofurantoin 100 mg orally twice daily for 5 days.", "Paracetamol 500 mg orally when needed for pain for 2 days."],
            "Follow-up": ["Review on 15 July 2026 if symptoms persist."],
            "Advice": ["Drink water regularly."],
        },
    },
    "lab_blood_count.pdf": {
        "title": "FICTIONAL LAB REPORT", "patient": "Demo Patient F", "id": "FAKE-LAB-006",
        "sections": {
            "Tests and results": ["Hemoglobin: 12.4 g/dL.", "White blood cells: 7,200 /uL.",
                                  "Platelets: 210,000 /uL."],
            "Follow-up": ["Discuss results with ordering doctor on 20 August 2026."],
        },
        "table": [["Test", "Result", "Unit"], ["Hemoglobin", "12.4", "g/dL"],
                  ["White blood cells", "7,200", "/uL"], ["Platelets", "210,000", "/uL"]],
    },
    "lab_metabolic.pdf": {
        "title": "FICTIONAL LAB REPORT", "patient": "Demo Patient G", "id": "FAKE-LAB-007",
        "sections": {
            "Tests and results": ["Creatinine: 0.9 mg/dL on 03 September 2026.",
                                  "Sodium: 139 mmol/L.", "Potassium: 4.1 mmol/L."],
            "Follow-up": ["Show this report to the ordering doctor on 10 September 2026."],
        },
    },
}


def create_pdf(filename: str, report: dict) -> None:
    path = DOCS / filename
    styles = getSampleStyleSheet()
    if "scanned" in filename:
        # A raster-only page forces the vision/OCR path; modest blur/noise imitates a phone scan.
        image = Image.new("RGB", (1240, 1754), "#f3f1e9")
        draw = ImageDraw.Draw(image)
        windows_font = Path("C:/Windows/Fonts/arial.ttf")
        if windows_font.exists():
            font = ImageFont.truetype(str(windows_font), 25)
        else:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 25)
            except OSError:
                font = ImageFont.load_default()
        y = 90
        for line in [report["title"], f"Patient: {report['patient']}   ID: {report['id']}",
                     "DEMONSTRATION ONLY - NOT A REAL MEDICAL RECORD"]:
            draw.text((95, y), line, font=font, fill="#202020")
            y += 65
        for heading, values in report["sections"].items():
            y += 28
            draw.text((95, y), heading, font=font, fill="#101010")
            y += 48
            for value in values:
                draw.text((105, y), value, font=font, fill="#202020")
                y += 52
        random.seed(5)
        for _ in range(1600):
            x, yy = random.randrange(1240), random.randrange(1754)
            draw.point((x, yy), fill="#d5d3ca")
        image.save(path, "PDF", resolution=150)
        return
    story = [Paragraph(report["title"], styles["Title"]), Spacer(1, 10),
             Paragraph(f"Patient: {report['patient']} | ID: {report['id']}", styles["Normal"]),
             Paragraph("DEMONSTRATION ONLY - NOT A REAL MEDICAL RECORD", styles["Normal"]), Spacer(1, 16)]
    for heading, values in report["sections"].items():
        story.append(Paragraph(heading, styles["Heading2"]))
        for value in values:
            story.append(Paragraph(value, styles["Normal"]))
            story.append(Spacer(1, 7))
        if heading == "Tests and results" and report.get("table"):
            table = Table(report["table"], colWidths=[180, 100, 90])
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                       ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                       ("PADDING", (0, 0), (-1, -1), 7)]))
            story.append(table)
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


# 20 English, 20 Malayalam; eight intentionally unanswerable.
QUESTIONS = [
    ("discharge_dengue.pdf", "What is the diagnosis?", "Dengue fever", "Diagnosis", True),
    ("discharge_dengue.pdf", "What was the platelet count?", "145,000 /uL", "Investigations", True),
    ("discharge_dengue.pdf", "How often is Paracetamol taken?", "every 8 hours", "Medications", True),
    ("discharge_dengue.pdf", "When is the clinic review?", "07 April 2026", "Follow-up", True),
    ("discharge_dengue.pdf", "What is the patient's blood group?", "", "", False),
    ("discharge_diabetes_hypertension.pdf", "What diagnoses are listed?", "Type 2 diabetes mellitus and hypertension", "Diagnosis", True),
    ("discharge_diabetes_hypertension.pdf", "What is the Metformin dose?", "500 mg twice daily", "Medications", True),
    ("discharge_diabetes_hypertension.pdf", "What was fasting glucose?", "156 mg/dL", "Investigations", True),
    ("discharge_diabetes_hypertension.pdf", "How long is Amlodipine prescribed?", "30 days", "Medications", True),
    ("discharge_diabetes_hypertension.pdf", "Should insulin be started?", "", "", False),
    ("discharge_appendectomy.pdf", "When was the operation?", "10 June 2026", "Diagnosis", True),
    ("discharge_appendectomy.pdf", "What lifting restriction is listed?", "Avoid heavy lifting for 2 weeks", "Advice", True),
    ("discharge_appendectomy.pdf", "What are the wound warning signs?", "fever, increasing wound redness, or discharge", "Warning signs", True),
    ("discharge_appendectomy.pdf", "When is the wound check?", "17 June 2026", "Follow-up", True),
    ("discharge_appendectomy.pdf", "What type of sutures were used?", "", "", False),
    ("prescription_asthma.pdf", "What is the Salbutamol strength?", "100 mcg", "Medications", True),
    ("prescription_asthma.pdf", "How often is Budesonide taken?", "twice daily", "Medications", True),
    ("prescription_asthma.pdf", "When is the respiratory review?", "01 July 2026", "Follow-up", True),
    ("lab_blood_count.pdf", "What is the hemoglobin result?", "12.4 g/dL", "Tests and results", True),
    ("lab_metabolic.pdf", "What is the potassium result?", "4.1 mmol/L", "Tests and results", True),
    ("discharge_dengue.pdf", "രോഗനിർണയം എന്താണ്?", "Dengue fever", "Diagnosis", True),
    ("discharge_dengue.pdf", "ഏത് ലക്ഷണങ്ങൾ വന്നാൽ ആശുപത്രിയിലേക്ക് മടങ്ങണം?", "bleeding, persistent vomiting, or severe abdominal pain", "Warning signs", True),
    ("discharge_dengue.pdf", "എത്ര ദിവസം വിശ്രമിക്കണം?", "3 days", "Diet and activity", True),
    ("discharge_dengue.pdf", "ആന്റിബയോട്ടിക് നൽകിയിട്ടുണ്ടോ?", "", "", False),
    ("discharge_diabetes_hypertension.pdf", "Amlodipine എത്ര mg ആണ്?", "5 mg", "Medications", True),
    ("discharge_diabetes_hypertension.pdf", "Blood pressure എത്രയായിരുന്നു?", "146/90 mmHg", "Investigations", True),
    ("discharge_diabetes_hypertension.pdf", "ദിവസവും എത്ര നേരം നടക്കണം?", "20 minutes", "Advice", True),
    ("discharge_diabetes_hypertension.pdf", "എന്റെ HbA1c എത്രയാണ്?", "", "", False),
    ("discharge_appendectomy.pdf", "മുറിവ് എങ്ങനെ സൂക്ഷിക്കണം?", "Keep wound clean and dry", "Advice", True),
    ("discharge_appendectomy.pdf", "Histopathology ഫലം എന്താണ്?", "", "", False),
    ("discharge_appendectomy.pdf", "വീണ്ടും എപ്പോൾ കാണിക്കണം?", "17 June 2026", "Follow-up", True),
    ("prescription_asthma.pdf", "Budesonide എത്ര ദിവസം ഉപയോഗിക്കണം?", "30 days", "Medications", True),
    ("prescription_asthma.pdf", "റിപ്പോർട്ടിലെ warning sign എന്താണ്?", "Seek urgent care for worsening breathlessness", "Warning signs", True),
    ("prescription_uti_scanned.pdf", "Nitrofurantoin എത്ര ദിവസം കഴിക്കണം?", "5 days", "Medications", True),
    ("prescription_uti_scanned.pdf", "എപ്പോൾ വീണ്ടും കാണിക്കണം?", "15 July 2026 if symptoms persist", "Follow-up", True),
    ("prescription_uti_scanned.pdf", "എനിക്ക് പെനിസിലിൻ അലർജി ഉണ്ടോ?", "", "", False),
    ("lab_blood_count.pdf", "Platelets ഫലം എത്ര?", "210,000 /uL", "Tests and results", True),
    ("lab_blood_count.pdf", "ഡോക്ടറെ എപ്പോൾ കാണിക്കണം?", "20 August 2026", "Follow-up", True),
    ("lab_metabolic.pdf", "Creatinine ഫലം എത്ര?", "0.9 mg/dL", "Tests and results", True),
    ("lab_metabolic.pdf", "എന്റെ kidney disease stage എന്താണ്?", "", "", False),
]


def main() -> None:
    for filename, report in REPORTS.items():
        create_pdf(filename, report)
    rows = [{"id": f"Q{n:02d}", "document": doc, "question": question,
             "expected_answer": answer, "expected_source_section": section, "answerable": answerable}
            for n, (doc, question, answer, section, answerable) in enumerate(QUESTIONS, 1)]
    assert len(rows) == 40 and sum(not x["answerable"] for x in rows) == 8
    (EVAL / "questions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated 7 PDFs and 40 evaluation questions")


if __name__ == "__main__":
    main()
