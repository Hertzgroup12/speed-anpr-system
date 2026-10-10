"""Calendar-period speed-case review PDF reports."""

from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any, Literal
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

from app.time_utils import GMT, gmt_now_iso

ReportPeriod = Literal["weekly", "monthly", "yearly"]


def calendar_period_bounds(
    period: ReportPeriod,
    today: date | None = None,
) -> tuple[date, date]:
    """Return the current calendar period's inclusive start and exclusive end."""
    current_date = today or datetime.now(GMT).date()
    if period == "weekly":
        start = current_date - timedelta(days=current_date.weekday())
        return start, start + timedelta(days=7)
    if period == "monthly":
        start = current_date.replace(day=1)
        if start.month == 12:
            end = date(start.year + 1, 1, 1)
        else:
            end = date(start.year, start.month + 1, 1)
        return start, end
    if period == "yearly":
        start = date(current_date.year, 1, 1)
        return start, date(current_date.year + 1, 1, 1)
    raise ValueError("Report period must be weekly, monthly, or yearly")


def _detected_date(case: dict[str, Any]) -> date | None:
    timestamp = case.get("detected_at")
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=GMT)
    return parsed.astimezone(GMT).date()


def cases_in_calendar_period(
    cases: list[dict[str, Any]],
    period: ReportPeriod,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], date, date]:
    """Select cases detected in the current GMT calendar period."""
    start, end = calendar_period_bounds(period, today)
    selected = [
        case
        for case in cases
        if (detected := _detected_date(case)) is not None
        and start <= detected < end
    ]
    return selected, start, end


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(value)), style)


def build_case_report_pdf(
    cases: list[dict[str, Any]],
    period: ReportPeriod,
    start: date,
    end: date,
) -> bytes:
    """Render a readable, paginated PDF summary of cases in one calendar period."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"Speed case review - {period}",
        author="JNRD PRO",
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#52667a"),
        spaceAfter=10,
    )
    cell_style = ParagraphStyle(
        "ReportCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9,
        spaceAfter=0,
    )
    header_style = ParagraphStyle(
        "ReportHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    status_counts = {
        status: sum(case.get("review_status") == status for case in cases)
        for status in ("pending_review", "reviewed", "dismissed")
    }
    period_end = end - timedelta(days=1)
    period_title = {
        "weekly": "Weekly",
        "monthly": "Monthly",
        "yearly": "Yearly",
    }[period]
    summary = (
        f"Cases detected: {len(cases)}"
        f" &nbsp;&bull;&nbsp; Pending review: {status_counts['pending_review']}"
        f" &nbsp;&bull;&nbsp; Reviewed: {status_counts['reviewed']}"
        f" &nbsp;&bull;&nbsp; Dismissed: {status_counts['dismissed']}"
    )
    story = [
        Paragraph(f"{period_title} Speed Case Review", title_style),
        Paragraph(
            f"Detection dates: {start.isoformat()} through {period_end.isoformat()} (GMT)"
            f" &nbsp;&bull;&nbsp; Generated: {escape(gmt_now_iso())}",
            subtitle_style,
        ),
        Paragraph(summary, styles["Normal"]),
        Spacer(1, 8),
    ]

    headings = [
        "Detected (GMT)",
        "Case ID",
        "Plate",
        "Speed / limit",
        "Location",
        "Review",
        "Notification",
    ]
    data: list[list[Any]] = [
        [_paragraph(heading, header_style) for heading in headings]
    ]
    for case in cases:
        detected_at = str(case.get("detected_at", ""))
        if detected_at:
            try:
                timestamp = datetime.fromisoformat(
                    detected_at.replace("Z", "+00:00")
                )
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=GMT)
                detected_at = timestamp.astimezone(GMT).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass
        data.append(
            [
                _paragraph(detected_at or "Unknown", cell_style),
                _paragraph(case.get("case_id", ""), cell_style),
                _paragraph(case.get("plate_number") or "Not recognized", cell_style),
                _paragraph(
                    f"{case.get('speed_kmh', '—')} / "
                    f"{case.get('speed_limit_kmh', '—')} km/h",
                    cell_style,
                ),
                _paragraph(case.get("location") or "Unknown", cell_style),
                _paragraph(
                    str(case.get("review_status", "unknown")).replace("_", " "),
                    cell_style,
                ),
                _paragraph(
                    str(case.get("notification_status", "not_sent")).replace("_", " "),
                    cell_style,
                ),
            ]
        )
    if not cases:
        data.append(
            [
                _paragraph("No cases were detected during this period.", cell_style),
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )

    table = LongTable(
        data,
        colWidths=[31 * mm, 31 * mm, 26 * mm, 28 * mm, 72 * mm, 27 * mm, 30 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102a43")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#f1f5f9"),
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    document.build(story)
    return buffer.getvalue()
