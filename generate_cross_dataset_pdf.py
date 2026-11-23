"""
Generate Publication-Quality PDF Documentation for Cross-Dataset Validation
Using ReportLab
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def build_pdf(filename="CROSS_DATASET_VALIDATION_DOCUMENTATION.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f2d59"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334e68"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f2d59"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#243b53"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#102a43"),
        backColor=colors.HexColor("#f0f4f8"),
        borderColor=colors.HexColor("#627d98"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=10,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#102a43"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    story = []

    # Title & Banner
    story.append(Paragraph("Cross-Dataset Validation & Facility Generalization Report", title_style))
    story.append(Paragraph("Complete Technical Documentation: Motivation, Thermodynamics, Domain Shift & Transfer Benchmark", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f2d59"), spaceAfter=10))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Purpose", h1_style))
    story.append(Paragraph(
        "<b>Project Motto:</b> <i>'Beyond Single-Site Memorization: Validating Zero-Shot Generalization, "
        "Scale Invariance, and Psychrometric Robustness Across Supercomputing Facilities.'</i>", callout_style
    ))
    story.append(Paragraph(
        "This engineering report provides an in-depth breakdown of the cross-dataset validation study conducted "
        "on the Data Center Cooling Power Prediction benchmark. In data center AI modeling, algorithms trained "
        "on a single facility frequently overfit to local mechanical chiller curves and limited historical IT workloads. "
        "To assess true physical generalizability, our pre-trained models were deployed zero-shot onto real-world "
        "operational telemetry from the <b>Collaborative Computing Center (C3) High Performance Computing Data Center "
        "(Building IF-692)</b> at <b>Idaho National Laboratory (INL)</b>.", body_style
    ))
    story.append(Paragraph(
        "<b>Key Study Findings:</b><br/>"
        "• <b>Dataset Identity:</b> C3 dataset comprises <b>48,567 continuous hourly records</b> (July 2020 to January 2026), "
        "with <b>38,469 concurrent hours</b> synchronized with on-site meteorological telemetry (outdoor dry-bulb temperature and humidity).<br/>"
        "• <b>Severe Domain Shift:</b> C3 operates as a scientific supercomputing facility with a 24.58% higher median IT baseline "
        "(1,475.72 kW vs. 1,184.52 kW) and 45.8% lower volatility (CV = 0.2682 vs. 0.5240; KS test D = 0.4610, p &lt; 1e-300).<br/>"
        "• <b>Standard Tree Failure (The Saturation Anomaly):</b> Baseline LightGBM collapsed into an invariant flat line "
        "of 10.40 kW (std = 0.11 kW) across all 38,469 hours (-0.1% economizer savings) due to leaf node boundary saturation.<br/>"
        "• <b>Physics-Guided Ratio Success:</b> Predicting scale-invariant cooling intensity (Cooling/IT) achieved 100% physical validity, "
        "realistic operational variance (mean 13.42 kW, peak 71.27 kW), and captured <b>24.5% economizer energy savings</b> during cold weather.", body_style
    ))

    # Section 2: Why Cross-Dataset Validation?
    story.append(Paragraph("2. Why Cross-Dataset Validation? (Motivation & Physical Context)", h1_style))
    story.append(Paragraph(
        "<b>1. The Single-Facility Overfitting Trap:</b> Machine learning models evaluated strictly on chronological "
        "splits within a single data center inadvertently memorize specific chiller plant staging points, structural building "
        "thermal mass, and pump head curves. Cross-facility testing proves whether the model learned fundamental thermodynamics or local coincidences.", body_style
    ))
    story.append(Paragraph(
        "<b>2. Tree Boundary Extrapolation Breakdown:</b> Gradient boosted decision trees (GBDT) partition feature space using "
        "orthogonal splits (x ≤ θ). For any input exceeding the training maximum (x &gt; θ_max), trees cannot extrapolate linearly "
        "and assign the constant value of the outermost leaf node. When a facility expands or when transferring to a higher-density "
        "HPC center, direct GBDT regression severely underestimates cooling requirements.", body_style
    ))
    story.append(Paragraph(
        "<b>3. The Digital Twin Scale-Invariance Mandate:</b> During facility commissioning or server cluster expansions, historical "
        "cooling telemetry does not exist. A model must predict cooling demand purely from anticipated IT capacity and weather. "
        "Normalizing target power by IT workload guarantees scale invariance.", body_style
    ))

    # Section 3: Dataset Profiles
    story.append(Paragraph("3. Dataset Profiles: Primary Facility vs. External C3 HPC Center", h1_style))
    story.append(Paragraph(
        "<b>Primary Facility:</b> Derived from 'power_file.csv' and 'weather_file.csv' (73,696 hourly records, 2016–2025). "
        "Represents enterprise hybrid workloads with pronounced diurnal peaks and multi-year hardware upgrades.<br/>"
        "<b>External C3 Facility:</b> Sourced from 'c3_datacenter_hourly_power_usage_july2020-jan2026.xlsx' (Building IF-692, INL). "
        "Dedicated supercomputing facility housing large scientific clusters (Sawtooth, Bitterroot). Operates continuous 24/7 MPI compute queues.", body_style
    ))

    # Table 1: Workload Comparison Table
    workload_data = [
        [Paragraph("<b>Metric</b>", table_header_style), Paragraph("<b>Primary Data Center</b>", table_header_style), Paragraph("<b>C3 HPC Center (IF-692)</b>", table_header_style), Paragraph("<b>Domain Shift Significance</b>", table_header_style)],
        [Paragraph("Facility Mission", table_cell_style), Paragraph("Enterprise Hybrid", table_cell_style), Paragraph("High-Performance Computing", table_cell_style), Paragraph("Distinct compute pattern", table_cell_style)],
        [Paragraph("Total Records", table_cell_style), Paragraph("73,696 hours", table_cell_style), Paragraph("48,567 hours", table_cell_style), Paragraph("Multi-year operational scale", table_cell_style)],
        [Paragraph("Time Horizon", table_cell_style), Paragraph("2016-06-12 to 2025-08-29", table_cell_style), Paragraph("2020-07-18 to 2026-01-31", table_cell_style), Paragraph("5.5-year overlapping era", table_cell_style)],
        [Paragraph("Mean Compute Power", table_cell_style), Paragraph("1,497.02 kW", table_cell_style), Paragraph("1,586.12 kW", table_cell_style), Paragraph("+5.95% higher average", table_cell_style)],
        [Paragraph("Median Compute Power", table_cell_style), Paragraph("1,184.52 kW", table_cell_style), Paragraph("1,475.72 kW", table_cell_style), Paragraph("+24.58% higher baseline", table_cell_style)],
        [Paragraph("Standard Deviation", table_cell_style), Paragraph("784.50 kW", table_cell_style), Paragraph("425.42 kW", table_cell_style), Paragraph("45.8% lower volatility", table_cell_style)],
        [Paragraph("Interquartile Range (IQR)", table_cell_style), Paragraph("887.11 kW", table_cell_style), Paragraph("313.96 kW", table_cell_style), Paragraph("Concentrated compute band", table_cell_style)],
        [Paragraph("Maximum Compute Power", table_cell_style), Paragraph("3,896.23 kW", table_cell_style), Paragraph("3,927.84 kW", table_cell_style), Paragraph("Identical peak envelope", table_cell_style)],
        [Paragraph("Peak-to-Average (PAR)", table_cell_style), Paragraph("2.60", table_cell_style), Paragraph("2.48", table_cell_style), Paragraph("Consistently saturated load", table_cell_style)],
        [Paragraph("Coeff. of Variation (CV)", table_cell_style), Paragraph("0.5240", table_cell_style), Paragraph("0.2682", table_cell_style), Paragraph("HPC runs continuous jobs", table_cell_style)],
        [Paragraph("Kolmogorov-Smirnov Test", table_cell_style), Paragraph("0.0000 (Reference)", table_cell_style), Paragraph("D = 0.4610 (p < 1e-300)", table_cell_style), Paragraph("Severe shift confirmed", table_cell_style)],
        [Paragraph("Wasserstein Distance", table_cell_style), Paragraph("0.00 kW", table_cell_style), Paragraph("360.30 kW", table_cell_style), Paragraph("Pronounced ECDF divergence", table_cell_style)],
    ]

    t1 = Table(workload_data, colWidths=[130, 110, 120, 170])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2d59")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd2d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # Page Break for Visuals & Model Results
    story.append(PageBreak())

    # Section 4: Visual Diagnostics (Workload Distributions)
    story.append(Paragraph("4. Workload Distribution & Temporal Dynamics Analysis", h1_style))
    story.append(Paragraph(
        "Figure 1 illustrates the comparative workload distributions, boxplot outlier dispersions, and empirical CDFs. "
        "Figure 2 contrasts the diurnal (24-hour) and weekly profiles between the enterprise facility and the C3 HPC center.", body_style
    ))

    fig1_path = "results/figures/cross_dataset_workload_distribution.png"
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=530, height=147))
        story.append(Spacer(1, 8))

    fig2_path = "results/figures/cross_dataset_temporal_profiles.png"
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=530, height=176))
        story.append(Spacer(1, 10))

    # Section 5: Model Transfer Results Table
    story.append(Paragraph("5. Cross-Facility Model Transfer Benchmark (38,469 Hours)", h1_style))
    story.append(Paragraph(
        "Zero-shot transfer performance of pre-trained models deployed across all synchronized operational cycles of C3:", body_style
    ))

    model_data = [
        [Paragraph("<b>Model Architecture</b>", table_header_style), Paragraph("<b>Pred Mean</b>", table_header_style), Paragraph("<b>Pred Std</b>", table_header_style), Paragraph("<b>Implied Ratio</b>", table_header_style), Paragraph("<b>Implied PUE</b>", table_header_style), Paragraph("<b>Free-Cool PUE</b>", table_header_style), Paragraph("<b>Chiller PUE</b>", table_header_style), Paragraph("<b>Economizer Savings</b>", table_header_style), Paragraph("<b>Validity</b>", table_header_style)],
        [Paragraph("<b>Physics-Guided Ratio (Ours)</b>", table_cell_style), Paragraph("13.42 kW", table_cell_style), Paragraph("5.68 kW", table_cell_style), Paragraph("0.0089", table_cell_style), Paragraph("1.0089", table_cell_style), Paragraph("1.0085", table_cell_style), Paragraph("1.0103", table_cell_style), Paragraph("<b>24.5%</b>", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Dynamic Autoregressive GBDT", table_cell_style), Paragraph("4.38 kW", table_cell_style), Paragraph("0.98 kW", table_cell_style), Paragraph("0.0030", table_cell_style), Paragraph("1.0030", table_cell_style), Paragraph("1.0029", table_cell_style), Paragraph("1.0033", table_cell_style), Paragraph("19.9%", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Baseline XGBoost", table_cell_style), Paragraph("12.67 kW", table_cell_style), Paragraph("4.65 kW", table_cell_style), Paragraph("0.0086", table_cell_style), Paragraph("1.0086", table_cell_style), Paragraph("1.0085", table_cell_style), Paragraph("1.0092", table_cell_style), Paragraph("16.1%", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("<b>Baseline LightGBM</b>", table_cell_style), Paragraph("10.40 kW", table_cell_style), Paragraph("<b>0.11 kW</b>", table_cell_style), Paragraph("0.0075", table_cell_style), Paragraph("1.0075", table_cell_style), Paragraph("1.0077", table_cell_style), Paragraph("1.0068", table_cell_style), Paragraph("<b>-0.1% (Failed)</b>", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Baseline Random Forest", table_cell_style), Paragraph("12.57 kW", table_cell_style), Paragraph("4.60 kW", table_cell_style), Paragraph("0.0086", table_cell_style), Paragraph("1.0086", table_cell_style), Paragraph("1.0084", table_cell_style), Paragraph("1.0091", table_cell_style), Paragraph("15.0%", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Baseline Time Features (RF)", table_cell_style), Paragraph("12.53 kW", table_cell_style), Paragraph("4.68 kW", table_cell_style), Paragraph("0.0085", table_cell_style), Paragraph("1.0085", table_cell_style), Paragraph("1.0084", table_cell_style), Paragraph("1.0090", table_cell_style), Paragraph("14.6%", table_cell_style), Paragraph("100.0%", table_cell_style)],
    ]

    t2 = Table(model_data, colWidths=[140, 48, 48, 52, 52, 54, 54, 48, 36])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2d59")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd2d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # Page Break for Diagnostic Visuals
    story.append(PageBreak())

    # Section 6: Visual Diagnostics (Time Series & PUE)
    story.append(Paragraph("6. Operational Diagnostics: Cooling Predictions & Economizer Transitions", h1_style))
    story.append(Paragraph(
        "Figure 3 demonstrates time-series operational tracking across seasonal transitions, while Figure 4 shows "
        "the implied PUE distributions and psychrometric economizer transitions at the 55°F wet-bulb threshold.", body_style
    ))

    fig3_path = "results/figures/cross_dataset_cooling_predictions.png"
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=530, height=220))
        story.append(Spacer(1, 8))

    fig4_path = "results/figures/cross_dataset_implied_pue.png"
    if os.path.exists(fig4_path):
        story.append(Image(fig4_path, width=530, height=165))
        story.append(Spacer(1, 10))

    # Section 7: Key Scientific Insights & Conclusion
    story.append(Paragraph("7. Scientific Insights & Engineering Recommendations", h1_style))
    story.append(Paragraph(
        "<b>1. Tree Model Saturation Resolved by Physics Guidance:</b> Direct tree regression breaks down under domain shift "
        "because decision trees clip outputs on unseen out-of-distribution feature spaces. Normalizing cooling by compute load "
        "(Cooling Ratio) enforces physical scaling across arbitrary data center capacities.<br/>"
        "<b>2. Psychrometrics are Mandatory:</b> Dry-bulb temperature alone is insufficient to predict evaporative chiller efficiency. "
        "Wet-bulb temperature (Stull formulation) and moist air enthalpy accurately indicate waterside economizer free-cooling bypass.<br/>"
        "<b>3. Production Deployment Strategy:</b> For digital twins and server expansion studies without cooling meters, "
        "the Physics-Guided Ratio model should be deployed as the primary surrogate.", body_style
    ))
    story.append(Spacer(1, 6))

    # Artifact Inventory Box
    story.append(Paragraph("8. Study Artifacts & Deliverables", h2_style))
    story.append(Paragraph(
        "• <b>Pipeline Script:</b> cross_dataset_validation.py<br/>"
        "• <b>Clean Data:</b> results/data/c3_cleaned_hourly_power.csv | results/data/c3_merged_with_weather.csv<br/>"
        "• <b>Tables:</b> cross_dataset_workload_comparison.csv | cross_dataset_model_predictions_summary.csv<br/>"
        "• <b>Figures:</b> cross_dataset_workload_distribution.png | cross_dataset_temporal_profiles.png | cross_dataset_cooling_predictions.png | cross_dataset_implied_pue.png | cross_dataset_model_agreement.png<br/>"
        "• <b>Technical Reports:</b> results/reports/cross_dataset_validation_report.txt | CROSS_DATASET_VALIDATION_DOCUMENTATION.md",
        callout_style
    ))

    doc.build(story)
    print(f"Publication-quality PDF generated successfully: {filename}")


if __name__ == "__main__":
    build_pdf("CROSS_DATASET_VALIDATION_DOCUMENTATION.pdf")
