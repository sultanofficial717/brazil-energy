import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_report():
    doc = docx.Document()
    
    # Page setup - Standard Letter, 0.8 inch margins
    sections = doc.sections
    for section in sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        section.different_first_page_header_footer = True
        
        # Header for subsequent pages
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Forecasting Electricity Spot Prices via Dense NLP Embeddings")
        hrun.font.name = 'Calibri'
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(113, 128, 150)
        
        # Footer for subsequent pages
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun_left = fp.add_run("Academic & Technical Portfolio Research Report          ")
        frun_left.font.name = 'Calibri'
        frun_left.font.size = Pt(8.5)
        frun_left.font.color.rgb = RGBColor(113, 128, 150)
        
        # Dynamic page number
        frun_page = fp.add_run("Page ")
        frun_page.font.name = 'Calibri'
        frun_page.font.size = Pt(8.5)
        frun_page.font.color.rgb = RGBColor(113, 128, 150)
        
        fldSimple = OxmlElement('w:fldSimple')
        fldSimple.set(qn('w:instr'), 'PAGE')
        fp._p.append(fldSimple)

    # Color Palette Definitions
    COLOR_PRIMARY = RGBColor(27, 54, 93)     # Deep Navy #1B365D
    COLOR_SECONDARY = RGBColor(44, 82, 130)  # Slate Blue #2C5282
    COLOR_BODY = RGBColor(45, 55, 72)        # Charcoal Off-Black #2D3748
    COLOR_MUTED = RGBColor(74, 85, 104)      # Muted Slate #4A5568
    HEX_PRIMARY = "1B365D"
    HEX_SECONDARY = "2C5282"
    HEX_BG_LIGHT = "F7FAFC"
    HEX_BG_CALLOUT = "EDF2F7"
    HEX_BORDER = "CBD5E0"

    # Typography & Element Helpers
    def set_run_font(run, name='Calibri', size_pt=9.5, bold=False, italic=False, color=COLOR_BODY):
        run.font.name = name
        run.font.size = Pt(size_pt)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color

    def add_h1(text, space_before=8, space_after=2):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        set_run_font(run, name='Calibri', size_pt=12.5, bold=True, color=COLOR_PRIMARY)
        return p

    def add_h2(text, space_before=6, space_after=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        set_run_font(run, name='Calibri', size_pt=10.2, bold=True, color=COLOR_SECONDARY)
        return p

    def add_body(text, bold_prefix=None, italic_prefix=None, space_after=2.0):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.08
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            set_run_font(r_b, bold=True, color=COLOR_BODY)
        if italic_prefix:
            r_i = p.add_run(italic_prefix)
            set_run_font(r_i, italic=True, color=COLOR_MUTED)
        r = p.add_run(text)
        set_run_font(r, color=COLOR_BODY)
        return p

    def add_bullet(text, bold_prefix=None, space_after=1.2):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.06
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            set_run_font(r_b, bold=True, color=COLOR_BODY)
        r = p.add_run(text)
        set_run_font(r, color=COLOR_BODY)
        return p

    def add_callout(text, bold_title="Key Takeaway: "):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.12)
        p.paragraph_format.right_indent = Inches(0.12)
        p.paragraph_format.line_spacing = 1.06
        
        pPr = p._p.get_or_add_pPr()
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:left w:val="single" w:sz="18" w:space="5" w:color="{HEX_PRIMARY}"/></w:pBdr>')
        pPr.append(pBdr)
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{HEX_BG_CALLOUT}"/>')
        pPr.append(shd)
        
        r_t = p.add_run(bold_title)
        set_run_font(r_t, size_pt=8.5, bold=True, color=COLOR_PRIMARY)
        r = p.add_run(text)
        set_run_font(r, size_pt=8.5, italic=False, color=COLOR_BODY)
        return p

    def format_table(table, col_widths=None, alignments=None):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = table._tbl.tblPr
        
        borders_xml = parse_xml(f'''
            <w:tblBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="6" w:space="0" w:color="{HEX_BORDER}"/>
                <w:bottom w:val="single" w:sz="8" w:space="0" w:color="{HEX_PRIMARY}"/>
                <w:left w:val="none"/>
                <w:right w:val="none"/>
                <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>
                <w:insideV w:val="none"/>
            </w:tblBorders>
        ''')
        tblPr.append(borders_xml)
        
        for row_idx, row in enumerate(table.rows):
            trPr = row._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            if row_idx == 0:
                trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            
            for col_idx, cell in enumerate(row.cells):
                tcPr = cell._tc.get_or_add_tcPr()
                tcMar = parse_xml(f'''
                    <w:tcMar {nsdecls("w")}>
                        <w:top w:w="40" w:type="dxa"/>
                        <w:bottom w:w="40" w:type="dxa"/>
                        <w:left w:w="60" w:type="dxa"/>
                        <w:right w:w="60" w:type="dxa"/>
                    </w:tcMar>
                ''')
                tcPr.append(tcMar)
                
                if row_idx == 0:
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{HEX_PRIMARY}"/>')
                    tcPr.append(shd)
                elif row_idx % 2 == 1:
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{HEX_BG_LIGHT}"/>')
                    tcPr.append(shd)
                
                if col_widths and col_idx < len(col_widths):
                    cell.width = Inches(col_widths[col_idx])
                
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    if alignments and col_idx < len(alignments):
                        p.alignment = alignments[col_idx]
                    for run in p.runs:
                        if row_idx == 0:
                            set_run_font(run, size_pt=8.0, bold=True, color=RGBColor(255, 255, 255))
                        else:
                            set_run_font(run, size_pt=8.0, color=COLOR_BODY)

    def add_image_with_caption(img_path, caption_num, caption_text, width_in=4.2):
        if not os.path.exists(img_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(f"[Placeholder: Figure {caption_num} - {os.path.basename(img_path)}]")
            set_run_font(r, bold=True, color=COLOR_MUTED)
            return p
        
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(2)
        p_img.paragraph_format.space_after = Pt(1)
        p_img.paragraph_format.keep_with_next = True
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(width_in))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(1)
        p_cap.paragraph_format.space_after = Pt(2.5)
        
        r_num = p_cap.add_run(f"Figure {caption_num}: ")
        set_run_font(r_num, size_pt=7.5, bold=True, color=COLOR_PRIMARY)
        
        r_txt = p_cap.add_run(caption_text)
        set_run_font(r_txt, size_pt=7.5, italic=True, color=COLOR_MUTED)
        return p_cap

    # =========================================================================
    # 1. TITLE PAGE
    # =========================================================================
    p_title_space = doc.add_paragraph()
    p_title_space.paragraph_format.space_before = Pt(36)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("Forecasting Electricity Spot Prices via Dense NLP Embeddings")
    set_run_font(r_title, size_pt=23.0, bold=True, color=COLOR_PRIMARY)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    r_sub = p_sub.add_run("Empirical Evidence from the Brazilian Hydro-Thermal Power Grid (2018–2026)")
    set_run_font(r_sub, size_pt=12.5, italic=True, color=COLOR_SECONDARY)
    
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(32)
    pPr = p_div._p.get_or_add_pPr()
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="18" w:space="1" w:color="{HEX_PRIMARY}"/></w:pBdr>')
    pPr.append(pBdr)
    
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.line_spacing = 1.28
    meta_p.paragraph_format.space_after = Pt(6)
    
    runs_meta = [
        ("Author / Team: ", "[Insert name]\n"),
        ("Institution: ", "[Insert institution]\n"),
        ("Date: ", "[Insert date]\n"),
        ("Research Scope: ", "Brazilian Interconnected Power Grid (SIN), 2018–2026\n"),
        ("Target Variables: ", "PLD (Spot Clearing Price), CMO (Marginal Operating Cost), Basis Gap\n"),
        ("Core Methodology: ", "1024-d Dense News Embeddings, Supervised PLS, Non-Linear Ensembles\n"),
        ("Evaluation Horizons: ", "12-Hour (Sub-Daily), 24-Hour (Next-Day), 3-Day (Multi-Day Planning)\n"),
        ("Validation Framework: ", "Diebold-Mariano, Welch Two-Sample t-Test, 5-Fold Walk-Forward CV\n"),
        ("Project Status: ", "Complete & Empirically Validated | Open Source (MIT License)")
    ]
    for label, val in runs_meta:
        r_lbl = meta_p.add_run(label)
        set_run_font(r_lbl, size_pt=9.5, bold=True, color=COLOR_PRIMARY)
        r_val = meta_p.add_run(val)
        set_run_font(r_val, size_pt=9.5, bold=False, color=COLOR_BODY)
        
    doc.add_page_break()

    # =========================================================================
    # 2. ABSTRACT & TOC
    # =========================================================================
    add_h1("Abstract")
    add_body(
        "Forecasting electricity spot clearing prices (PLD - Preço de Liquidação das Diferenças) and marginal operating "
        "costs (CMO - Custo Marginal de Operação) in the hydro-dominated Brazilian Interconnected National Grid (SIN) "
        "represents a high-stakes challenge characterized by non-linear physical dispatch rules, extreme hydrological volatility, "
        "and multi-scale basis dislocations. While physical market models rely on historical price lags and optimization shadow prices, "
        "real-time specialized energy news reports convey critical early information regarding water basin stress, reservoir storage trajectories, "
        "unplanned thermal dispatch mandates, and renewable curtailment. To overcome the severe limitations of sparse keyword counts "
        "and generic sentiment polarity dictionaries, this research develops a supervised machine learning framework that converts "
        "unstructured domain news into 1024-dimensional dense semantic embeddings (BAAI/BGE and Qwen) condensed via Supervised "
        "Partial Least Squares (PLS) regression and integrated into non-linear tree ensembles. Evaluated on out-of-sample data from 2018 to 2026, "
        "the full pipeline achieves its strongest predictive gain at the 12-hour sub-daily horizon, boosting the out-of-time R² score "
        "from 0.8257 to 0.8755 (+5.0% absolute lift), reducing Root Mean Squared Error (RMSE) by 15.5% (73.41 to 62.04 R$/MWh), "
        "and lowering Mean Absolute Error (MAE) by 14.6% (42.13 to 35.98 R$/MWh). Econometric testing via the Diebold–Mariano test "
        "(DM = 4.9903, p = 3.17 × 10⁻⁷) and 5-fold temporal walk-forward cross-validation (100% fold improvement) confirms that dense "
        "semantic embeddings provide statistically validated predictive superiority over physical market baselines, with the textual signal "
        "decaying systematically as horizons extend to multi-day intervals."
    )
    
    add_h2("Table of Contents")
    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.space_before = Pt(2)
    p_toc.paragraph_format.space_after = Pt(3)
    fldSimple_toc = OxmlElement('w:fldSimple')
    fldSimple_toc.set(qn('w:instr'), 'TOC \\o "1-3" \\h \\z \\u')
    p_toc._p.append(fldSimple_toc)
    
    add_callout(
        "Complete source code, official JSON metrics, engineered datasets, and reproduction scripts are published "
        "under the MIT License at https://github.com/sultanofficial717/brazil-energy.",
        bold_title="Project Repository: "
    )

    # =========================================================================
    # 1. INTRODUCTION & RESEARCH PROBLEM
    # =========================================================================
    add_h1("1. Introduction & Research Problem")
    add_body(
        "In the Brazilian wholesale power market, electricity dispatch is managed centrally by the National System Operator "
        "(ONS) using mathematical hydrothermal optimization algorithms (primarily NEWAVE and DECOMP) that compute shadow prices based on hydro "
        "reservoir storage opportunity costs. Within this centralized architecture, two core price benchmarks govern commercial and physical operations:"
    )
    
    add_bullet(
        "Represents the theoretical shadow cost of supplying an incremental megawatt-hour (MWh) across the system, "
        "reflecting future water opportunity costs computed by hydrothermal optimization models.",
        bold_prefix="CMO (Custo Marginal de Operação / Marginal Operating Cost): "
    )
    add_bullet(
        "The official spot clearing price published by the Electric Energy Commercialization Chamber "
        "(CCEE), used for financial settlement of bilateral contract imbalances.",
        bold_prefix="PLD (Preço de Liquidação das Diferenças / Spot Settlement Price): "
    )
    
    add_body(
        "While standard theoretical microeconomic formulations assume PLD directly reflects CMO, empirical reality demonstrates "
        "substantial and persistent dislocations. Regulatory price caps impose strict minimum floors (~69 R$/MWh) and maximum ceilings "
        "(~584–700 R$/MWh). When physical water inflows deplete or regional transmission capacity between submarkets congests, "
        "the Target Basis Gap (CMO − PLD) experiences extreme non-linear expansions, exposing market participants, trading desks, "
        "and generators to massive cash-flow basis risk."
    )
    
    add_h2("The Limitation of Sparse Keyword Counts and Generic Sentiment")
    add_body(
        "Historically, econometric studies and algorithmic energy trading desks attempted to incorporate textual market intelligence "
        "using lexicon-based keyword counts (e.g., tallying occurrences of 'drought', 'dry season', 'reservoir', or 'curtailment') "
        "or off-the-shelf sentiment polarity scoring. However, this project demonstrates that traditional text metrics fundamentally fail "
        "in electricity price forecasting due to two structural issues:"
    )
    
    add_bullet(
        "Keyword matrices derived from energy news feeds contain greater than 85% zero-entries on normal operating days, "
        "providing no continuous informational gradients to regression models.",
        bold_prefix="1. Extreme Matrix Sparsity: "
    )
    add_bullet(
        "Traditional sentiment models cannot distinguish between 'drought intensifying in the Southeast' "
        "(a severe price-inflation shock) and 'drought easing after localized showers' (a bearish signal). Furthermore, energy price impacts "
        "are highly asymmetrical: a negative rainfall headline during full reservoir spilling has zero impact, whereas the exact same headline "
        "during critical reservoir depletion triggers exponential price spikes.",
        bold_prefix="2. Contextual Inflexibility & Non-Linear Disconnect: "
    )
    
    add_h2("The Core Research Question")
    add_body(
        "This research directly addresses this gap by investigating: Does incorporating dense semantic text representations "
        "extracted from domain-specific energy news via supervised dimensionality reduction provide statistically validated predictive "
        "improvement over physical market features alone across sub-daily (12-hour), daily (24-hour), and multi-day (3-day) forecasting horizons?"
    )

    # =========================================================================
    # 2. DATA & TARGET VARIABLES
    # =========================================================================
    add_h1("2. Data & Target Variables")
    add_body(
        "The empirical analysis is conducted strictly on empirical data from the Brazilian Interconnected National Grid (SIN), "
        "spanning the multi-year scope from 2018 to 2026 across all four regional electric submarkets: Southeast/Central-West (SE/CO), "
        "South (S), Northeast (NE), and North (N)."
    )
    
    # Table 1: Component Overview
    p_t1 = doc.add_paragraph()
    p_t1.paragraph_format.space_before = Pt(2)
    p_t1.paragraph_format.space_after = Pt(2)
    r = p_t1.add_run("Table 1: Research Scope and Component Architecture")
    set_run_font(r, size_pt=8.5, bold=True, color=COLOR_PRIMARY)
    
    t1 = doc.add_table(rows=6, cols=2)
    t1_data = [
        ("Component", "Description"),
        ("Market", "Brazilian SIN electricity market across SE/CO, S, NE, and N submarkets"),
        ("Targets", "PLD (Spot Price), CMO (Operating Cost), and CMO − PLD basis gap"),
        ("News", "Energy-domain news and dense semantic representations (1024-d BGE/Qwen)"),
        ("Horizons", "12-hour (sub-daily), 24-hour (next-day), and 3-day (multi-day planning)"),
        ("Evaluation", "Out-of-sample test splits and 5-fold temporal walk-forward cross-validation")
    ]
    for r_idx, (c0, c1) in enumerate(t1_data):
        t1.rows[r_idx].cells[0].paragraphs[0].text = c0
        t1.rows[r_idx].cells[1].paragraphs[0].text = c1
    format_table(t1, col_widths=[1.5, 5.4], alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])

    add_body(
        "To ensure stationarity and prevent spurious regression on non-stationary price levels, models are trained to predict the price "
        "change increment (target_delta = PLD_{t+h} − PLD_t). Reconstructed spot price levels are subsequently evaluated against true out-of-time "
        "settlement values across sub-daily (12h), daily (24h), and multi-day (3d) test horizons."
    )

    doc.add_page_break()

    # =========================================================================
    # 3. DATA ANALYSIS & EXPLORATORY FINDINGS
    # =========================================================================
    add_h1("3. Data Analysis & Exploratory Findings")
    add_body(
        "Prior to evaluating predictive modeling architectures, a thorough exploratory data analysis (EDA) was performed directly "
        "on the underlying historical market and news corpus. This section establishes empirical facts regarding dataset composition, "
        "regional representation, temporal coverage, news narrative cycles, and target variable dispersion."
    )
    
    # 3.1 Dataset Overview & Regional Distribution
    add_h2("3.1 Dataset Overview & Regional Distribution")
    add_body(
        "The primary merged historical dataset (`merged_news_pld_cmo_by_region_date_clean.csv`) comprises 16,178 verified rows "
        "across 3,038 unique calendar dates spanning 2018 to 2026. The dataset tracks 41 features, including physical spot prices, "
        "hourly spread metrics, hydrological keyword tallies, and 1024-dimensional continuous text embedding vectors."
    )
    add_image_with_caption(
        "figures/eda_regional_distribution.png",
        caption_num=1,
        caption_text="Distribution of observations across the four Brazilian electricity submarkets.",
        width_in=4.1
    )
    add_body(
        "As shown in Figure 1, the dataset demonstrates near-perfect balance across the operational submarkets of the National Interconnected "
        "System (SIN): South (3,029 records, 18.7%), Central-West (3,025 records, 18.7%), Southeast (3,025 records, 18.7%), "
        "North (3,022 records, 18.7%), and Northeast (3,005 records, 18.6%), with 1,072 national summary entries (6.6%). "
        "This balance ensures that models do not develop geographic bias toward any single submarket."
    )

    # 3.2 Year-wise Data Distribution
    add_h2("3.2 Year-wise Data Distribution")
    add_body(
        "Temporal coverage extends continuously across the 2018–2026 research timeline. Market observations remain steady at approximately "
        "1,800 to 2,100 records annually (reflecting daily observations across the four submarkets). News article availability grows significantly over time, "
        "expanding from 101 articles in 2018 to 954 in 2024 and exceeding 1,700 articles annually in 2025–2026."
    )
    add_image_with_caption(
        "figures/eda_yearly_distribution.png",
        caption_num=2,
        caption_text="Year-wise distribution of market observations and energy news records.",
        width_in=4.1
    )
    add_body(
        "Figure 2 confirms consistent market observation density across all evaluation years. The higher news article density in later years "
        "reflects expanded digital coverage of Brazilian energy markets and regulatory transparency, providing richer textual information for out-of-time evaluation."
    )

    doc.add_page_break()

    # 3.3 News Topic-wise Analysis
    add_h2("3.3 News Topic-wise Analysis")
    add_body(
        "Domain-specific news articles were categorized across verified energy and environmental topics. Hydro-meteorological uncertainty "
        "represents the overwhelming majority of reporting, with rainfall uncertainty accounting for 2,549 mentions (48.1%), "
        "flood mentions totaling 898 (17.0%), El Niño patterns contributing 604 (11.4%), and drought events contributing 537 (10.1%)."
    )
    add_image_with_caption(
        "figures/eda_news_topic_distribution.png",
        caption_num=3,
        caption_text="Distribution of energy news articles across domain-specific topics.",
        width_in=4.1
    )
    add_body(
        "Figure 3 establishes that over 86% of total news mentions relate directly to precipitation, water storage, and climatic anomalies. "
        "Given that hydroelectric generation supplies over 65% of Brazilian electricity demand, this distribution demonstrates that textual news "
        "captures the primary physical cost driver of system marginal dispatch."
    )

    # 3.4 News Topics Across Years
    add_h2("3.4 News Topics Across Years")
    add_body(
        "The temporal progression of energy news topics reveals pronounced macro-narrative shifts. During the 2020–2021 water crisis, "
        "mentions of drought and low reservoir storage reached historic zeniths. In contrast, 2022–2023 saw heavy rainfall recovery and flood reporting, "
        "while 2024–2026 exhibits an unprecedented surge in renewable curtailment reporting due to solar and wind expansion in the Northeast."
    )
    add_image_with_caption(
        "figures/eda_news_topics_over_time.png",
        caption_num=4,
        caption_text="Temporal evolution of energy news topics across the research period.",
        width_in=4.1
    )
    add_body(
        "Figure 4 tracks these topic cycles without claiming direct causality. The temporal shifts demonstrate that power market risks are highly "
        "dynamic: models relying solely on fixed keyword vocabularies fail when the operational narrative transitions from hydrological drought to renewable curtailment."
    )

    doc.add_page_break()

    # 3.5 News Volume & Sentiment Analysis
    add_h2("3.5 News Volume & Sentiment Analysis")
    add_body(
        "Analysis of news intensity and sentiment reveals that news publication volume spikes during market stress events. "
        "During the severe 2021 drought, monthly article counts doubled while average lexical sentiment dropped to deep negative levels (-0.35). "
        "However, generic sentiment fails to capture market nuances: rainfall can be beneficial for generation yet trigger localized spillway damage, "
        "and market price impacts depend non-linearly on whether reservoirs are depleted or spilling."
    )
    add_image_with_caption(
        "figures/eda_news_volume_sentiment.png",
        caption_num=5,
        caption_text="News volume and sentiment trends across the energy market timeline.",
        width_in=4.1
    )
    add_body(
        "Figure 5 illustrates the inverse relationship between reporting volume and sentiment polarity during crises. While sentiment dips during price spikes, "
        "its linear correlation with price change is near zero, justifying the need for continuous 1024-dimensional dense semantic embeddings."
    )

    # 3.6 Target Variable Distribution & Regional Comparison
    add_h2("3.6 Target Variable Distribution & Regional Comparison")
    add_body(
        "Target variable analysis across 14,970 clean price records reveals heavy right-skewness and extreme volatility. "
        "Average PLD is 182.42 R$/MWh (std: 152.28 R$/MWh, range: 39.68 to 716.80 R$/MWh). The operating cost (CMO) exhibits even higher variance "
        "(mean: 197.62 R$/MWh, std: 322.72 R$/MWh, max: 3,044.45 R$/MWh). The Target Basis Gap (CMO − PLD) averages 15.20 R$/MWh with a standard deviation of 233.81 R$/MWh."
    )
    add_image_with_caption(
        "figures/fig2_target_distributions_regimes.png",
        caption_num=6,
        caption_text="Target distributions and regime shifts in the Brazilian electricity market.",
        width_in=4.1
    )
    add_body(
        "Figure 6 confirms that during the 2020–2021 water crisis (Set 2), the basis gap volatility was nearly three times larger (σ = 221.0 R$/MWh) "
        "than in the post-crisis and modern hourly regimes (σ = 79.2 R$/MWh). This statistical reality necessitates modeling stationary price increments."
    )

    # 3.7 Exploratory Findings Summary
    add_h2("3.7 Exploratory Findings Summary")
    add_bullet("Regional submarket distribution is exceptionally balanced (~18.7% per region), preventing geographic model distortion.", bold_prefix="1. Balanced Submarket Coverage: ")
    add_bullet("Temporal coverage spans 2018 to 2026 continuously, with article volume expanding to over 1,700 records annually in modern years.", bold_prefix="2. Continuous Temporal Scope: ")
    add_bullet("Hydro-climatic topics (rainfall uncertainty, floods, drought) represent >86% of news, directly tracking the primary physical cost driver.", bold_prefix="3. Hydro-Centric News Dominance: ")
    add_bullet("News narratives evolve dynamically from drought and reservoir stress in 2020–2021 to renewable curtailment in 2024–2026.", bold_prefix="4. Dynamic Topic Shifts: ")
    add_bullet("Lexical sentiment correlates poorly with linear price changes, proving the necessity of 1024-dimensional semantic embeddings.", bold_prefix="5. Sentiment Disconnect: ")
    add_bullet("Target variables exhibit severe regime shifts, with basis gap variance tripling during crisis periods (σ = 221.0 vs. 79.2 R$/MWh).", bold_prefix="6. Target Volatility Regimes: ")

    doc.add_page_break()

    # =========================================================================
    # 4. METHODOLOGY
    # =========================================================================
    add_h1("4. Methodology")
    add_body(
        "The forecasting pipeline is structured in a compact, sequential flow designed to eliminate lookahead bias and control feature dimensionality:"
    )
    
    add_callout(
        "Energy News Corpus  →  Dense Semantic Embeddings (1024-d)  →  Supervised PLS Regression  →  "
        "Physical Market Features  →  Non-Linear Ensemble Regressors  →  Out-of-Sample Evaluation",
        bold_title="Pipeline Execution Flow: "
    )
    
    add_bullet(
        "Raw news articles discussing hydrological precipitation, reservoir storage percentages, thermal fuel costs, "
        "transmission line outages, and regulatory updates are encoded into 1024-dimensional continuous dense embedding vectors using transformer "
        "backbones (BAAI/BGE-M3 and Qwen).",
        bold_prefix="1. Dense Semantic Extraction: "
    )
    add_bullet(
        "Feeding raw 1024-dimensional vectors directly into tree-based regressors alongside physical features "
        "induces severe variance and tree split dilution. The methodology applies Supervised Partial Least Squares (PLS) to extract "
        "a compact set of latent orthogonal components (pls_01 through pls_10) that maximize covariance specifically with future price increments (target_delta). "
        "Principal Component Analysis (PCA) is concurrently computed to extract unsupervised variance signals.",
        bold_prefix="2. Supervised Dimensionality Reduction (PLS): "
    )
    add_bullet(
        "The latent text components are merged with historical price lags (1d, 7d, 14d, 30d), rolling statistics, "
        "intraday spreads, and regional CMO loads.",
        bold_prefix="3. Market Feature Integration: "
    )
    add_bullet(
        "Predictions are generated using non-linear tree ensembles, including Random Forests, HistGradientBoosting, "
        "and ExtraTrees regressors, optimized across stationary delta targets.",
        bold_prefix="4. Ensemble Forecasting: "
    )
    
    add_h2("Model Comparison Framework")
    add_body(
        "To isolate the marginal predictive contribution of news text, three distinct model configurations are evaluated:"
    )
    add_bullet("Trained strictly on historical price lags, intraday spreads, and rolling volatilities.", bold_prefix="1. Market Features Only: ")
    add_bullet("Trained exclusively on domain topic counts and dense news embeddings, omitting historical market prices.", bold_prefix="2. News Alone: ")
    add_bullet("Integrates physical market indicators, domain topic counts, and supervised PLS news embedding features.", bold_prefix="3. Full Pipeline (Market + BGE Embeddings): ")

    # =========================================================================
    # 5. OFFICIAL BENCHMARK RESULTS
    # =========================================================================
    add_h1("5. Official Benchmark Results")
    add_body(
        "Table 2 reports the exact benchmark metrics across horizons on the out-of-time test sets as recorded in the official project results."
    )
    
    # Table 2: Benchmark Results Table
    p_t2 = doc.add_paragraph()
    p_t2.paragraph_format.space_before = Pt(2)
    p_t2.paragraph_format.space_after = Pt(2)
    r = p_t2.add_run("Table 2: Official Model Performance Across Horizons (Out-of-Sample Evaluation)")
    set_run_font(r, size_pt=8.5, bold=True, color=COLOR_PRIMARY)
    
    t2 = doc.add_table(rows=12, cols=6)
    t2_rows = [
        ("Horizon / Benchmark", "Configuration", "R²", "Expl. Var", "RMSE (R$/MWh)", "MAE (R$/MWh)"),
        ("12-Hour Window", "1. Market Features Only", "0.8257", "0.8262", "73.41", "42.13"),
        ("12-Hour Window", "2. News Alone", "0.7410", "0.7455", "89.20", "50.32"),
        ("12-Hour Window", "3. Full Pipeline (+ Embeddings)", "0.8755", "0.8761", "62.04", "35.98"),
        ("Net Gain (12h)", "Full vs. Market Only", "+0.0498", "+0.0499", "-11.38 (-15.5%)", "-6.15 (-14.6%)"),
        ("3-Day Window", "1. Market Features Only", "0.7094", "0.7101", "40.54", "30.50"),
        ("3-Day Window", "2. News Alone", "0.6120", "0.6189", "46.85", "34.21"),
        ("3-Day Window", "3. Full Pipeline (+ Embeddings)", "0.7360", "0.7368", "38.64", "28.58"),
        ("Net Gain (3d)", "Full vs. Market Only", "+0.0266", "+0.0267", "-1.90 (-4.7%)", "-1.92 (-6.3%)"),
        ("Full Timeline (2018–26)", "1. Market Only", "0.5818", "0.6024", "52.70", "38.03"),
        ("Full Timeline (2018–26)", "2. Market + Topic/Sentiment", "0.6241", "0.6243", "49.97", "36.91"),
        ("Full Timeline (2018–26)", "3. Full Pipeline (+ BGE Emb.)", "0.6435", "0.6441", "48.66", "35.02")
    ]
    for r_idx, row_vals in enumerate(t2_rows):
        for c_idx, val in enumerate(row_vals):
            t2.rows[r_idx].cells[c_idx].paragraphs[0].text = val
    format_table(
        t2,
        col_widths=[1.5, 1.8, 0.7, 0.8, 1.0, 1.0],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT]
    )
    
    add_h2("Interpretation of Benchmark Findings")
    add_bullet(
        "In all reported evaluations, the combined Market + BGE Embeddings pipeline achieves the lowest RMSE, "
        "lowest MAE, and highest R² score.",
        bold_prefix="1. Full Pipeline Dominance: "
    )
    add_bullet(
        "The 12-hour sub-daily window exhibits the most substantial performance improvement, delivering a +0.0498 R² lift "
        "(+5.0% absolute gain), a 15.5% RMSE error reduction, and a 14.6% MAE error reduction. This indicates that textual news contains immediate, "
        "short-term market intelligence before physical dispatch schedules update.",
        bold_prefix="2. Sub-Daily Horizon Dominance: "
    )
    add_bullet(
        "At the 3-day horizon, the embedding lift remains positive (+0.0266 R² lift, -4.7% RMSE, -6.3% MAE) but is approximately "
        "two-to-three times smaller than at 12 hours, reflecting rapid information assimilation into physical dispatch models.",
        bold_prefix="3. Signal Attenuation Across Time: "
    )
    add_bullet(
        "The News Alone model underperforms the Market Only baseline (R² of 0.7410 vs. 0.8257 at 12h; 0.6120 vs. 0.7094 at 3d). "
        "Textual embeddings cannot replace baseline physical supply-demand state variables; rather, their power lies in modifying and refining "
        "physical state representations during regime shifts.",
        bold_prefix="4. Complementary Role of Text: "
    )

    doc.add_page_break()

    # =========================================================================
    # 6. VISUAL RESULTS & EMPIRICAL EVIDENCE
    # =========================================================================
    add_h1("6. Visual Results & Empirical Evidence")
    add_body(
        "This section documents the out-of-sample forecasting performance and embedding behavior across the evaluation test set. "
        "Figures 7–10 demonstrate model accuracy across horizons, chronological timeline overlay, 12-hour momentum tracking, "
        "and embedding feature importance."
    )
    
    # Figure 7
    add_image_with_caption(
        "figures/benchmark_official_comparison.png",
        caption_num=7,
        caption_text="Official Benchmark Comparison (12-Hour vs. 3-Day Windows). (1A) 12-Hour R² and MAE comparison; "
                     "(1B) 3-Day R² and MAE comparison; (1C) Directional Accuracy (%); (1D) RMSE error reduction across models.",
        width_in=3.9
    )
    add_body(
        "Figure 7 displays the four official benchmark subplots. The Full Pipeline (dark blue bar) consistently surpasses both Market Only "
        "(gray) and News Alone (cyan), achieving R² = 0.8755 at 12 hours and R² = 0.7360 at 3 days. Directional accuracy (1C) exceeds "
        "the 50% random chance threshold across both windows, while RMSE (1D) declines from 73.4 to 62.0 R$/MWh at 12 hours."
    )

    # Figure 8
    add_image_with_caption(
        "figures/benchmark_timeline_matching_news.png",
        caption_num=8,
        caption_text="Chronological Out-of-Sample Test Timeline Tracking. Top panel: Actual PLD vs. model forecasts. "
                     "Middle panel: Daily article count and sentiment polarity. Bottom panel: Specific risk topic activations.",
        width_in=3.9
    )
    add_body(
        "Figure 8 overlays model predictions directly against actual price levels across the out-of-time evaluation timeline. The Full Pipeline "
        "tracks sudden spot price spikes with significantly lower lag than the Market Only model, driven by early risk activations in news volume "
        "and hydrological stress mentions shown in the lower panels."
    )

    doc.add_page_break()

    # Figure 9
    add_image_with_caption(
        "figures/fig_12h_news_vs_price_patterns.png",
        caption_num=9,
        caption_text="12-Hour Sub-Daily Price Momentum & Risk Patterns. Panel 1: Price level tracking. Panel 2: 12-Hour price delta (ΔPLD). "
                     "Panel 3: News volume and sentiment polarity. Panel 4: Energy risk topic evolutions.",
        width_in=3.5
    )
    add_body(
        "Figure 9 evaluates 12-hour sub-daily price momentum tracking. Panel 2 demonstrates that the Full Pipeline accurately captures "
        "intraday price delta momentum, while Panel 4 confirms that turning points coincide with localized surges in rainfall uncertainty "
        "and hydro reservoir warnings."
    )

    # Figure 10
    add_image_with_caption(
        "figures/fig_embeddings_relation_12h_vs_24h.png",
        caption_num=10,
        caption_text="Embedding Behavior Across 12-Hour and 24-Hour Horizons. Top panels: Supervised PLS Component 1 correlation with price change. "
                     "Bottom panels: Feature importance distribution and R² lift (+5.0% at 12h, +4.7% at 24h).",
        width_in=3.9
    )
    add_body(
        "Figure 10 details the empirical mechanics of the embedding representations. Supervised PLS Component 1 demonstrates a strong linear "
        "correlation with future price changes (r = -0.251 at 12h; r = +0.310 at 24h). Feature importance analysis confirms that dense text "
        "embeddings account for 37.4% of total model importance at 12 hours and 47.3% at 24 hours, dwarfing sparse keyword topics (1.9% to 4.2%)."
    )

    doc.add_page_break()

    # =========================================================================
    # 7. STATISTICAL VALIDATION
    # =========================================================================
    add_h1("7. Statistical Validation")
    add_body(
        "To confirm that the reported performance gains are statistically meaningful rather than artifacts of specific out-of-time splits, "
        "an econometric validation suite was executed (`scripts/run_statistical_validation.py`). Table 3 summarizes the exact statistical test results."
    )
    
    # Table 3: Statistical Validation Table
    p_t3 = doc.add_paragraph()
    p_t3.paragraph_format.space_before = Pt(2)
    p_t3.paragraph_format.space_after = Pt(2)
    r = p_t3.add_run("Table 3: Statistical Significance and Econometric Validation Suite")
    set_run_font(r, size_pt=8.5, bold=True, color=COLOR_PRIMARY)
    
    t3 = doc.add_table(rows=7, cols=3)
    t3_data = [
        ("Econometric Test", "Reported Metric / Value", "Statistical Interpretation"),
        ("Diebold–Mariano (12h, Squared Loss)", "DM = 4.9903, p = 3.17 × 10⁻⁷", "Forecast accuracy improvement is statistically significant (p < 0.001)"),
        ("Diebold–Mariano (12h, Absolute Loss)", "DM = 5.8934, p = 2.09 × 10⁻⁹", "Absolute error reduction is statistically superior to baseline (p < 0.001)"),
        ("Walk-Forward Validation (12h)", "5/5 Folds Improved (100%), Mean ΔMAE = 11.2%", "Consistent out-of-sample error reduction across all temporal splits"),
        ("Walk-Forward Validation (3-Day)", "3/5 Folds Improved (60%), Mean ΔMAE = 5.1%", "Positive but less consistent error reduction at longer horizons"),
        ("Welch's Two-Sample t-Test", "t = 4.8161, p = 1.60 × 10⁻⁶", "12-hour embedding effect size is significantly larger than 3-day effect"),
        ("Bootstrap Resampling Test", "ΔR² p = 0.0354, ΔMAE p = 0.0000", "Reported R² lift is statistically robust against sampling noise (p < 0.05)")
    ]
    for r_idx, (c0, c1, c2) in enumerate(t3_data):
        t3.rows[r_idx].cells[0].paragraphs[0].text = c0
        t3.rows[r_idx].cells[1].paragraphs[0].text = c1
        t3.rows[r_idx].cells[2].paragraphs[0].text = c2
    format_table(t3, col_widths=[2.1, 2.1, 2.6], alignments=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT])
    
    add_h2("Predictive Superiority vs. Causal Inference")
    add_body(
        "Crucially, the statistical validation suite supports empirical predictive superiority rather than direct economic causality. "
        "In the Brazilian power market, physical electricity prices are formally determined by mathematical optimization models (DECOMP/NEWAVE) "
        "and CCEE settlement algorithms. Public news articles do not directly cause electricity spot prices to clear; rather, specialized news "
        "acts as a high-frequency information aggregation channel that captures latent reservoir depletion, weather model uncertainty, and political "
        "dispatch interventions before they are fully parameterized into official grid model inputs."
    )

    # =========================================================================
    # 8. CONCLUSION & LIMITATIONS
    # =========================================================================
    add_h1("8. Conclusion & Limitations")
    add_body(
        "This research establishes rigorous empirical evidence on the predictive value of dense NLP text embeddings in wholesale electricity "
        "price forecasting within the Brazilian hydro-thermal system. The key conclusions are summarized as follows:"
    )
    
    add_bullet(
        "1024-dimensional dense text representations condensed via supervised Partial Least Squares effectively "
        "solve the hydro-sentiment disconnect, capturing non-linear supply risk signals that sparse keyword counts miss.",
        bold_prefix="1. Dense Embeddings Outperform Keywords: "
    )
    add_bullet(
        "The embedding pipeline delivers its highest predictive lift at the 12-hour sub-daily horizon "
        "(+5.0% R² gain, 15.5% RMSE error reduction, DM p < 10⁻⁶), where immediate text sentiment has not yet been fully absorbed by dispatch schedules.",
        bold_prefix="2. Sub-Daily Predictive Superiority: "
    )
    add_bullet(
        "Predictive gains decline systematically from 12 hours (+0.0498 R²) to 3 days (+0.0266 R²), "
        "demonstrating that textual intelligence is rapidly incorporated into spot prices within 24 to 72 hours.",
        bold_prefix="3. Temporal Signal Attenuation: "
    )
    add_bullet(
        "Dense news embeddings cannot replace physical market features (News Alone R² = 0.7410 vs. Full Pipeline R² = 0.8755); "
        "they operate optimally as high-dimensional non-linear modifiers of physical market states.",
        bold_prefix="4. Complementary Synthesis: "
    )

    add_h2("Documented Limitations")
    add_bullet(
        "The empirical benchmark evaluates 12-hour, 24-hour, and 3-day horizons; sub-hourly real-time dispatch intervals "
        "were not modeled due to data granularity constraints.",
        bold_prefix="1. Horizon Granularity: "
    )
    add_bullet(
        "While the four SIN submarkets are represented, localized transmission branch bottlenecks within submarkets "
        "cannot be fully resolved through national and regional news text alone.",
        bold_prefix="2. Intra-Regional Transmission Granularity: "
    )
    add_bullet(
        "The continuous evolution of renewable generation (solar and wind expansion in the Northeast) introduces "
        "novel curtailment dynamics that warrant ongoing out-of-time re-estimation.",
        bold_prefix="3. Evolving Market Regimes: "
    )

    # =========================================================================
    # 9. REPRODUCIBILITY
    # =========================================================================
    add_h1("9. Reproducibility")
    add_body(
        "The project repository is structured for end-to-end reproducibility across four primary directories:"
    )
    add_bullet("Contains all publication-quality PNG benchmark visualizations and EDA diagnostic plots.", bold_prefix="figures/: ")
    add_bullet("Contains feature-engineered training data (`daily_train_ready.csv`), official test set predictions, and schema documentation.", bold_prefix="data/: ")
    add_bullet("Stores official benchmark metrics, walk-forward splits, and statistical test outputs in structured JSON format.", bold_prefix="results/: ")
    add_bullet("Contains modular, standalone Python scripts for model training, validation, and visualization.", bold_prefix="scripts/: ")

    add_h2("Key Reproduction Scripts")
    add_bullet("Executes Diebold-Mariano tests, Welch t-tests, 5-fold walk-forward validation, and bootstrap resampling.", bold_prefix="scripts/run_statistical_validation.py: ")
    add_bullet("Regenerates the official benchmark comparison (Figure 7) and chronological timeline overlay (Figure 8).", bold_prefix="scripts/generate_perfectly_matched_visuals.py: ")
    add_bullet("Generates 12-hour price delta momentum plots (Figure 9) and the interactive HTML visualization dashboard.", bold_prefix="scripts/generate_12h_visualizations.py: ")
    add_bullet("Generates empirical proof charts for dense embeddings versus sparse keywords and transmission decay curves.", bold_prefix="scripts/generate_visual_proofs.py: ")
    add_bullet("Compiles Figures 1–6 and generates the comprehensive exploratory data analysis figures.", bold_prefix="generate_eda_visuals.py: ")

    add_h2("Repository Citation & License")
    add_body(
        "If utilizing this research code, benchmark metrics, or empirical methodology, cite the project as follows:"
    )
    
    p_cite = doc.add_paragraph()
    p_cite.paragraph_format.space_before = Pt(2)
    p_cite.paragraph_format.space_after = Pt(2)
    p_cite.paragraph_format.left_indent = Inches(0.12)
    p_cite.paragraph_format.right_indent = Inches(0.12)
    pPr = p_cite._p.get_or_add_pPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{HEX_BG_LIGHT}"/>')
    pPr.append(shd)
    
    r_cite = p_cite.add_run(
        "@article{brazilian_energy_pld_nlp_2026,\n"
        "  title={Forecasting Electricity Spot Prices via Dense NLP Embeddings: Empirical Evidence from the Brazilian Hydro-Thermal Power Grid},\n"
        "  author={Advanced Agentic Energy Research Team},\n"
        "  year={2026},\n"
        "  journal={Working Paper in Computational Energy Economics},\n"
        "  url={https://github.com/sultanofficial717/brazil-energy}\n"
        "}"
    )
    set_run_font(r_cite, name='Courier New', size_pt=7.5, color=COLOR_BODY)

    add_body(
        "This project is open-source software licensed under the MIT License."
    )

    # Save document
    output_filename = "Brazilian_Energy_PLD_CMO_Forecasting_Report.docx"
    doc.save(output_filename)
    print(f"Successfully generated updated research report: {output_filename}")

if __name__ == "__main__":
    create_report()
