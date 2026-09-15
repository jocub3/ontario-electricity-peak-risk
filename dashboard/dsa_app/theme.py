"""Central visual theme for Streamlit components and Plotly figures."""
from __future__ import annotations

COLORS = {
    "primary": "#1D5D9B",
    "primary_dark": "#102A43",
    "primary_mid": "#2F80C9",
    "primary_soft": "#EAF3FB",
    "background": "#F7F9FC",
    "surface": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#64748B",
    "normal": "#2E8B57",
    "warning": "#F2A900",
    "alert": "#D64545",
    "border": "#D9E2EC",
}

FSA_COLORS = ["#1D5D9B", "#2E8B57", "#7A5195", "#EF8354", "#4C78A8", "#9C755F"]


def app_css() -> str:
    """Return the global CSS. Colors are edited only in this module."""
    c = COLORS
    return f"""
    <style>
    .stApp {{ background: {c['background']}; }}
    .block-container {{ padding-top: 1.15rem; padding-bottom: 2.5rem; max-width: 1480px; }}
    [data-testid="stSidebar"] {{ background: {c['primary_dark']}; }}
    [data-testid="stSidebar"] * {{ color: #F7FAFC; }}
    [data-testid="stSidebarNav"] span {{ font-weight: 600; font-size: 1.06rem; }}
    [data-testid="stSidebarNav"] li {{ margin-bottom: 0.22rem; }}
    [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: rgba(47, 128, 201, 0.30);
        border-left: 4px solid {c['warning']};
        border-radius: 0.45rem;
    }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, {c['surface']} 0%, {c['primary_soft']} 100%);
        border: 1px solid #C8D9EA;
        border-radius: 0.75rem;
        padding: 0.85rem 1rem;
        box-shadow: 0 5px 14px rgba(16, 42, 67, 0.13);
    }}
    [data-testid="stMetricLabel"] {{ color: {c['muted']}; }}
    [data-testid="stMetricValue"] {{ color: {c['text']}; font-size: 1.55rem; }}
    div[data-testid="stForm"] {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 0.8rem;
        padding: 0.35rem 0.75rem 0.15rem;
        box-shadow: 0 1px 3px rgba(16, 42, 67, 0.05);
    }}
    .analysis-context {{ color: {c['muted']}; font-size: 0.88rem; margin: 0.25rem 0 1rem; }}
    .section-kicker {{ color: {c['primary']}; font-weight: 700; letter-spacing: .04em; font-size: .78rem; }}
    .page-section-header {{
        background: linear-gradient(90deg, {c['primary_dark']} 0%, {c['primary']} 100%);
        color: #FFFFFF;
        border-radius: 0.7rem;
        padding: 0.9rem 1.15rem;
        margin: 0.35rem 0 1rem;
        box-shadow: 0 3px 10px rgba(16, 42, 67, 0.14);
    }}
    .page-section-title {{ font-size: 1.55rem; font-weight: 700; line-height: 1.2; }}
    .page-section-caption {{ color: #E7F1FA; font-size: 0.94rem; margin-top: 0.28rem; }}
    .risk-alert {{
        background: #FFF4F4; border-left: 5px solid {c['alert']};
        border-radius: .5rem; padding: .85rem 1rem; margin: .4rem 0 1rem;
    }}
    .risk-alert-title {{ color: {c['alert']}; font-size: 1.08rem; font-weight: 750; margin-bottom: .45rem; }}
    .risk-alert-actions {{ margin-top: .65rem; padding-top: .55rem; border-top: 1px solid #F2C6C6; }}
    .risk-alert-actions ul {{ margin: .35rem 0 0 1.15rem; padding: 0; }}
    .risk-normal {{
        background: #EDF8F1; border-left: 5px solid {c['normal']};
        border-radius: .5rem; padding: .85rem 1rem; margin: .4rem 0 1rem;
    }}
    </style>
    """


def page_header(title: str, caption: str) -> str:
    """Return a visually separated heading for each application page."""
    return (
        '<div class="page-section-header">'
        f'<div class="page-section-title">{title}</div>'
        f'<div class="page-section-caption">{caption}</div>'
        '</div>'
    )


def style_figure(fig, *, height: int | None = None):
    """Apply a consistent, accessible Plotly layout."""
    fig.update_layout(
        font=dict(family="Arial, sans-serif", color=COLORS["text"]),
        title_font=dict(size=18, color=COLORS["primary_dark"]),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        margin=dict(l=55, r=30, t=70, b=55),
        legend_title_text="",
        hoverlabel=dict(bgcolor=COLORS["surface"], font_color=COLORS["text"]),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor="#E8EEF5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E8EEF5", zeroline=False)
    return fig
