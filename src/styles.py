CUSTOM_CSS = """
<style>
:root {
    --gu-dark: #0b3d2e;
    --gu-mid: #12563f;
    --gu-accent: #1f9d68;
    --gu-accent-light: #e6f6ee;
    --gu-gold: #f5b301;
}

.stApp {
    background: #f4f8f6;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, var(--gu-dark) 0%, var(--gu-mid) 55%, var(--gu-accent) 130%);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    color: #ffffff;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(11, 61, 46, 0.25);
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    margin-bottom: 0.8rem;
}
.hero-badge .dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    margin-right: 6px;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    line-height: 1.15;
}
.hero-sub {
    opacity: 0.9;
    font-size: 1rem;
    max-width: 640px;
}

/* Feature / stat cards */
.gu-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px;
    padding: 14px 16px;
    height: 100%;
}
.gu-card h4 {
    margin: 0 0 4px 0;
    font-size: 0.95rem;
    color: #ffffff;
}
.gu-card p {
    margin: 0;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.8);
}

/* White content cards used throughout the app */
.content-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 18px rgba(20, 60, 45, 0.06);
    border: 1px solid #e7f0ea;
    margin-bottom: 1.1rem;
}

.destination-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    border: 1px solid #e2f0e8;
    border-left: 5px solid var(--gu-accent);
    margin-bottom: 0.8rem;
    box-shadow: 0 3px 10px rgba(20,60,45,0.05);
}
.destination-card h4 { margin: 0 0 2px 0; color: var(--gu-dark); }
.destination-rank {
    display: inline-block;
    background: var(--gu-accent-light);
    color: var(--gu-mid);
    font-weight: 700;
    border-radius: 8px;
    padding: 1px 9px;
    font-size: 0.78rem;
    margin-right: 8px;
}
.match-pill {
    float: right;
    background: var(--gu-dark);
    color: white;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

.verdict-yes {
    background: linear-gradient(135deg, #e6f6ee, #d3f0e0);
    border: 1px solid #9fe0bd;
    color: #0b3d2e;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    font-weight: 600;
}
.verdict-no {
    background: linear-gradient(135deg, #fdecec, #fbdada);
    border: 1px solid #f2a9a9;
    color: #7a1f1f;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    font-weight: 600;
}

.factor-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px dashed #e4ece7;
    font-size: 0.88rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--gu-dark) 0%, var(--gu-mid) 100%);
}
section[data-testid="stSidebar"] * {
    color: #f2f7f4 !important;
}
section[data-testid="stSidebar"] .stRadio > label { color: #f2f7f4 !important; }

.stButton>button {
    background: var(--gu-accent);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
}
.stButton>button:hover {
    background: var(--gu-mid);
    color: white;
}

footer, #MainMenu {visibility: hidden;}
.gu-footer {
    text-align: center;
    color: #6b8f7c;
    font-size: 0.8rem;
    padding: 1.2rem 0 0.4rem 0;
}
</style>
"""
