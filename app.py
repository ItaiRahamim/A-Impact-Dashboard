import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import numpy as np

st.set_page_config(
    page_title="מגדל שירותים פיננסיים – דשבורד ניהולי",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS + RTL bootstrap ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* RTL base — most reliable cross-browser approach */
html, body { direction: rtl !important; }
/* Cover Streamlit's dynamically-generated class names AND key test-ids */
[class*="css"], [class*="st-"],
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-baseweb="tab"],
[data-baseweb="tab-list"],
[data-baseweb="select"],
[data-baseweb="menu"]           { direction: rtl !important; }

.stApp                          { background-color: #f4f6fb; direction: rtl !important; }
[data-testid="metric-container"]{ background:#fff; border-radius:12px;
                                   padding:16px 20px; border:1px solid #e5e7eb;
                                   box-shadow:0 1px 3px rgba(0,0,0,.06); }
[data-testid="stSidebar"]       { background:#1e293b !important; }
[data-testid="stSidebar"] *     { color:#e2e8f0 !important; }
.stTabs [data-baseweb="tab-list"]{ background:#fff; border-radius:12px;
                                   padding:5px; gap:4px;
                                   box-shadow:0 1px 3px rgba(0,0,0,.06); }
.stTabs [data-baseweb="tab"]    { border-radius:8px; font-weight:600;
                                   color:#64748b; padding:8px 18px; }
.stTabs [aria-selected="true"]  { background:#2563eb !important; color:#fff !important; }
.sec-title { font-size:1rem; font-weight:700; color:#1e293b;
             border-bottom:3px solid #2563eb; display:inline-block;
             padding-bottom:4px; margin-bottom:12px; }
.ac { background:#fff; border-radius:12px; padding:14px 16px;
      margin-bottom:8px; box-shadow:0 1px 3px rgba(0,0,0,.06); }
.ac.red   { border-right:5px solid #dc2626; }
.ac.amber { border-right:5px solid #d97706; }
.ac.green { border-right:5px solid #16a34a; }
.ac-title { font-weight:700; font-size:.95rem; color:#1e293b; margin-bottom:4px; }
.ac-body  { font-size:.84rem; color:#475569; line-height:1.55; }
</style>
""", unsafe_allow_html=True)

# ── Palette ──────────────────────────────────────────────────────────────────
CA = "#2563eb"   # active / blue
CC = "#dc2626"   # churn  / red
AM = "#d97706"   # amber
GR = "#16a34a"   # green
CM = {"פעיל": CA, "לא פעיל": CC}
HOVER_LABEL = dict(
    bgcolor="#fff",
    font_size=13,
    font_family="Arial",
    bordercolor="#e5e7eb",
    align="left",
    namelength=-1,
)

def fig_base(fig, h=340, lm=10, rm=16, bm=50, tm=36, legend=True, xangle=0):
    """Apply clean base layout to any figure."""
    fig.update_layout(
        height=h, paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Arial", size=12, color="#1e293b"),
        margin=dict(l=lm, r=rm, t=tm, b=bm),
        xaxis=dict(showgrid=False, linecolor="#e5e7eb",
                   tickangle=xangle, automargin=True),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                   linecolor="#e5e7eb", automargin=True),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1) if legend else dict(visible=False),
        hoverlabel=HOVER_LABEL,
    )
    return fig

def action(cls, title, body):
    st.markdown(
        f'<div class="ac {cls}"><div class="ac-title">{title}</div>'
        f'<div class="ac-body">{body}</div></div>',
        unsafe_allow_html=True,
    )

# helper: format shekel amounts as clean string
def _nis(v):
    return f"\u200e₪{v:,.0f}"

# helper: abbreviated format for KPI tiles so numbers never truncate
def _fmt_kpi(v):
    if v >= 1_000_000: return f"₪{v/1_000_000:.1f}M"
    if v >= 1_000:     return f"₪{v/1_000:.0f}K"
    return f"₪{int(v):,}"

def _hover_tpl(*lines):
    # Use Unicode 1.0 embedding chars (RLE/LRE/PDF) — supported in ALL browsers
    # including Safari's SVG text engine, unlike the newer bidi isolates (U+2066-2069).
    RLE, LRE, PDF = "\u202b", "\u202a", "\u202c"
    result = []
    for line in lines:
        # <span dir='ltr'>…</span>  →  LRE … PDF  (force LTR for numbers)
        ln = line.replace("<span dir='ltr'>", LRE).replace("</span>", PDF)
        # Wrap entire line in RTL embedding so label appears on right
        result.append(f"{RLE}{ln}{PDF}")
    return "<br>".join(result) + "<extra></extra>"

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("clients_data.csv")
    df["תאריך_הצטרפות"] = pd.to_datetime(df["תאריך_הצטרפות"], dayfirst=True)
    df["תאריך_נטישה"]   = pd.to_datetime(df["תאריך_נטישה"],   dayfirst=True, errors="coerce")
    df["שנת_הצטרפות"]   = df["תאריך_הצטרפות"].dt.year
    df["churn"]          = (df["סטטוס"] == "לא פעיל").astype(int)
    ref = pd.Timestamp("2026-03-17")
    df["tenure_days"] = np.where(
        df["תאריך_נטישה"].notna(),
        (df["תאריך_נטישה"] - df["תאריך_הצטרפות"]).dt.days,
        (ref - df["תאריך_הצטרפות"]).dt.days,
    )
    def risk(r):
        if r["סטטוס"] != "פעיל": return np.nan
        s = 0
        s += 30 if r["שביעות_רצון"] <= 3 else 15 if r["שביעות_רצון"] <= 5 else 0
        s += 25 if r["זמן_תגובה_ממוצע_שעות"] > 70 else 12 if r["זמן_תגובה_ממוצע_שעות"] > 50 else 0
        s += 20 if r["סכום_תיק"] < 600_000 else 8 if r["סכום_תיק"] < 800_000 else 0
        s += 15 if r["מספר_פניות_שנה_אחרונה"] <= 2 else 6 if r["מספר_פניות_שנה_אחרונה"] <= 4 else 0
        s += 10 if 270 <= r["tenure_days"] <= 430 else 0
        return min(s, 100)
    df["risk_score"] = df.apply(risk, axis=1)
    df["risk_label"] = pd.cut(df["risk_score"], bins=[-1,30,60,100],
                               labels=["נמוך","בינוני","גבוה"])
    return df

df = load()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 סינון נתונים")
    st.markdown("---")
    sel_city    = st.selectbox("🏙️ עיר",       ["הכל"] + sorted(df["עיר"].unique()))
    sel_service = st.selectbox("📋 סוג שירות", ["הכל"] + sorted(df["סוג_שירות"].unique()))
    sel_status  = st.selectbox("🔵 סטטוס",     ["הכל","פעיל","לא פעיל"])
    st.markdown("---")
    st.caption("מגדל שירותים פיננסיים | v2")

def filt(data):
    d = data.copy()
    if sel_city    != "הכל": d = d[d["עיר"]       == sel_city]
    if sel_service != "הכל": d = d[d["סוג_שירות"] == sel_service]
    if sel_status  != "הכל": d = d[d["סטטוס"]     == sel_status]
    return d

F = filt(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:14px;
            padding:20px 28px;margin-bottom:18px;color:#fff;">
  <h1 style="margin:0;font-size:1.55rem;font-weight:800;">
    📊 דשבורד ניהולי — מגדל שירותים פיננסיים
  </h1>
  <p style="margin:4px 0 0;opacity:.75;font-size:.88rem;">
    ניתוח נתוני 2,000 לקוחות | עודכן לפי הסינון הנוכחי
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
total    = len(F)
active   = int((F["סטטוס"]=="פעיל").sum())
inactive = total - active
churn_r  = round(inactive/total*100,1) if total else 0
avg_sat  = round(F["שביעות_רצון"].mean(),1) if total else 0
rev_act  = F[F["סטטוס"]=="פעיל"]["הכנסה_חודשית"].sum()
high_risk= int((F["risk_score"]>=60).sum())
rev_risk = F[F["risk_score"]>=60]["הכנסה_חודשית"].sum()

r1 = st.columns(3)
r1[0].metric("סה\"כ לקוחות",       f"{total:,}")
r1[1].metric("לקוחות פעילים",      f"{active:,}", delta=f"נטשו {inactive:,}", delta_color="inverse")
r1[2].metric("שיעור נטישה",        f"{churn_r}%")
r2 = st.columns(3)
r2[0].metric("שביעות רצון ממוצעת", f"{avg_sat} / 10")
r2[1].metric("הכנסה חודשית (פעילים)", _fmt_kpi(rev_act))
r2[2].metric("🔴 הכנסה בסיכון גבוה", _fmt_kpi(rev_risk),
             delta=f"{high_risk} לקוחות בסיכון גבוה", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# ── Pre-compute insight values (used in multiple tabs) ────────────────────────
worst_svc      = df.groupby("סוג_שירות")["churn"].mean().idxmax()
worst_svc_pct  = round(df.groupby("סוג_שירות")["churn"].mean().max()*100,1)
worst_city     = df.groupby("עיר")["churn"].mean().idxmax()
worst_city_pct = round(df.groupby("עיר")["churn"].mean().max()*100,1)
approaching    = int(((df["סטטוס"]=="פעיל") & df["tenure_days"].between(270,365)).sum())
top_cell       = df.groupby(["סוג_שירות","עיר"])["churn"].mean().idxmax()
top_cell_pct   = round(df.groupby(["סוג_שירות","עיר"])["churn"].mean().max()*100,1)
top_rev_city   = df[df["סטטוס"]=="פעיל"].groupby("עיר")["הכנסה_חודשית"].sum().idxmax()
top_rev_val    = df[df["סטטוס"]=="פעיל"].groupby("עיר")["הכנסה_חודשית"].sum().max()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
t1,t2,t3,t4,t5 = st.tabs([
    "📈 סקירה כללית",
    "🔴 ניתוח נטישה",
    "🧩 פילוח לקוחות",
    "⚠️ לקוחות בסיכון",
    "📥 ייצוא",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ──────────────────────────────────────────────────────────────────────────────
with t1:
    # ── INSIGHTS FIRST ──
    st.markdown('<p class="sec-title">💡 המלצות פעולה — מבוססות נתונים</p>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    with i1:
        action("red",
               f"🚨 יזום שיחות retention מיידי",
               f"<b>{high_risk} לקוחות פעילים</b> בציון סיכון גבוה (≥60).<br>"
               f"הכנסה חודשית מאוימת: <b>₪{rev_risk:,.0f}</b>.<br>"
               f"פעולה: שלח ליועצים רשימה ממוקדת + תסריט שיחה.")
    with i2:
        action("amber",
               f"⏰ {approaching} לקוחות מתקרבים לשנת הסיכון",
               f"נטישה מגיעה בממוצע אחרי <b>~12 חודש</b>.<br>"
               f"פעולה: שגר מייל נאמנות אישי + הצעת שדרוג לכל לקוח בחלון זה.")
    with i3:
        action("amber",
               f"📉 {worst_svc} — שיעור נטישה {worst_svc_pct}%",
               f"השירות עם הנטישה הגבוהה ביותר.<br>"
               f"פעולה: בחן מחדש תמחור ושדרג חבילות נאמנות לשירות זה.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    ca,cb = st.columns(2)

    with ca:
        st.markdown('<p class="sec-title">נטישה לאורך זמן (רבעוני)</p>', unsafe_allow_html=True)
        churned_f = F[F["תאריך_נטישה"].notna()].copy()
        if not churned_f.empty:
            churned_f["רבעון"] = churned_f["תאריך_נטישה"].dt.to_period("Q").astype(str)
            ct = churned_f.groupby("רבעון").size().reset_index(name="נוטשים")
            fig = go.Figure([go.Scatter(
                x=ct["רבעון"], y=ct["נוטשים"],
                mode="lines+markers",
                line=dict(color=CC, width=2.5), marker=dict(size=7),
                fill="tozeroy", fillcolor="rgba(220,38,38,.1)",
                customdata=list(zip(ct["רבעון"].tolist(), ct["נוטשים"].astype(str).tolist())),
                hovertemplate=_hover_tpl(
                    "רבעון: <span dir='ltr'>%{customdata[0]}</span>",
                    "מספר נוטשים: <span dir='ltr'>%{customdata[1]}</span>",
                ),
            )])
            fig = fig_base(fig, lm=10, bm=60, legend=False)
            fig.update_layout(xaxis_title="רבעון", yaxis_title="מספר נוטשים",
                              xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("אין נתוני נטישה בסינון הנוכחי")

    with cb:
        st.markdown('<p class="sec-title">הכנסה חודשית לפי סוג שירות (פעילים)</p>', unsafe_allow_html=True)
        rev_s = (F[F["סטטוס"]=="פעיל"]
                 .groupby("סוג_שירות")["הכנסה_חודשית"].sum()
                 .reset_index().sort_values("הכנסה_חודשית"))
        rev_s.columns = ["שירות","הכנסה"]
        fig = go.Figure([go.Bar(
            x=rev_s["הכנסה"], y=rev_s["שירות"],
            orientation="h",
            marker_color=[CA]*len(rev_s), marker_line_width=0,
            text=[f"₪{v/1000:.0f}K" for v in rev_s["הכנסה"]],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="white", size=12),
            customdata=list(zip(
                rev_s["שירות"].tolist(),
                [f"₪{v/1000:.0f}K" for v in rev_s["הכנסה"]],
            )),
            hovertemplate=_hover_tpl(
                "סוג שירות: %{customdata[0]}",
                "הכנסה חודשית: <span dir='ltr'>%{customdata[1]}</span>",
            ),
        )])
        max_rev = rev_s["הכנסה"].max()
        fig = fig_base(fig, lm=130, rm=10, bm=10, tm=36)
        fig.update_layout(legend=dict(visible=False),
                          xaxis_title="הכנסה חודשית (₪)",
                          yaxis_title="סוג שירות")
        st.plotly_chart(fig, use_container_width=True)

    cc,cd = st.columns(2)

    with cc:
        st.markdown('<p class="sec-title">שביעות רצון — פעילים vs נוטשים</p>', unsafe_allow_html=True)
        sat_d = (F.groupby(["שביעות_רצון","סטטוס"]).size().reset_index(name="כמות"))
        fig = go.Figure()
        for status, color in [("פעיל", CA),("לא פעיל", CC)]:
            d = sat_d[sat_d["סטטוס"]==status].copy()
            fig.add_trace(go.Bar(
                name=status, x=d["שביעות_רצון"], y=d["כמות"],
                marker_color=color, marker_line_width=0,
                customdata=list(zip(
                    d["שביעות_רצון"].astype(str).tolist(),
                    [status]*len(d),
                    d["כמות"].astype(str).tolist(),
                )),
                hovertemplate=_hover_tpl(
                    "ציון שביעות רצון: <span dir='ltr'>%{customdata[0]}</span>",
                    "סטטוס: %{customdata[1]}",
                    "מספר לקוחות: <span dir='ltr'>%{customdata[2]}</span>",
                ),
            ))
        fig = fig_base(fig, bm=50)
        fig.update_layout(barmode="group",
                          xaxis_title="ציון שביעות רצון (1–10)",
                          yaxis_title="מספר לקוחות",
                          xaxis=dict(tickmode="linear"))
        st.plotly_chart(fig, use_container_width=True)

    with cd:
        st.markdown('<p class="sec-title">שיעור נטישה לפי ציון שביעות רצון</p>', unsafe_allow_html=True)
        cs_sat = (F.groupby("שביעות_רצון")["churn"]
                  .mean().mul(100).round(1).reset_index())
        cs_sat.columns = ["ציון","נטישה"]
        colors = [CC if v>38 else AM if v>33 else GR for v in cs_sat["נטישה"]]
        fig = go.Figure([go.Bar(
            x=cs_sat["ציון"], y=cs_sat["נטישה"],
            marker_color=colors, marker_line_width=0,
            text=[f"{v}%" for v in cs_sat["נטישה"]], textposition="outside",
            customdata=list(zip(
                cs_sat["ציון"].astype(str).tolist(),
                [f"{v}%" for v in cs_sat["נטישה"]],
            )),
            hovertemplate=_hover_tpl(
                "ציון שביעות רצון: <span dir='ltr'>%{customdata[0]}</span>",
                "שיעור נטישה: <span dir='ltr'>%{customdata[1]}</span>",
            ),
        )])
        fig = fig_base(fig, bm=50)
        fig.update_layout(legend=dict(visible=False),
                          xaxis_title="ציון שביעות רצון (1–10)",
                          yaxis_title="שיעור נטישה (%)",
                          xaxis=dict(tickmode="linear"),
                          yaxis_range=[0,55])
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Churn Analysis
# ──────────────────────────────────────────────────────────────────────────────
with t2:
    # ── INSIGHTS FIRST ──
    st.markdown('<p class="sec-title">💡 ממצאים ופעולות מומלצות</p>', unsafe_allow_html=True)
    j1,j2 = st.columns(2)
    with j1:
        action("red",
               f"🔥 נקודת חום: {top_cell[0]} ב{top_cell[1]} — {top_cell_pct}% נטישה",
               f"שילוב השירות + העיר עם הנטישה הגבוהה ביותר בכל הנתונים.<br>"
               f"פעולה: בצע סקר שביעות רצון ממוקד לאותם לקוחות ובדוק גורם מתחרה מקומי.")
    with j2:
        action("amber",
               f"📍 {worst_city} — עיר עם נטישה גבוהה: {worst_city_pct}%",
               f"פעולה: בדוק אם יש יועץ ספציפי שמשרת את האזור ונתח את ביצועיו.")

    st.markdown("<br>", unsafe_allow_html=True)

    ca,cb = st.columns(2)

    with ca:
        st.markdown('<p class="sec-title">שיעור נטישה לפי סוג שירות</p>', unsafe_allow_html=True)
        cs_svc = (F.groupby("סוג_שירות")["churn"].mean()
                  .mul(100).round(1).reset_index().sort_values("churn"))
        cs_svc.columns = ["שירות","נטישה"]
        bar_c = [CC if v>38 else AM if v>33 else GR for v in cs_svc["נטישה"]]
        fig = go.Figure([go.Bar(
            x=cs_svc["נטישה"], y=cs_svc["שירות"],
            orientation="h", marker_color=bar_c, marker_line_width=0,
            text=[f"{v}%" for v in cs_svc["נטישה"]], textposition="outside",
            customdata=list(zip(
                cs_svc["שירות"].tolist(),
                [f"{v}%" for v in cs_svc["נטישה"]],
            )),
            hovertemplate=_hover_tpl(
                "סוג שירות: %{customdata[0]}",
                "שיעור נטישה: <span dir='ltr'>%{customdata[1]}</span>",
            ),
        )])
        fig = fig_base(fig, lm=130, rm=10, bm=10)
        fig.update_layout(legend=dict(visible=False),
                          xaxis_title="שיעור נטישה (%)",
                          yaxis_title="סוג שירות",
                          xaxis_range=[0, 58])
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown('<p class="sec-title">שיעור נטישה לפי עיר</p>', unsafe_allow_html=True)
        cs_city = (F.groupby("עיר")["churn"].mean()
                   .mul(100).round(1).reset_index().sort_values("churn"))
        cs_city.columns = ["עיר","נטישה"]
        bar_c2 = [CC if v>42 else AM if v>35 else GR for v in cs_city["נטישה"]]
        fig = go.Figure([go.Bar(
            x=cs_city["נטישה"], y=cs_city["עיר"],
            orientation="h", marker_color=bar_c2, marker_line_width=0,
            text=[f"{v}%" for v in cs_city["נטישה"]], textposition="outside",
            customdata=list(zip(
                cs_city["עיר"].tolist(),
                [f"{v}%" for v in cs_city["נטישה"]],
            )),
            hovertemplate=_hover_tpl(
                "עיר: %{customdata[0]}",
                "שיעור נטישה: <span dir='ltr'>%{customdata[1]}</span>",
            ),
        )])
        fig = fig_base(fig, lm=110, rm=10, bm=10)
        fig.update_layout(legend=dict(visible=False),
                          xaxis_title="שיעור נטישה (%)",
                          yaxis_title="עיר",
                          xaxis_range=[0, 70])
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap — full width, enough left margin for Hebrew labels
    st.markdown('<p class="sec-title">מפת חום — שיעור נטישה (%) לפי סוג שירות × עיר</p>',
                unsafe_allow_html=True)
    st.caption("ירוק = נטישה נמוכה | צהוב = בינוני | אדום = נטישה גבוהה")
    hm = (df.groupby(["סוג_שירות","עיר"])["churn"]
          .mean().mul(100).round(1).unstack(fill_value=0))
    # Build customdata as list of lists of tuples (service, city, pct)
    hm_cd = [
        [[svc, city, f"{hm.loc[svc, city]:.1f}%"] for city in hm.columns]
        for svc in hm.index
    ]
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=hm.columns.tolist(), y=hm.index.tolist(),
        colorscale=[[0,"#dcfce7"],[0.4,"#fef3c7"],[1,"#dc2626"]],
        text=[[f"{v:.0f}%" for v in row] for row in hm.values],
        texttemplate="%{text}",
        zmin=0, zmax=65,
        colorbar=dict(title="נטישה %", ticksuffix="%", len=0.8),
        customdata=hm_cd,
        hovertemplate=_hover_tpl(
            "סוג שירות: %{customdata[0]}",
            "עיר: %{customdata[1]}",
            "שיעור נטישה: <span dir='ltr'>%{customdata[2]}</span>",
        ),
    ))
    fig.update_layout(
        height=340,
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Arial", size=12),
        margin=dict(l=150, r=80, t=20, b=80),
        xaxis=dict(title="עיר", tickangle=-35, automargin=True,
                   tickfont=dict(size=11)),
        yaxis=dict(title="סוג שירות", automargin=True,
                   tickfont=dict(size=12)),
        hoverlabel=HOVER_LABEL,
    )
    st.plotly_chart(fig, use_container_width=True)

    ce,cf = st.columns(2)

    with ce:
        st.markdown('<p class="sec-title">נטישה לפי ותק לקוח (חודשים עד עזיבה)</p>',
                    unsafe_allow_html=True)
        dc = df[df["churn"]==1].copy()
        dc["ותק"] = (dc["tenure_days"]/30).round(0).astype(int)
        bins = pd.cut(dc["ותק"], bins=[0,3,6,12,24,48,200],
                      labels=["0–3","3–6","6–12","12–24","24–48","48+"])
        td = bins.value_counts().sort_index().reset_index()
        td.columns = ["ותק_חודשים","נוטשים"]
        fig = go.Figure([go.Bar(
            x=td["ותק_חודשים"].astype(str), y=td["נוטשים"],
            marker_color=CC, marker_line_width=0,
            text=td["נוטשים"], textposition="outside",
            customdata=list(zip(
                td["ותק_חודשים"].astype(str).tolist(),
                td["נוטשים"].astype(str).tolist(),
            )),
            hovertemplate=_hover_tpl(
                "ותק (חודשים): <span dir='ltr'>%{customdata[0]}</span>",
                "מספר נוטשים: <span dir='ltr'>%{customdata[1]}</span>",
            ),
        )])
        fig = fig_base(fig, bm=50)
        fig.update_layout(legend=dict(visible=False),
                          xaxis_title="ותק לקוח (חודשים)",
                          yaxis_title="מספר נוטשים")
        st.plotly_chart(fig, use_container_width=True)

    with cf:
        st.markdown('<p class="sec-title">שיעור נטישה לפי קבוצת גיל</p>',
                    unsafe_allow_html=True)
        df2 = F.copy()
        df2["גיל_קבוצה"] = pd.cut(df2["גיל"], bins=[24,34,44,54,64,73],
                                   labels=["25–34","35–44","45–54","55–64","65–72"])
        ac = (df2.groupby("גיל_קבוצה", observed=True)["churn"]
              .mean().mul(100).round(1).reset_index())
        ac.columns = ["גיל","נטישה"]
        fig = go.Figure([go.Scatter(
            x=ac["גיל"].astype(str), y=ac["נטישה"],
            mode="lines+markers",
            line=dict(color=CC, width=2.5), marker=dict(size=10),
            customdata=list(zip(
                ac["גיל"].astype(str).tolist(),
                [f"{v}%" for v in ac["נטישה"]],
            )),
            hovertemplate=_hover_tpl(
                "קבוצת גיל: %{customdata[0]}",
                "שיעור נטישה: <span dir='ltr'>%{customdata[1]}</span>",
            ),
        )])
        fig = fig_base(fig, bm=50)
        fig.update_layout(legend=dict(visible=False),
                          xaxis_title="קבוצת גיל",
                          yaxis_title="שיעור נטישה (%)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Segmentation
# ──────────────────────────────────────────────────────────────────────────────
with t3:
    # ── INSIGHTS FIRST ──
    st.markdown('<p class="sec-title">💡 ממצאים ופעולות מומלצות</p>', unsafe_allow_html=True)
    action("green",
           f"💰 עיר הכנסה מובילה: {top_rev_city} — ₪{top_rev_val:,.0f}/חודש",
           f"פעולה: הגדל נוכחות פיזית ושיווקית ב{top_rev_city} — שוק עם פוטנציאל הגדלת תיק. שקול אירוע לקוחות אזורי.")

    st.markdown("<br>", unsafe_allow_html=True)

    ca,cb = st.columns(2)

    with ca:
        st.markdown('<p class="sec-title">ממוצע שווי תיק — פעיל vs נטש, לפי שירות</p>',
                    unsafe_allow_html=True)
        ps = (F.groupby(["סוג_שירות","סטטוס"])["סכום_תיק"]
              .mean().round(0).reset_index())
        ps.columns = ["שירות","סטטוס","ממוצע_תיק"]
        fig = go.Figure()
        for status, color in [("פעיל", CA),("לא פעיל", CC)]:
            d = ps[ps["סטטוס"]==status].copy()
            fig.add_trace(go.Bar(
                name=status, x=d["שירות"], y=d["ממוצע_תיק"],
                marker_color=color, marker_line_width=0,
                customdata=list(zip(
                    d["שירות"].tolist(),
                    [status]*len(d),
                    [f"₪{v/1000:.0f}K" for v in d["ממוצע_תיק"]],
                )),
                hovertemplate=_hover_tpl(
                    "סוג שירות: %{customdata[0]}",
                    "סטטוס: %{customdata[1]}",
                    "ממוצע שווי תיק: <span dir='ltr'>%{customdata[2]}</span>",
                ),
            ))
        fig = fig_base(fig, bm=70, xangle=-35)
        fig.update_layout(barmode="group",
                          xaxis_title="סוג שירות",
                          yaxis_title="ממוצע שווי תיק (₪)")
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown('<p class="sec-title">מפת עץ — הכנסה חודשית לפי עיר וסוג שירות</p>',
                    unsafe_allow_html=True)
        tree = (F[F["סטטוס"]=="פעיל"]
                .groupby(["עיר","סוג_שירות"])["הכנסה_חודשית"].sum().reset_index())
        tree.columns = ["עיר","שירות","הכנסה"]
        if not tree.empty:
            tree["הכנסה_fmt"] = tree["הכנסה"].apply(lambda v: f"₪{v/1000:.0f}K")
            fig = px.treemap(
                tree, path=["עיר","שירות"], values="הכנסה",
                color="הכנסה",
                color_continuous_scale=[[0,"#dbeafe"],[1,"#1d4ed8"]],
                custom_data=["עיר","שירות","הכנסה_fmt"],
            )
            fig.update_traces(
                hovertemplate=_hover_tpl(
                    "עיר: %{customdata[0]}",
                    "סוג שירות: %{customdata[1]}",
                    "הכנסה חודשית: <span dir='ltr'>%{customdata[2]}</span>",
                )
            )
            fig.update_layout(height=340, margin=dict(l=0,r=0,t=0,b=0),
                              coloraxis_showscale=False,
                              hoverlabel=HOVER_LABEL)
            st.plotly_chart(fig, use_container_width=True)

    cc3,cd3 = st.columns(2)

    with cc3:
        st.markdown('<p class="sec-title">גידול בסיס לקוחות לפי שנת הצטרפות</p>',
                    unsafe_allow_html=True)
        coh = (F.groupby(["שנת_הצטרפות","סטטוס"]).size().reset_index(name="לקוחות"))
        fig = go.Figure()
        for status, color in [("פעיל", CA),("לא פעיל", CC)]:
            d = coh[coh["סטטוס"]==status].copy()
            fig.add_trace(go.Bar(
                name=status, x=d["שנת_הצטרפות"], y=d["לקוחות"],
                marker_color=color, marker_line_width=0,
                customdata=list(zip(
                    d["שנת_הצטרפות"].astype(str).tolist(),
                    [status]*len(d),
                    d["לקוחות"].astype(str).tolist(),
                )),
                hovertemplate=_hover_tpl(
                    "שנת הצטרפות: <span dir='ltr'>%{customdata[0]}</span>",
                    "סטטוס: %{customdata[1]}",
                    "מספר לקוחות: <span dir='ltr'>%{customdata[2]}</span>",
                ),
            ))
        fig = fig_base(fig, bm=50)
        fig.update_layout(barmode="stack",
                          xaxis_title="שנת הצטרפות",
                          yaxis_title="מספר לקוחות",
                          xaxis=dict(tickmode="linear"))
        st.plotly_chart(fig, use_container_width=True)

    with cd3:
        st.markdown('<p class="sec-title">שיעור נטישה לפי מגדר וסוג שירות</p>',
                    unsafe_allow_html=True)
        gs = (F.groupby(["סוג_שירות","מגדר"])["churn"]
              .mean().mul(100).round(1).reset_index())
        gs.columns = ["שירות","מגדר","נטישה"]
        fig = go.Figure()
        for gender, color in [("זכר","#6366f1"),("נקבה","#ec4899")]:
            d = gs[gs["מגדר"]==gender].copy()
            fig.add_trace(go.Bar(
                name=gender, x=d["שירות"], y=d["נטישה"],
                marker_color=color, marker_line_width=0,
                customdata=list(zip(
                    d["שירות"].tolist(),
                    [gender]*len(d),
                    [f"{v}%" for v in d["נטישה"]],
                )),
                hovertemplate=_hover_tpl(
                    "סוג שירות: %{customdata[0]}",
                    "מגדר: %{customdata[1]}",
                    "שיעור נטישה: <span dir='ltr'>%{customdata[2]}</span>",
                ),
            ))
        fig = fig_base(fig, bm=70, xangle=-35)
        fig.update_layout(barmode="group",
                          xaxis_title="סוג שירות",
                          yaxis_title="שיעור נטישה (%)")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — At-Risk Clients
# ──────────────────────────────────────────────────────────────────────────────
with t4:
    at_risk = filt(df[df["סטטוס"]=="פעיל"].copy()).sort_values("risk_score", ascending=False)
    high = int((at_risk["risk_label"]=="גבוה").sum())
    mid  = int((at_risk["risk_label"]=="בינוני").sum())
    low  = int((at_risk["risk_label"]=="נמוך").sum())
    rev_high = at_risk[at_risk["risk_label"]=="גבוה"]["הכנסה_חודשית"].sum()

    # ── INSIGHTS FIRST ──
    if high > 0:
        top1 = at_risk.iloc[0]
        action("red",
               "🚨 המלצת פעולה מיידית — הלקוח בסיכון הגבוה ביותר",
               f"<b>{top1['שם']}</b> | {top1['עיר']} | {top1['סוג_שירות']} | "
               f"ציון סיכון: <b>{int(top1['risk_score'])}/100</b><br>"
               f"שביעות רצון: {top1['שביעות_רצון']}/10 · "
               f"זמן תגובה: {top1['זמן_תגובה_ממוצע_שעות']:.0f} שעות · "
               f"פניות/שנה: {int(top1['מספר_פניות_שנה_אחרונה'])}<br>"
               f"<b>פעולה:</b> התקשר אישית תוך 24 שעות. הצע שדרוג חינמי או פגישת ייעוץ.")

    st.markdown("<br>", unsafe_allow_html=True)

    k1,k2,k3 = st.columns(3)
    k1.metric("🔴 סיכון גבוה",   f"{high} לקוחות",
              delta=f"₪{rev_high:,.0f}/חודש בסיכון", delta_color="inverse")
    k2.metric("🟡 סיכון בינוני", f"{mid} לקוחות")
    k3.metric("🟢 סיכון נמוך",   f"{low} לקוחות")

    st.markdown("<br>", unsafe_allow_html=True)

    ca,cb = st.columns([1,2])

    with ca:
        st.markdown('<p class="sec-title">התפלגות רמות סיכון</p>', unsafe_allow_html=True)
        pie = pd.DataFrame({"רמה":["גבוה","בינוני","נמוך"],"כמות":[high,mid,low]})
        fig = go.Figure([go.Pie(
            labels=pie["רמה"], values=pie["כמות"], hole=0.52,
            marker_colors=[CC, AM, GR],
            textinfo="percent+label",
            textfont_size=13,
            customdata=list(zip(
                pie["רמה"].tolist(),
                pie["כמות"].astype(str).tolist(),
            )),
            hovertemplate=_hover_tpl(
                "רמת סיכון: %{customdata[0]}",
                "מספר לקוחות: <span dir='ltr'>%{customdata[1]}</span>",
                "אחוז: <span dir='ltr'>%{percent}</span>",
            ),
        )])
        fig.update_layout(height=290, paper_bgcolor="#fff",
                          font=dict(family="Arial"),
                          margin=dict(l=10,r=10,t=10,b=10),
                          showlegend=False,
                          hoverlabel=HOVER_LABEL)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown('<p class="sec-title">ציון סיכון ממוצע לפי סוג שירות</p>', unsafe_allow_html=True)
        rs = (at_risk.groupby("סוג_שירות")["risk_score"]
              .mean().round(1).reset_index().sort_values("risk_score"))
        rs.columns = ["שירות","ציון"]
        bar_c = [CC if v>=50 else AM if v>=35 else GR for v in rs["ציון"]]
        fig = go.Figure([go.Bar(
            x=rs["ציון"], y=rs["שירות"],
            orientation="h", marker_color=bar_c, marker_line_width=0,
            text=[f"{v:.0f}" for v in rs["ציון"]], textposition="outside",
            customdata=list(zip(
                rs["שירות"].tolist(),
                [f"{v:.1f} / 100" for v in rs["ציון"]],
            )),
            hovertemplate=_hover_tpl(
                "סוג שירות: %{customdata[0]}",
                "ציון סיכון ממוצע: <span dir='ltr'>%{customdata[1]}</span>",
            ),
        )])
        fig = fig_base(fig, lm=130, rm=10, bm=10, h=290)
        fig.update_layout(legend=dict(visible=False),
                          xaxis_title="ציון סיכון ממוצע (0–100)",
                          yaxis_title="סוג שירות",
                          xaxis_range=[0, 48])
        st.plotly_chart(fig, use_container_width=True)

    # Top 20 table
    st.markdown('<p class="sec-title">טופ 20 לקוחות בסיכון גבוה ביותר</p>', unsafe_allow_html=True)
    top20 = at_risk.nlargest(20,"risk_score")[[
        "client_id","שם","עיר","סוג_שירות",
        "שביעות_רצון","זמן_תגובה_ממוצע_שעות",
        "סכום_תיק","הכנסה_חודשית",
        "מספר_פניות_שנה_אחרונה","risk_score","risk_label",
    ]].rename(columns={
        "client_id":"מזהה","risk_score":"ציון סיכון","risk_label":"רמה",
        "שביעות_רצון":"שב\"ר","זמן_תגובה_ממוצע_שעות":"זמן תגובה",
        "סכום_תיק":"שווי תיק","הכנסה_חודשית":"הכנסה/חודש",
        "מספר_פניות_שנה_אחרונה":"פניות/שנה",
    })

    def col_risk(v):
        if v >= 60: return "background-color:#fee2e2;color:#dc2626;font-weight:700"
        if v >= 30: return "background-color:#fef3c7;color:#d97706;font-weight:600"
        return "background-color:#dcfce7;color:#16a34a"

    def col_sat(v):
        if v <= 3: return "color:#dc2626;font-weight:700"
        if v <= 5: return "color:#d97706"
        return "color:#16a34a"

    styled = (top20.style
              .map(col_risk, subset=["ציון סיכון"])
              .map(col_sat,  subset=["שב\"ר"])
              .format({"שווי תיק":"₪{:,.0f}","הכנסה/חודש":"₪{:,.0f}",
                       "זמן תגובה":"{:.1f}"}))
    st.dataframe(styled, use_container_width=True, height=480)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Export
# ──────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<p class="sec-title">📋 טבלת לקוחות מלאה</p>', unsafe_allow_html=True)
    show_cols = ["client_id","שם","גיל","מגדר","עיר","סוג_שירות",
                 "תאריך_הצטרפות","סכום_תיק","הכנסה_חודשית",
                 "שביעות_רצון","זמן_תגובה_ממוצע_שעות","סטטוס"]
    disp = F[show_cols].rename(columns={
        "client_id":"מזהה","סכום_תיק":"שווי תיק (₪)",
        "הכנסה_חודשית":"הכנסה/חודש (₪)","שביעות_רצון":"שב\"ר",
        "זמן_תגובה_ממוצע_שעות":"זמן תגובה (שעות)",
        "תאריך_הצטרפות":"תאריך הצטרפות",
    })
    st.dataframe(
        disp.style.format({"שווי תיק (₪)":"₪{:,.0f}","הכנסה/חודש (₪)":"₪{:,.0f}"}),
        use_container_width=True, height=360,
    )

    st.divider()
    st.markdown('<p class="sec-title">🔧 תיקון מספרי לקוח — ייצוא Excel</p>',
                unsafe_allow_html=True)
    st.info("מספרי הלקוח אינם לפי סדר הצטרפות. "
            "הכפתור מייצר Excel מתוקן: C1000 = ראשון שהצטרף, C2999 = אחרון.")

    @st.cache_data
    def make_excel(source_df: pd.DataFrame) -> bytes:
        drop = ["churn","tenure_days","risk_score","risk_label","שנת_הצטרפות"]
        fixed = source_df.sort_values("תאריך_הצטרפות").reset_index(drop=True)
        fixed["client_id"] = [f"C{1000+i}" for i in range(len(fixed))]
        cols = [c for c in fixed.columns if c not in drop]
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            fixed[cols].to_excel(w, index=False, sheet_name="לקוחות_מתוקן")
        return buf.getvalue()

    st.download_button(
        "📥 ייצוא קובץ Excel עם מספרי לקוח מתוקנים",
        data=make_excel(df),
        file_name="clients_fixed_ids.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    with st.expander("👁️ תצוגה מקדימה — 5 שורות ראשונות בקובץ המתוקן"):
        prev = df.sort_values("תאריך_הצטרפות").reset_index(drop=True).head(5).copy()
        prev["client_id"] = [f"C{1000+i}" for i in range(5)]
        st.dataframe(prev[["client_id","שם","תאריך_הצטרפות","עיר","סוג_שירות","סטטוס"]],
                     use_container_width=True)
