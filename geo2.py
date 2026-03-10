import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import pandas as pd

# ─────────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stabilité des Pentes",
    page_icon="⛰️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; color: #e0e0e0; }

    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

    .title-block {
        background: linear-gradient(90deg, #1e3a5f, #0d2137);
        border-left: 5px solid #3b9eff;
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 25px;
    }
    .title-block h1 { color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700; }
    .title-block p  { color: #8bb8e8; margin: 5px 0 0 0; font-size: 0.95rem; }

    .metric-card {
        background: linear-gradient(135deg, #1e2d45, #162236);
        border: 1px solid #2a4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(59,158,255,0.1);
    }
    .metric-card .label { color: #8bb8e8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 8px 0; }
    .metric-card .unit  { color: #3b9eff; font-size: 0.8rem; }

    .fos-safe    { color: #2ecc71 !important; }
    .fos-warning { color: #f39c12 !important; }
    .fos-danger  { color: #e74c3c !important; }

    .section-title {
        color: #3b9eff;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }
    .info-box {
        background: #1a2a3a;
        border-left: 4px solid #3b9eff;
        padding: 12px 16px;
        border-radius: 6px;
        color: #ccd9e8;
        font-size: 0.88rem;
        margin: 10px 0;
    }
    .stSlider > div > div > div { background: #3b9eff !important; }
    div[data-testid="stSidebar"] { background: #0d1623 !important; }
    div[data-testid="stSidebar"] .stMarkdown { color: #8bb8e8; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MODULE GEOMETRY
# ══════════════════════════════════════════════════════════════
def creer_profil_talus(H, beta_deg, L_base):
    beta_rad = np.radians(beta_deg)
    L_pente  = H / np.tan(beta_rad)
    A = (0.0,        0.0)
    B = (L_base/3,   0.0)
    C = (L_base/3 + L_pente, H)
    D = (L_base,     H)
    return [A[0], B[0], C[0], D[0]], [A[1], B[1], C[1], D[1]]


def creer_tranches(xc, yc, R, x_profil, y_profil, n_tranches=20):
    discriminant = R**2 - yc**2
    if discriminant < 0:
        return []
    x_gauche = xc - np.sqrt(discriminant)
    x_droite = xc + np.sqrt(discriminant)
    x_bords  = np.linspace(x_gauche, x_droite, n_tranches + 1)
    b        = x_bords[1] - x_bords[0]
    tranches = []
    for i in range(n_tranches):
        x_mid = (x_bords[i] + x_bords[i+1]) / 2
        val   = R**2 - (x_mid - xc)**2
        if val < 0:
            continue
        y_base    = yc - np.sqrt(val)
        y_surface = np.interp(x_mid, x_profil, y_profil)
        h = y_surface - y_base
        if h <= 0:
            continue
        sin_alpha = np.clip((x_mid - xc) / R, -1, 1)
        alpha_rad = np.arcsin(sin_alpha)
        tranches.append({
            "x_milieu": x_mid,  "b": b,           "h": h,
            "alpha_rad": alpha_rad, "alpha_deg": np.degrees(alpha_rad),
            "y_base": y_base,   "y_surface": y_surface,
        })
    return tranches


# ══════════════════════════════════════════════════════════════
#  MODULE MATERIAL
# ══════════════════════════════════════════════════════════════
def creer_materiau(gamma, c, phi_deg):
    return {"gamma": gamma, "c": c, "phi_rad": np.radians(phi_deg), "phi_deg": phi_deg}


# ══════════════════════════════════════════════════════════════
#  MODULE ANALYSIS — Bishop Simplifié
# ══════════════════════════════════════════════════════════════
def bishop_simplifie(tranches, materiau, u=0.0, tol=1e-6, max_iter=100):
    """
    Calcule le FoS par la méthode de Bishop Simplifié.
    Équation itérative :
       FoS = Σ[(c'b + (W - ub)tanφ') / mα] / Σ[W sinα]
    avec  mα = cosα + sinα·tanφ'/FoS
    """
    c     = materiau["c"]
    phi   = materiau["phi_rad"]
    gamma = materiau["gamma"]

    # Poids de chaque tranche
    W_list = [t["h"] * t["b"] * gamma for t in tranches]

    # Somme des moments moteurs
    sum_moteur = sum(W * np.sin(t["alpha_rad"]) for W, t in zip(W_list, tranches))
    if sum_moteur <= 0:
        return None

    FoS = 1.5  # valeur initiale
    for _ in range(max_iter):
        sum_resist = 0.0
        for W, t in zip(W_list, tranches):
            b     = t["b"]
            alpha = t["alpha_rad"]
            m_alpha = np.cos(alpha) + np.sin(alpha) * np.tan(phi) / FoS
            if abs(m_alpha) < 1e-10:
                continue
            numerateur  = c * b + (W - u * b) * np.tan(phi)
            sum_resist += numerateur / m_alpha

        FoS_new = sum_resist / sum_moteur
        if abs(FoS_new - FoS) < tol:
            return round(FoS_new, 4)
        FoS = FoS_new

    return round(FoS, 4)


def fellenius(tranches, materiau, u=0.0):
    """
    Méthode de Fellenius (Ordinary Method of Slices).
    FoS = Σ[(c'b + (W cosα - ub)tanφ')] / Σ[W sinα]
    """
    c     = materiau["c"]
    phi   = materiau["phi_rad"]
    gamma = materiau["gamma"]

    sum_resist = 0.0
    sum_moteur = 0.0
    for t in tranches:
        W     = t["h"] * t["b"] * gamma
        alpha = t["alpha_rad"]
        b     = t["b"]
        sum_resist += c * b + (W * np.cos(alpha) - u * b) * np.tan(phi)
        sum_moteur += W * np.sin(alpha)

    if sum_moteur <= 0:
        return None
    return round(sum_resist / sum_moteur, 4)


def chercher_cercle_critique(x_profil, y_profil, materiau, H, L_base, n_tranches=20):
    """
    Cherche le cercle de glissement avec le FoS minimum
    en testant une grille de centres et rayons.
    """
    best_FoS = 1e9
    best_params = None

    xc_range = np.linspace(L_base * 0.2, L_base * 0.7, 8)
    yc_range = np.linspace(H * 0.8, H * 2.5, 8)
    R_range  = np.linspace(H * 1.0, H * 2.5, 6)

    for xc in xc_range:
        for yc in yc_range:
            for R in R_range:
                try:
                    tranches = creer_tranches(xc, yc, R, x_profil, y_profil, n_tranches)
                    if len(tranches) < 3:
                        continue
                    FoS = bishop_simplifie(tranches, materiau)
                    if FoS and 0.5 < FoS < best_FoS:
                        best_FoS    = FoS
                        best_params = (xc, yc, R)
                except:
                    continue

    return best_params, best_FoS


# ══════════════════════════════════════════════════════════════
#  MODULE PLOT
# ══════════════════════════════════════════════════════════════
def plot_talus(x_profil, y_profil, tranches, xc, yc, R, FoS, materiau, H):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f1117')

    # ── Graphique principal : talus + tranches + cercle ──────
    ax = axes[0]
    ax.set_facecolor('#0d1623')

    # Remplissage du sol
    x_fill = x_profil + [x_profil[-1], x_profil[0]]
    y_fill = y_profil + [0, 0]
    ax.fill(x_fill, y_fill, color='#3d5a3e', alpha=0.6, zorder=1)
    ax.plot(x_profil, y_profil, color='#7ec882', linewidth=2.5, zorder=3, label='Profil du talus')

    # Tranches
    for t in tranches:
        x_left  = t["x_milieu"] - t["b"] / 2
        rect = plt.Rectangle(
            (x_left, t["y_base"]), t["b"], t["h"],
            linewidth=0.5, edgecolor='#3b9eff', facecolor='#1a4a7a', alpha=0.4, zorder=4
        )
        ax.add_patch(rect)

    # Arc de cercle (surface de rupture)
    theta = np.linspace(0, 2 * np.pi, 500)
    xc_arr = xc + R * np.cos(theta)
    yc_arr = yc + R * np.sin(theta)
    mask = yc_arr >= -1
    ax.plot(xc_arr[mask], yc_arr[mask], color='#ff4444', linewidth=2.5,
            linestyle='--', zorder=5, label='Surface de rupture')
    ax.plot(xc, yc, 'r+', markersize=12, markeredgewidth=2, zorder=6)

    # Centre du cercle annoté
    ax.annotate(f'Centre\n({xc:.1f}, {yc:.1f})',
                xy=(xc, yc), xytext=(xc + 2, yc + 1.5),
                color='#ff8888', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#ff8888'))

    # FoS coloré selon la valeur
    fos_color = '#2ecc71' if FoS >= 1.5 else '#f39c12' if FoS >= 1.0 else '#e74c3c'
    ax.text(0.05, 0.92, f'FoS (Bishop) = {FoS}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            color=fos_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1623', edgecolor=fos_color))

    ax.set_xlim(-1, max(x_profil) + 2)
    ax.set_ylim(-R * 0.3, yc + R * 0.5)
    ax.set_xlabel('Distance horizontale (m)', color='#8bb8e8')
    ax.set_ylabel('Altitude (m)', color='#8bb8e8')
    ax.set_title('Profil du talus et surface de rupture critique', color='white', fontsize=13, pad=10)
    ax.tick_params(colors='#8bb8e8')
    ax.spines[:].set_color('#2a4a6e')
    ax.legend(facecolor='#0d1623', edgecolor='#2a4a6e', labelcolor='white', fontsize=9)
    ax.grid(True, color='#1e3a5f', linewidth=0.5, alpha=0.7)

    # ── Graphique 2 : sensibilité FoS vs cohésion c ──────────
    ax2 = axes[1]
    ax2.set_facecolor('#0d1623')

    c_values   = np.linspace(1, materiau["c"] * 2 + 5, 30)
    fos_values = []
    for c_val in c_values:
        mat_tmp = creer_materiau(materiau["gamma"], c_val, materiau["phi_deg"])
        try:
            f = bishop_simplifie(tranches, mat_tmp)
            fos_values.append(f if f else np.nan)
        except:
            fos_values.append(np.nan)

    ax2.plot(c_values, fos_values, color='#3b9eff', linewidth=2.5, zorder=3)
    ax2.fill_between(c_values, fos_values, alpha=0.15, color='#3b9eff')
    ax2.axhline(y=1.5, color='#2ecc71', linestyle='--', linewidth=1.5, label='FoS = 1.5 (sécurité)')
    ax2.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='FoS = 1.0 (rupture)')
    ax2.axvline(x=materiau["c"], color='#f39c12', linestyle=':', linewidth=1.5,
                label=f'c actuel = {materiau["c"]} kPa')

    ax2.set_xlabel("Cohésion c (kPa)", color='#8bb8e8')
    ax2.set_ylabel("Facteur de sécurité (FoS)", color='#8bb8e8')
    ax2.set_title("Sensibilité du FoS à la cohésion", color='white', fontsize=13, pad=10)
    ax2.tick_params(colors='#8bb8e8')
    ax2.spines[:].set_color('#2a4a6e')
    ax2.legend(facecolor='#0d1623', edgecolor='#2a4a6e', labelcolor='white', fontsize=9)
    ax2.grid(True, color='#1e3a5f', linewidth=0.5, alpha=0.7)

    plt.tight_layout(pad=2.5)
    return fig


def plot_sensibilite_phi(tranches, materiau):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0d1623')

    phi_values = np.linspace(5, 45, 30)
    fos_bishop = []
    fos_fell   = []

    for phi_val in phi_values:
        mat_tmp = creer_materiau(materiau["gamma"], materiau["c"], phi_val)
        try:
            fb = bishop_simplifie(tranches, mat_tmp)
            ff = fellenius(tranches, mat_tmp)
            fos_bishop.append(fb if fb else np.nan)
            fos_fell.append(ff if ff else np.nan)
        except:
            fos_bishop.append(np.nan)
            fos_fell.append(np.nan)

    ax.plot(phi_values, fos_bishop, color='#3b9eff', linewidth=2.5, label='Bishop Simplifié')
    ax.plot(phi_values, fos_fell,   color='#f39c12', linewidth=2.5, label='Fellenius', linestyle='--')
    ax.axhline(y=1.5, color='#2ecc71', linestyle=':', linewidth=1.5, label='Seuil sécurité (1.5)')
    ax.axhline(y=1.0, color='#e74c3c', linestyle=':', linewidth=1.5, label='Rupture (1.0)')
    ax.axvline(x=materiau["phi_deg"], color='#ffffff', linestyle=':', linewidth=1, alpha=0.5,
               label=f'φ actuel = {materiau["phi_deg"]}°')

    ax.set_xlabel("Angle de frottement φ (°)", color='#8bb8e8')
    ax.set_ylabel("Facteur de sécurité (FoS)", color='#8bb8e8')
    ax.set_title("Sensibilité du FoS à l'angle de frottement", color='white', fontsize=12)
    ax.tick_params(colors='#8bb8e8')
    ax.spines[:].set_color('#2a4a6e')
    ax.legend(facecolor='#0d1623', edgecolor='#2a4a6e', labelcolor='white', fontsize=9)
    ax.grid(True, color='#1e3a5f', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
#  INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════

# En-tête
st.markdown("""
<div class="title-block">
    <h1>⛰️ Analyse de Stabilité des Pentes</h1>
    <p>Méthode de Bishop Simplifié · Fellenius · Facteur de Sécurité · Visualisation Interactive</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR : Paramètres ──────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres du modèle")

    st.markdown('<p class="section-title">📐 Géométrie du talus</p>', unsafe_allow_html=True)
    H      = st.slider("Hauteur H (m)",          5.0,  30.0, 10.0, 0.5)
    beta   = st.slider("Angle de pente β (°)",   15.0, 60.0, 30.0, 1.0)
    L_base = st.slider("Longueur de base L (m)", 20.0, 80.0, 40.0, 1.0)

    st.markdown('<p class="section-title">🪨 Propriétés du sol</p>', unsafe_allow_html=True)
    gamma  = st.slider("Poids volumique γ (kN/m³)", 14.0, 22.0, 18.0, 0.5)
    c      = st.slider("Cohésion c (kPa)",            0.0, 50.0, 15.0, 0.5)
    phi    = st.slider("Angle de frottement φ (°)",   5.0, 45.0, 25.0, 1.0)

    st.markdown('<p class="section-title">💧 Conditions hydrauliques</p>', unsafe_allow_html=True)
    u = st.slider("Pression interstitielle u (kPa)", 0.0, 20.0, 0.0, 0.5)

    st.markdown('<p class="section-title">🔧 Options de calcul</p>', unsafe_allow_html=True)
    n_tranches     = st.slider("Nombre de tranches", 5, 30, 15, 1)
    chercher_crit  = st.checkbox("🔍 Chercher le cercle critique automatiquement", value=True)

    st.markdown("---")
    st.markdown('<div class="info-box">💡 Le FoS > 1.5 indique une pente stable.<br>FoS < 1.0 = rupture imminente.</div>',
                unsafe_allow_html=True)

# ── CALCUL ────────────────────────────────────────────────────
x_profil, y_profil = creer_profil_talus(H, beta, L_base)
materiau            = creer_materiau(gamma, c, phi)

with st.spinner("🔄 Calcul en cours..."):
    if chercher_crit:
        best_params, best_FoS = chercher_cercle_critique(
            x_profil, y_profil, materiau, H, L_base, n_tranches
        )
        if best_params:
            xc_opt, yc_opt, R_opt = best_params
        else:
            xc_opt = L_base * 0.45
            yc_opt = H * 1.5
            R_opt  = H * 1.8
    else:
        xc_opt = L_base * 0.45
        yc_opt = H * 1.5
        R_opt  = H * 1.8

    tranches = creer_tranches(xc_opt, yc_opt, R_opt, x_profil, y_profil, n_tranches)

    FoS_bishop = bishop_simplifie(tranches, materiau, u)
    FoS_fell   = fellenius(tranches, materiau, u)

# ── MÉTRIQUES ─────────────────────────────────────────────────
def fos_class(v):
    if v is None: return ""
    return "fos-safe" if v >= 1.5 else "fos-warning" if v >= 1.0 else "fos-danger"

def fos_label(v):
    if v is None: return "—"
    return "✅ Stable" if v >= 1.5 else "⚠️ Limite" if v >= 1.0 else "❌ Rupture"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">FoS — Bishop</div>
        <div class="value {fos_class(FoS_bishop)}">{FoS_bishop if FoS_bishop else '—'}</div>
        <div class="unit">{fos_label(FoS_bishop)}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">FoS — Fellenius</div>
        <div class="value {fos_class(FoS_fell)}">{FoS_fell if FoS_fell else '—'}</div>
        <div class="unit">{fos_label(FoS_fell)}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Nombre de tranches</div>
        <div class="value" style="color:#3b9eff">{len(tranches)}</div>
        <div class="unit">tranches actives</div>
    </div>""", unsafe_allow_html=True)

with col4:
    ecart = ""
    if FoS_bishop and FoS_fell:
        diff = abs(FoS_bishop - FoS_fell)
        ecart = f"{diff:.4f}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Écart Bishop / Fell.</div>
        <div class="value" style="color:#f39c12">{ecart if ecart else '—'}</div>
        <div class="unit">différence FoS</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── GRAPHIQUES ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Talus & Surface de rupture", "📈 Sensibilité à φ", "📋 Détail des tranches"])

with tab1:
    if tranches and FoS_bishop:
        fig = plot_talus(x_profil, y_profil, tranches, xc_opt, yc_opt, R_opt, FoS_bishop, materiau, H)
        st.pyplot(fig, use_container_width=True)
    else:
        st.error("⚠️ Impossible de générer le graphique. Vérifiez les paramètres.")

with tab2:
    if tranches:
        fig2 = plot_sensibilite_phi(tranches, materiau)
        st.pyplot(fig2, use_container_width=True)

with tab3:
    if tranches:
        df = pd.DataFrame([{
            "N° tranche":  i + 1,
            "x_milieu (m)": round(t["x_milieu"], 3),
            "b (m)":        round(t["b"], 3),
            "h (m)":        round(t["h"], 3),
            "W (kN/m)":     round(t["h"] * t["b"] * gamma, 3),
            "α (°)":        round(t["alpha_deg"], 2),
            "y_base (m)":   round(t["y_base"], 3),
        } for i, t in enumerate(tranches)])

        st.dataframe(
            df.style.background_gradient(subset=["W (kN/m)"], cmap="Blues")
                    .background_gradient(subset=["h (m)"], cmap="Greens"),
            use_container_width=True, height=400
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger les données (.csv)", csv,
                           "tranches.csv", "text/csv")

# ── RÉSUMÉ PARAMÈTRES ─────────────────────────────────────────
with st.expander("📌 Résumé des paramètres utilisés"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Géométrie**")
        st.write(f"- Hauteur H : **{H} m**")
        st.write(f"- Angle β : **{beta}°**")
        st.write(f"- Base L : **{L_base} m**")
    with col_b:
        st.markdown("**Sol**")
        st.write(f"- γ : **{gamma} kN/m³**")
        st.write(f"- c : **{c} kPa**")
        st.write(f"- φ : **{phi}°**")
    with col_c:
        st.markdown("**Cercle critique**")
        st.write(f"- Centre xc : **{xc_opt:.2f} m**")
        st.write(f"- Centre yc : **{yc_opt:.2f} m**")
        st.write(f"- Rayon R : **{R_opt:.2f} m**")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#2a4a6e; font-size:0.8rem;">'
    'Outil de Stabilité des Pentes · Bishop Simplifié · Geotechnique 2025</p>',
    unsafe_allow_html=True
)