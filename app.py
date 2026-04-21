import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.4rem; font-weight:700; color:#4f46e5; margin-bottom:0.2rem;}
    .sub-title  {font-size:1rem; color:#6b7280; margin-bottom:1.5rem;}
    .metric-card{background:#f3f4f6; border-radius:12px; padding:1rem 1.2rem; text-align:center;}
    .metric-val {font-size:1.8rem; font-weight:700; color:#4f46e5;}
    .metric-lbl {font-size:0.8rem; color:#6b7280;}
    .section-header{font-size:1.2rem; font-weight:600; color:#1f2937; border-left:4px solid #4f46e5;
                    padding-left:0.6rem; margin:1.2rem 0 0.6rem 0;}
</style>
""", unsafe_allow_html=True)


# ── Data loading & caching ───────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# ── Cleaning pipeline ────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = df.copy()

    # 1. Standardise column names
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_"))
    log.append("✅ Column names normalised (lower-snake-case).")

    # 2. Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    log.append(f"✅ Duplicate rows removed: {removed}.")

    # 3. Drop rows where ALL values are NaN
    df.dropna(how="all", inplace=True)
    log.append("✅ All-NaN rows dropped.")

    # 4. Fill remaining numeric NaNs with column median
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        n_miss = df[col].isna().sum()
        if n_miss:
            df[col].fillna(df[col].median(), inplace=True)
            log.append(f"✅ '{col}': {n_miss} missing → filled with median.")

    # 5. Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        n_miss = df[col].isna().sum()
        if n_miss:
            df[col].fillna(df[col].mode()[0], inplace=True)
            log.append(f"✅ '{col}': {n_miss} missing → filled with mode.")

    # 6. Remove age / score outliers (IQR on numeric non-ID cols)
    id_col = [c for c in df.columns if "id" in c]
    num_feat = [c for c in num_cols if c not in id_col]
    before = len(df)
    for col in num_feat:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)]
    log.append(f"✅ Outliers removed (3×IQR): {before - len(df)} rows dropped.")

    # 7. Encode gender
    if "gender" in df.columns:
        df["gender_enc"] = LabelEncoder().fit_transform(df["gender"].str.strip().str.title())
        log.append("✅ 'gender' encoded → 'gender_enc' (0/1).")

    log.append(f"✅ Clean dataset shape: {df.shape[0]} rows × {df.shape[1]} cols.")
    return df, log


# ── Feature selection helper ─────────────────────────────────────────────────
FEATURE_PRESETS = {
    "Income + Spending (classic)": ["annual_income_k_", "spending_score_1_100_"],
    "Age + Spending":               ["age", "spending_score_1_100_"],
    "Age + Income":                 ["age", "annual_income_k_"],
    "All numeric features":         ["age", "annual_income_k_", "spending_score_1_100_"],
}


# ── Cluster label helper ─────────────────────────────────────────────────────
CLUSTER_DESCRIPTIONS = {
    0: "Segment A", 1: "Segment B", 2: "Segment C",
    3: "Segment D", 4: "Segment E", 5: "Segment F",
}


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-mall.png", width=64)
    st.title("⚙️ Controls")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        raw_df = load_data(uploaded)
        st.success(f"Loaded {len(raw_df)} rows")
    else:
        raw_df = None

    st.divider()
    st.markdown("**Feature preset**")
    preset_name = st.selectbox("Choose features for clustering",
                                list(FEATURE_PRESETS.keys()))

    st.markdown("**Number of clusters  K**")
    k = st.slider("K", min_value=2, max_value=10, value=5)

    st.markdown("**KMeans settings**")
    init_method = st.selectbox("Init method", ["k-means++", "random"])
    max_iter    = st.slider("Max iterations", 100, 500, 300, step=50)
    n_init      = st.slider("n_init (restarts)", 5, 20, 10)

    run_btn = st.button("🚀 Run Clustering", use_container_width=True,
                         disabled=(raw_df is None))


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🛍️ Mall Customer Segmentation</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">K-Means clustering with interactive controls & cleaning pipeline</div>',
            unsafe_allow_html=True)

if raw_df is None:
    st.info("👈  Upload `Mall_Customers.csv` from the sidebar to get started.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_data, tab_clean, tab_eda, tab_elbow, tab_cluster, tab_export = st.tabs([
    "📄 Raw Data", "🧹 Cleaning", "📊 EDA", "📈 Elbow", "🎯 Clusters", "💾 Export"
])

# ── Tab 1 : Raw data ──────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df, use_container_width=True, height=320)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing values", int(raw_df.isnull().sum().sum()))

# ── Clean data (always) ───────────────────────────────────────────────────────
df_clean, clean_log = clean_data(raw_df)

# ── Tab 2 : Cleaning log ──────────────────────────────────────────────────────
with tab_clean:
    st.markdown('<div class="section-header">Cleaning Pipeline Log</div>', unsafe_allow_html=True)
    for msg in clean_log:
        st.write(msg)
    st.divider()
    st.markdown('<div class="section-header">Cleaned Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_clean, use_container_width=True, height=300)

# ── Tab 3 : EDA ───────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown('<div class="section-header">Exploratory Analysis</div>', unsafe_allow_html=True)

    num_cols = df_clean.select_dtypes(include="number").columns.tolist()
    num_cols_noid = [c for c in num_cols if "id" not in c and "enc" not in c]

    col_a, col_b = st.columns(2)

    # Distribution plots
    with col_a:
        st.markdown("**Feature distributions**")
        fig, axes = plt.subplots(1, len(num_cols_noid), figsize=(5 * len(num_cols_noid), 3))
        if len(num_cols_noid) == 1:
            axes = [axes]
        for ax, col in zip(axes, num_cols_noid):
            ax.hist(df_clean[col].dropna(), bins=20, color="#4f46e5", alpha=0.7, edgecolor="white")
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Gender pie
    with col_b:
        if "gender" in df_clean.columns:
            st.markdown("**Gender split**")
            counts = df_clean["gender"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
            ax2.pie(counts, labels=counts.index, autopct="%1.1f%%",
                    colors=["#6366f1", "#a78bfa"], startangle=90)
            st.pyplot(fig2)
            plt.close(fig2)

    # Correlation heatmap
    st.markdown("**Correlation matrix**")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    corr = df_clean[num_cols_noid].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3,
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax3.set_title("Feature Correlation", fontsize=12)
    st.pyplot(fig3)
    plt.close(fig3)

# ── Resolve feature columns ────────────────────────────────────────────────────
feature_cols_raw = FEATURE_PRESETS[preset_name]
# Map to actual cleaned column names (handle k$ → k_)
col_map = {c: c for c in df_clean.columns}
resolved_features = []
for fc in feature_cols_raw:
    if fc in df_clean.columns:
        resolved_features.append(fc)
    else:
        # Fuzzy fallback
        matches = [c for c in df_clean.columns if fc[:6] in c]
        if matches:
            resolved_features.append(matches[0])

if not resolved_features:
    st.error("Could not map features to cleaned columns. Check column names.")
    st.stop()

X = df_clean[resolved_features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Tab 4 : Elbow ─────────────────────────────────────────────────────────────
with tab_elbow:
    st.markdown('<div class="section-header">Elbow Method + Silhouette Score</div>',
                unsafe_allow_html=True)
    st.caption(f"Features used: **{', '.join(resolved_features)}**")

    with st.spinner("Computing elbow curve…"):
        inertias, sil_scores = [], []
        ks = range(2, 11)
        for ki in ks:
            km = KMeans(n_clusters=ki, init=init_method, max_iter=max_iter,
                        n_init=n_init, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, km.labels_))

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 4))

    ax4a.plot(list(ks), inertias, "o-", color="#4f46e5", linewidth=2)
    ax4a.axvline(k, color="#ef4444", linestyle="--", label=f"Selected K={k}")
    ax4a.set_xlabel("K"); ax4a.set_ylabel("Inertia (WCSS)")
    ax4a.set_title("Elbow Curve"); ax4a.legend()

    ax4b.plot(list(ks), sil_scores, "s-", color="#10b981", linewidth=2)
    ax4b.axvline(k, color="#ef4444", linestyle="--", label=f"Selected K={k}")
    ax4b.set_xlabel("K"); ax4b.set_ylabel("Silhouette Score")
    ax4b.set_title("Silhouette Score"); ax4b.legend()

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    best_k = int(np.argmax(sil_scores)) + 2
    st.info(f"💡 Best silhouette score at **K = {best_k}** "
            f"({sil_scores[best_k-2]:.3f}).  You selected **K = {k}**.")


# ── Tab 5 : Clusters ──────────────────────────────────────────────────────────
with tab_cluster:
    if not run_btn:
        st.info("Configure K in the sidebar and click **🚀 Run Clustering**.")
    else:
        with st.spinner(f"Fitting KMeans with K={k}…"):
            km_final = KMeans(n_clusters=k, init=init_method, max_iter=max_iter,
                              n_init=n_init, random_state=42)
            labels = km_final.fit_predict(X_scaled)
            df_result = df_clean.loc[X.index].copy()
            df_result["cluster"] = labels
            sil = silhouette_score(X_scaled, labels)

        # Metrics row
        st.markdown('<div class="section-header">Model Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("K (clusters)", k)
        m2.metric("Silhouette Score", f"{sil:.3f}")
        m3.metric("Inertia", f"{km_final.inertia_:,.0f}")
        m4.metric("Customers clustered", len(df_result))

        # ── Scatter plot ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Cluster Scatter Plot</div>',
                    unsafe_allow_html=True)

        palette = sns.color_palette("tab10", k)
        x_feat, y_feat = resolved_features[0], resolved_features[-1]

        fig5, ax5 = plt.subplots(figsize=(9, 5))
        for ci in range(k):
            mask = df_result["cluster"] == ci
            ax5.scatter(df_result.loc[mask, x_feat],
                        df_result.loc[mask, y_feat],
                        c=[palette[ci]], label=f"Cluster {ci}",
                        alpha=0.75, edgecolors="white", linewidths=0.4, s=70)

        # Centroids (inverse-transform)
        centroids_orig = scaler.inverse_transform(km_final.cluster_centers_)
        feat_idx_x = resolved_features.index(x_feat)
        feat_idx_y = resolved_features.index(y_feat)
        ax5.scatter(centroids_orig[:, feat_idx_x], centroids_orig[:, feat_idx_y],
                    marker="X", s=200, c="black", zorder=5, label="Centroids")
        ax5.set_xlabel(x_feat); ax5.set_ylabel(y_feat)
        ax5.set_title(f"K-Means Clusters (K={k})")
        ax5.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        # ── PCA 2D view (if > 2 features) ────────────────────────────────────
        if len(resolved_features) > 2:
            st.markdown('<div class="section-header">PCA 2D Projection</div>',
                        unsafe_allow_html=True)
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            fig6, ax6 = plt.subplots(figsize=(8, 4.5))
            for ci in range(k):
                mask = labels == ci
                ax6.scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=[palette[ci]], label=f"Cluster {ci}",
                            alpha=0.75, edgecolors="white", linewidths=0.4, s=70)
            ax6.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax6.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax6.set_title("PCA Projection of Clusters")
            ax6.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
            fig6.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)

        # ── Cluster profiles ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Cluster Profiles</div>',
                    unsafe_allow_html=True)
        num_feat_clean = [c for c in df_result.select_dtypes(include="number").columns
                          if "id" not in c and "enc" not in c and c != "cluster"]
        profile = df_result.groupby("cluster")[num_feat_clean].mean().round(2)
        profile["count"] = df_result["cluster"].value_counts().sort_index()
        st.dataframe(profile, use_container_width=True)

        # Box plots per cluster
        st.markdown('<div class="section-header">Feature Distribution by Cluster</div>',
                    unsafe_allow_html=True)
        fig7, axes7 = plt.subplots(1, len(num_feat_clean),
                                    figsize=(5 * len(num_feat_clean), 4))
        if len(num_feat_clean) == 1:
            axes7 = [axes7]
        for ax7, feat in zip(axes7, num_feat_clean):
            data_by_cluster = [df_result[df_result["cluster"] == ci][feat].values
                               for ci in range(k)]
            bp = ax7.boxplot(data_by_cluster, patch_artist=True, notch=False)
            for patch, color in zip(bp["boxes"], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax7.set_xticklabels([f"C{ci}" for ci in range(k)])
            ax7.set_title(feat, fontsize=9)
        fig7.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

        # Store result for export tab
        st.session_state["df_result"] = df_result

# ── Tab 6 : Export ────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)

    df_exp = st.session_state.get("df_result", df_clean)

    csv_bytes = df_exp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download clustered CSV",
        data=csv_bytes,
        file_name="mall_customers_clustered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("**Preview**")
    st.dataframe(df_exp, use_container_width=True, height=300)
