# app.py – Part 1

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

LANG = {
    "en": {
        "title": "🚗 Motor Insurance GLM Analyzer",
        "desc": "Analyze your motor vehicle insurance claims using statistical GLMs. Choose Poisson for frequency or Gamma for severity modeling. View metrics, top predictions, and visualizations in beautiful dark mode.",
        "upload": "📤 Upload CSV File",
        "model_select": "🔘 Select GLM Model",
        "language": "🌐 Language",
        "submit": "🔥 Analyze Dataset",
        "summary": "📈 Model Summary & Predictions",
        "gallery": "📸 Visual Gallery (Dark Mode)",
        "about": "👨‍💻 Created by Tamaghna Nag • Powered by Gradio + Plotly",
        "csv_hint": "📌 CSV must include:\n- Column: 'ClaimNb' or 'Claim_Count'\n- Optionally: 'Severity' or 'Amount'\n- All others must be numeric or categorical"
    }
}

def MAE(y, p): return mean_absolute_error(y, p)
def RMSE(y, p): return np.sqrt(mean_squared_error(y, p))
def R2(y, p): return r2_score(y, p)

def preprocess_data(df):
    df = df.dropna()
    cat_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    num_cols = df.select_dtypes(include=np.number).columns
    target_keys = ['claim', 'amount', 'severity']
    target_cols = [col for col in num_cols if any(k in col.lower() for k in target_keys)]
    scale_cols = [col for col in num_cols if col not in target_cols]
    df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])
    return df

def split_targets(df):
    count_col = next((col for col in df.columns if "claimnb" in col.lower() or ("claim" in col.lower() and "count" in col.lower())), None)
    sev_col = next((col for col in df.columns if "severity" in col.lower() or "amount" in col.lower()), None)
    if not count_col:
        raise ValueError("❌ No valid claim count column found.")
    X = df.drop(columns=[count_col] + ([sev_col] if sev_col else []))
    return X, df[count_col], df[sev_col] if sev_col else None

def train_poisson(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    model = PoissonRegressor(alpha=0.01, max_iter=10000).fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return model, preds, y_te, MAE(y_te, preds), RMSE(y_te, preds), R2(y_te, preds), X.columns, model.coef_

def train_gamma(X, y):
    X, y = X[y > 0], y[y > 0]
    y_log = np.log(y + 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2)
    model = GammaRegressor(alpha=0.01, max_iter=10000).fit(X_tr, y_tr)
    preds = np.exp(model.predict(X_te)) - 1
    y_te = np.exp(y_te) - 1
    return model, preds, y_te, MAE(y_te, preds), RMSE(y_te, preds), R2(y_te, preds), X.columns, model.coef_
# 📊 Plotly Visuals

def plot_residual(y_true, y_pred, model_type):
    residuals = y_true - y_pred
    fig = px.histogram(residuals, nbins=30, title=f"{model_type} Residual Distribution", template="plotly_dark")
    return fig

def plot_feature_importance(cols, coefs, model_type):
    df = pd.DataFrame({"Feature": cols, "Importance": coefs}).sort_values("Importance", ascending=False).head(10)
    fig = px.bar(df, x="Importance", y="Feature", orientation='h', title=f"{model_type} Top Features", template="plotly_dark")
    return fig

def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        colorbar=dict(title="Correlation"),
        zmin=-1, zmax=1
    ))
    fig.update_layout(title="📉 Feature Correlation Heatmap", template="plotly_dark")
    return fig

# 🧠 ML Pipeline
def pipeline(file, lang="en", model_toggle="Poisson"):
    labels = LANG[lang]
    try:
        df = pd.read_csv(file.name)
        df_clean = preprocess_data(df)
        X, y_freq, y_sev = split_targets(df_clean)

        p_out = train_poisson(X, y_freq)
        g_out = train_gamma(X, y_sev) if y_sev is not None else None

        selected = p_out if model_toggle == "Poisson" else g_out
        if selected is None:
            raise ValueError("⚠️ Gamma model requires 'Severity' or 'Amount' column with values > 0.")

        summary = f"✅ **{model_toggle} GLM Selected**\n\n📌 MAE: `{selected[3]:.3f}` | 📌 RMSE: `{selected[4]:.3f}` | 📌 R²: `{selected[5]:.3f}`"
        table = pd.DataFrame({
            "Actual": selected[2].round(2),
            "Predicted": selected[1].round(2)
        }).head(10).to_markdown(index=False)

        vis1 = plot_correlation_matrix(df_clean)
        vis2 = plot_residual(selected[2], selected[1], model_toggle)
        vis3 = plot_feature_importance(selected[6], selected[7], model_type=model_toggle)

        return summary + "\n\n### 🧾 Top 10 Predictions\n" + table, vis1, vis2, vis3

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None

# 🎛 Gradio UI
def launch_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        lang = gr.Radio(["en"], value="en", label=LANG["en"]["language"])
        model = gr.Radio(["Poisson", "Gamma"], value="Poisson", label=LANG["en"]["model_select"])

        title = gr.Markdown()
        desc = gr.Markdown()
        csv_hint = gr.Markdown()

        def update_texts(l):
            return (
                f"## {LANG[l]['title']}",
                LANG[l]["desc"],
                f"🔎 **{LANG[l]['csv_hint']}**"
            )

        lang.change(fn=update_texts, inputs=lang, outputs=[title, desc, csv_hint])
        title.value, desc.value, csv_hint.value = update_texts("en")

        with gr.Row():
            file = gr.File(label=LANG["en"]["upload"], file_types=[".csv"])
            submit = gr.Button(value=LANG["en"]["submit"])

        with gr.Row():
            summary = gr.Markdown()

        with gr.Row():
            plot1 = gr.Plot(label="Correlation Heatmap")
            plot2 = gr.Plot(label="Residuals")
            plot3 = gr.Plot(label="Feature Importance")

        submit.click(fn=pipeline, inputs=[file, lang, model], outputs=[summary, plot1, plot2, plot3])

        gr.Markdown("---")
        gr.Markdown(f"{LANG['en']['about']}")

    demo.launch(inbrowser=True)

if __name__ == "__main__":
    launch_app()
