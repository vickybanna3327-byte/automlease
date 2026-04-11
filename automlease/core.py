import os
import subprocess
import sys
import textwrap

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

console = Console()


class AutoML:
    def __init__(self):
        self.model = None
        self.best_model_name = None
        self.task_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        self.df = None
        self.target = None

    # ------------------------------------------------------------------
    # Improvement 3 helper — data quality report
    # ------------------------------------------------------------------
    def _print_data_quality_report(self, df_raw):
        console.print("\n[bold yellow]📋 Data Quality Report[/bold yellow]")
        table = Table(title="Missing Values per Column", show_lines=True)
        table.add_column("Column", style="cyan")
        table.add_column("Missing Count", style="magenta")
        table.add_column("Missing %", style="magenta")
        table.add_column("Action", style="green")

        has_missing = False
        for col in df_raw.columns:
            n_missing = df_raw[col].isnull().sum()
            pct = n_missing / len(df_raw) * 100
            if n_missing > 0:
                has_missing = True
                table.add_row(col, str(n_missing), f"{pct:.1f}%", "Row dropped")
            else:
                table.add_row(col, "0", "0.0%", "OK")

        console.print(table)
        if not has_missing:
            console.print("[green]✅ No missing values found — dataset is clean![/green]")
        else:
            total_missing_rows = df_raw.isnull().any(axis=1).sum()
            console.print(
                f"[yellow]⚠️  {total_missing_rows} rows with missing values will be dropped.[/yellow]"
            )

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------
    def fit(self, data, target):
        console.print("\n[bold blue]AutoMLease — Starting Automatic ML Pipeline...[/bold blue]\n")

        # Step 1 — Load data
        if isinstance(data, str):
            df = pd.read_csv(data)
            console.print(f"[green]✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns[/green]")
        else:
            df = data.copy()
            console.print(f"[green]✅ Dataset received: {df.shape[0]} rows, {df.shape[1]} columns[/green]")

        # Improvement 3 — data quality report (before cleaning)
        self._print_data_quality_report(df)

        # Step 2 — Clean data
        console.print("\n[yellow]🔄 Cleaning data...[/yellow]")
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col].astype(str))
        df = df.dropna()
        console.print(f"[green]✅ Data cleaned: {df.shape[0]} rows remaining[/green]")
        self.df = df
        self.target = target

        # Step 3 — Split features and target
        X = df.drop(target, axis=1)
        y = df[target]
        self.feature_names = list(X.columns)

        # Step 4 — Detect task type
        unique_values = y.nunique()
        if unique_values <= 10:
            self.task_type = 'classification'
            console.print("[green]✅ Task detected: Classification[/green]")
        else:
            self.task_type = 'regression'
            console.print("[green]✅ Task detected: Regression[/green]")

        # Step 5 — Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        console.print(f"[green]✅ Data split: {len(self.X_train)} training, {len(self.X_test)} testing[/green]")

        # Step 6 — Train models and pick best
        # Improvement 1 — XGBoost added to the competition
        console.print("[yellow]🔄 Training models (Random Forest, XGBoost, Linear/Logistic)...[/yellow]")
        if self.task_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'XGBoost': XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0),
            }
            best_score = 0
            for name, m in models.items():
                m.fit(self.X_train, self.y_train)
                score = accuracy_score(self.y_test, m.predict(self.X_test))
                self.results[name] = score
                console.print(f"  [cyan]{name}: {score*100:.2f}% accuracy[/cyan]")
                if score > best_score:
                    best_score = score
                    self.model = m
                    self.best_model_name = name
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            }
            best_score = -999
            for name, m in models.items():
                m.fit(self.X_train, self.y_train)
                score = r2_score(self.y_test, m.predict(self.X_test))
                self.results[name] = score
                console.print(f"  [cyan]{name}: R² = {score:.4f}[/cyan]")
                if score > best_score:
                    best_score = score
                    self.model = m
                    self.best_model_name = name

        # Step 7 — Save best model + dashboard data
        joblib.dump(self.model, 'best_model.pkl')
        self._save_dashboard_data()
        console.print(f"\n[bold green]🏆 Best Model: {self.best_model_name}[/bold green]")
        console.print("[green]✅ Model saved as best_model.pkl[/green]")
        return self

    # ------------------------------------------------------------------
    # report()  — now includes Improvement 5: SHAP explanations
    # ------------------------------------------------------------------
    def report(self):
        console.print("\n[bold blue]====== AUTOMLEASE REPORT ======[/bold blue]\n")

        # Results table
        table = Table(title="Model Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Score", style="green")
        for name, score in self.results.items():
            marker = "🏆" if name == self.best_model_name else ""
            if self.task_type == 'classification':
                table.add_row(f"{marker} {name}", f"{score*100:.2f}%")
            else:
                table.add_row(f"{marker} {name}", f"R² = {score:.4f}")
        console.print(table)

        y_pred = self.model.predict(self.X_test)

        if self.task_type == 'classification':
            console.print("\n[bold]Detailed Classification Report:[/bold]")
            console.print(classification_report(self.y_test, y_pred))

            console.print("[yellow]📊 Generating confusion matrix...[/yellow]")
            cm = confusion_matrix(self.y_test, y_pred)
            labels = sorted(self.y_test.unique())
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            annot = np.array([
                [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
                for i in range(cm.shape[0])
            ])
            fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
            sns.heatmap(
                cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax,
            )
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("Actual", fontsize=12)
            ax.set_title(f"Confusion Matrix — {self.best_model_name}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=150)
            plt.show()
            console.print("[green]✅ Saved: confusion_matrix.png[/green]")
        else:
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            console.print(f"\n[bold]Regression Results:[/bold]")
            console.print(f"Mean Squared Error: {mse:.4f}")
            console.print(f"R² Score: {r2:.4f}")

            console.print("[yellow]📊 Generating actual vs predicted plot...[/yellow]")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3, color="steelblue")
            lim = [min(self.y_test.min(), y_pred.min()), max(self.y_test.max(), y_pred.max())]
            ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
            ax.set_xlabel("Actual", fontsize=12)
            ax.set_ylabel("Predicted", fontsize=12)
            ax.set_title(
                f"Actual vs Predicted — {self.best_model_name}\nR² = {r2:.4f}",
                fontsize=14, fontweight="bold",
            )
            ax.legend()
            plt.tight_layout()
            plt.savefig("actual_vs_predicted.png", dpi=150)
            plt.show()
            console.print("[green]✅ Saved: actual_vs_predicted.png[/green]")

        # Feature importance chart
        if hasattr(self.model, 'feature_importances_'):
            console.print("\n[yellow]📊 Generating feature importance chart...[/yellow]")
            importances = self.model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances,
            }).sort_values('Importance', ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
            plt.title(f'Top Features — {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()
            console.print("[green]✅ Chart saved as feature_importance.png[/green]")

        # Improvement 5 — SHAP explanations
        self._shap_explanations()

        console.print("\n[bold green]====== REPORT COMPLETE ======[/bold green]\n")

    # ------------------------------------------------------------------
    # Improvement 5 — SHAP explanations
    # ------------------------------------------------------------------
    def _shap_explanations(self):
        try:
            import shap
        except ImportError:
            console.print("[red]SHAP not installed. Run: pip install shap[/red]")
            return

        console.print("\n[yellow]🔍 Computing SHAP explanations...[/yellow]")
        try:
            # TreeExplainer works for RF and XGBoost; fall back to generic Explainer
            if hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(self.X_test)
            else:
                explainer = shap.Explainer(self.model, self.X_train)
                shap_values = explainer(self.X_test).values

            # For multi-class classification, shap_values is a list — use class 1
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            # Mean absolute SHAP per feature
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean |SHAP|': mean_abs_shap,
            }).sort_values('Mean |SHAP|', ascending=False)

            # SHAP summary bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=shap_df, x='Mean |SHAP|', y='Feature', palette='coolwarm')
            plt.title(f'SHAP Feature Importance — {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('shap_importance.png', dpi=150)
            plt.show()
            console.print("[green]✅ Saved: shap_importance.png[/green]")

            # Plain-English explanation
            console.print("\n[bold cyan]🧠 SHAP Plain-English Explanation:[/bold cyan]")
            console.print(
                f"The model '{self.best_model_name}' makes predictions mainly based on these features:\n"
            )
            for i, row in shap_df.head(5).iterrows():
                feat = row['Feature']
                importance = row['Mean |SHAP|']
                # Find average direction of effect
                feat_idx = self.feature_names.index(feat)
                avg_direction = np.mean(shap_vals[:, feat_idx])
                direction = "increases" if avg_direction > 0 else "decreases"
                console.print(
                    f"  • [bold]{feat}[/bold] — on average {direction} the prediction "
                    f"(impact score: {importance:.4f})"
                )

            console.print(
                "\n[dim]Higher impact score = stronger influence on model output.[/dim]"
            )

        except Exception as e:
            console.print(f"[red]⚠️  SHAP computation failed: {e}[/red]")

    # ------------------------------------------------------------------
    # eda()
    # ------------------------------------------------------------------
    def eda(self):
        if self.df is None:
            console.print("[red]❌ No data found. Run .fit() first.[/red]")
            return

        console.print("\n[bold blue]====== EDA VISUALIZATIONS ======[/bold blue]\n")

        console.print("[yellow]📊 Generating correlation heatmap...[/yellow]")
        corr = self.df.corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.6)))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=150)
        plt.show()
        console.print("[green]✅ Saved: correlation_heatmap.png[/green]")

        console.print("[yellow]📊 Generating target distribution plot...[/yellow]")
        fig, ax = plt.subplots(figsize=(8, 5))
        target_series = self.df[self.target]
        if self.task_type == "classification":
            counts = target_series.value_counts().sort_index()
            sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis", ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            for i, v in enumerate(counts.values):
                ax.text(i, v + counts.values.max() * 0.01, str(v), ha="center", fontsize=10)
        else:
            sns.histplot(target_series, kde=True, color="steelblue", ax=ax)
            ax.set_xlabel(self.target)
            ax.set_ylabel("Frequency")
        ax.set_title(f"Target Distribution — {self.target}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("target_distribution.png", dpi=150)
        plt.show()
        console.print("[green]✅ Saved: target_distribution.png[/green]")

        console.print("\n[bold green]====== EDA COMPLETE ======[/bold green]\n")

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------
    def predict(self, data):
        return self.model.predict(data)

    # ------------------------------------------------------------------
    # Improvement 2 — predict_new()
    # ------------------------------------------------------------------
    def predict_new(self, input_dict: dict):
        """
        Predict on a single observation provided as a dictionary.

        Example:
            model.predict_new({'RM': 6.5, 'LSTAT': 10.0})
        """
        if self.model is None:
            console.print("[red]❌ No model trained. Run .fit() first.[/red]")
            return None

        # Build a row with the exact feature order used during training,
        # filling any missing keys with 0.
        row = {feat: input_dict.get(feat, 0) for feat in self.feature_names}
        X_new = pd.DataFrame([row])

        prediction = self.model.predict(X_new)[0]

        if self.task_type == 'classification':
            console.print(
                f"\n[bold green]🔮 Predicted Class: {prediction}[/bold green]"
            )
        else:
            console.print(
                f"\n[bold green]🔮 Predicted Value: {prediction:.4f}[/bold green]"
            )

        return prediction

    # ------------------------------------------------------------------
    # Improvement 4 — dashboard()
    # ------------------------------------------------------------------
    def _save_dashboard_data(self):
        """Persist everything the Streamlit dashboard needs."""
        data = {
            'feature_names': self.feature_names,
            'results': self.results,
            'task_type': self.task_type,
            'best_model_name': self.best_model_name,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'X_train': self.X_train,
        }
        joblib.dump(data, 'dashboard_data.pkl')

    def dashboard(self):
        """
        Launch a Streamlit web app showing:
          • EDA charts
          • Model comparison scores
          • Actual vs Predicted / Confusion matrix
          • Live prediction form
        """
        # Write the Streamlit app script
        script_path = os.path.join(os.getcwd(), 'automlease_dashboard.py')
        script = textwrap.dedent("""\
            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import streamlit as st
            from sklearn.metrics import (accuracy_score, confusion_matrix,
                                         mean_squared_error, r2_score)

            st.set_page_config(page_title="AutoMLease Dashboard", layout="wide")
            st.title("🤖 AutoMLease Dashboard")

            @st.cache_resource
            def load_data():
                model = joblib.load("best_model.pkl")
                data  = joblib.load("dashboard_data.pkl")
                return model, data

            model, data = load_data()
            feature_names  = data["feature_names"]
            results        = data["results"]
            task_type      = data["task_type"]
            best_model_name = data["best_model_name"]
            X_test         = data["X_test"]
            y_test         = data["y_test"]

            # ── Sidebar ────────────────────────────────────────────────
            st.sidebar.header("Navigation")
            page = st.sidebar.radio(
                "Go to",
                ["📊 Model Comparison", "📈 Charts", "🔮 Live Prediction"],
            )

            # ── Model Comparison ───────────────────────────────────────
            if page == "📊 Model Comparison":
                st.header("Model Comparison")
                rows = []
                for name, score in results.items():
                    label = "🏆 " + name if name == best_model_name else name
                    if task_type == "classification":
                        rows.append({"Model": label, "Score": f"{score*100:.2f}% accuracy"})
                    else:
                        rows.append({"Model": label, "Score": f"R² = {score:.4f}"})
                st.table(pd.DataFrame(rows))
                st.success(f"Best model: **{best_model_name}**")

            # ── Charts ─────────────────────────────────────────────────
            elif page == "📈 Charts":
                st.header("Charts")
                col1, col2 = st.columns(2)

                import os
                for fname, title in [
                    ("correlation_heatmap.png", "Correlation Heatmap"),
                    ("target_distribution.png", "Target Distribution"),
                    ("feature_importance.png", "Feature Importance"),
                    ("actual_vs_predicted.png", "Actual vs Predicted"),
                    ("confusion_matrix.png", "Confusion Matrix"),
                    ("shap_importance.png", "SHAP Feature Importance"),
                ]:
                    if os.path.exists(fname):
                        st.subheader(title)
                        st.image(fname)

            # ── Live Prediction ────────────────────────────────────────
            elif page == "🔮 Live Prediction":
                st.header("Live Prediction")
                st.write("Enter feature values below and click **Predict**.")

                user_input = {}
                cols = st.columns(min(3, len(feature_names)))
                for i, feat in enumerate(feature_names):
                    with cols[i % len(cols)]:
                        user_input[feat] = st.number_input(feat, value=0.0, format="%.4f")

                if st.button("🔮 Predict"):
                    X_new = pd.DataFrame([user_input])
                    prediction = model.predict(X_new)[0]
                    if task_type == "classification":
                        st.success(f"**Predicted Class:** {prediction}")
                    else:
                        st.success(f"**Predicted Value:** {prediction:.4f}")
        """)

        with open(script_path, 'w') as f:
            f.write(script)

        console.print(
            f"\n[bold green]🚀 Launching AutoMLease Dashboard...[/bold green]"
        )
        console.print(
            "[cyan]Open your browser at http://localhost:8501[/cyan]\n"
        )
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
