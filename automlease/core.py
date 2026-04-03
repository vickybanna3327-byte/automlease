import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
from rich.table import Table
from rich import print as rprint

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

    def fit(self, data, target):
        console.print("\n[bold blue]AutoMLease — Starting Automatic ML Pipeline...[/bold blue]\n")

        # Step 1 - Load data
        if isinstance(data, str):
            df = pd.read_csv(data)
            console.print(f"[green]✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns[/green]")
        else:
            df = data.copy()
            console.print(f"[green]✅ Dataset received: {df.shape[0]} rows, {df.shape[1]} columns[/green]")

        # Step 2 - Clean data
        console.print("[yellow]🔄 Cleaning data...[/yellow]")
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col].astype(str))
        df = df.dropna()
        console.print(f"[green]✅ Data cleaned: {df.shape[0]} rows remaining[/green]")

        # Step 3 - Split features and target
        X = df.drop(target, axis=1)
        y = df[target]
        self.feature_names = list(X.columns)

        # Step 4 - Detect task type
        unique_values = y.nunique()
        if unique_values <= 10:
            self.task_type = 'classification'
            console.print("[green]✅ Task detected: Classification[/green]")
        else:
            self.task_type = 'regression'
            console.print("[green]✅ Task detected: Regression[/green]")

        # Step 5 - Train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        console.print(f"[green]✅ Data split: {len(self.X_train)} training, {len(self.X_test)} testing[/green]")

        # Step 6 - Train models and pick best
        console.print("[yellow]🔄 Training models...[/yellow]")
        if self.task_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
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
                'Linear Regression': LinearRegression()
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

        # Step 7 - Save best model
        joblib.dump(self.model, 'best_model.pkl')
        console.print(f"\n[bold green]🏆 Best Model: {self.best_model_name}[/bold green]")
        console.print("[green]✅ Model saved as best_model.pkl[/green]")
        return self

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

        # Detailed report
        y_pred = self.model.predict(self.X_test)
        if self.task_type == 'classification':
            console.print("\n[bold]Detailed Classification Report:[/bold]")
            console.print(classification_report(self.y_test, y_pred))
        else:
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            console.print(f"\n[bold]Regression Results:[/bold]")
            console.print(f"Mean Squared Error: {mse:.4f}")
            console.print(f"R² Score: {r2:.4f}")

        # Feature importance chart
        if hasattr(self.model, 'feature_importances_'):
            console.print("\n[yellow]📊 Generating feature importance chart...[/yellow]")
            importances = self.model.feature_importances_
            feat_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
            plt.title(f'Top Features — {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()
            console.print("[green]✅ Chart saved as feature_importance.png[/green]")

        console.print("\n[bold green]====== REPORT COMPLETE ======[/bold green]\n")

    def predict(self, data):
        return self.model.predict(data)