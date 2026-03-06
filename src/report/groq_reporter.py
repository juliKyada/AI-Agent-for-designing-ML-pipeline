"""
Groq-powered Industry-Grade ML Pipeline Report Generator
Produces a comprehensive, professional report using the Groq LLM API.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file: src/report -> src -> project)
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_env_path)


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_prompt(results: Dict[str, Any]) -> str:
    """Construct the structured prompt from pipeline results."""

    task_type = results.get("task_type", "unknown").upper()
    metadata = results.get("metadata", {})
    best = results.get("best_pipeline", {})
    all_pipelines = results.get("all_pipelines", [])
    improvement_plan = results.get("improvement_plan", {})
    preprocessing = results.get("preprocessing", {}) or {}
    training_warnings = results.get("training_warnings", []) or []

    # ── Dataset facts ────────────────────────────────────────────────────────── #
    n_samples = metadata.get("n_samples", "N/A")
    n_features = metadata.get("n_features", "N/A")
    missing_pct = _safe_float(metadata.get("missing_percentage", 0))
    class_balance = metadata.get("class_balance", {})
    feature_types = metadata.get("feature_types", {})

    # ── Best model metrics ────────────────────────────────────────────────────── #
    best_name = best.get("name", "N/A")
    metrics = best.get("metrics", {})
    issues = best.get("issues", [])

    if task_type == "CLASSIFICATION":
        key_metrics = {
            "Train Accuracy": f"{_safe_float(metrics.get('train_accuracy')):.4f}",
            "Test Accuracy":  f"{_safe_float(metrics.get('test_accuracy')):.4f}",
            "Train F1":       f"{_safe_float(metrics.get('train_f1')):.4f}",
            "Test F1":        f"{_safe_float(metrics.get('test_f1')):.4f}",
            "Test Precision": f"{_safe_float(metrics.get('test_precision')):.4f}",
            "Test Recall":    f"{_safe_float(metrics.get('test_recall')):.4f}",
            "CV Mean":        f"{_safe_float(metrics.get('cv_mean')):.4f}",
            "CV Std":         f"{_safe_float(metrics.get('cv_std')):.4f}",
        }
        primary_metric_label = "Accuracy"
        primary_metric_value = f"{_safe_float(metrics.get('test_accuracy')):.4f}"
    else:
        key_metrics = {
            "Train R²":  f"{_safe_float(metrics.get('train_r2')):.4f}",
            "Test R²":   f"{_safe_float(metrics.get('test_r2')):.4f}",
            "Train RMSE":f"{_safe_float(metrics.get('train_rmse')):.4f}",
            "Test RMSE": f"{_safe_float(metrics.get('test_rmse')):.4f}",
            "Train MAE": f"{_safe_float(metrics.get('train_mae')):.4f}",
            "Test MAE":  f"{_safe_float(metrics.get('test_mae')):.4f}",
            "CV Mean":   f"{_safe_float(metrics.get('cv_mean')):.4f}",
            "CV Std":    f"{_safe_float(metrics.get('cv_std')):.4f}",
        }
        primary_metric_label = "R² Score"
        primary_metric_value = f"{_safe_float(metrics.get('test_r2')):.4f}"

    metrics_str = "\n".join(f"  - {k}: {v}" for k, v in key_metrics.items())

    # ── All-pipeline comparison ──────────────────────────────────────────────── #
    pipeline_rows = []
    for p in all_pipelines:
        pm = p.get("metrics", {})
        if task_type == "CLASSIFICATION":
            score = f"{_safe_float(pm.get('test_accuracy')):.4f}"
            extra = f"F1={_safe_float(pm.get('test_f1')):.4f}"
        else:
            score = f"{_safe_float(pm.get('test_r2')):.4f}"
            extra = f"RMSE={_safe_float(pm.get('test_rmse')):.4f}"
        cv = f"{_safe_float(pm.get('cv_mean')):.4f}±{_safe_float(pm.get('cv_std')):.4f}"
        n_issues = len(p.get("issues", []))
        pipeline_rows.append(
            f"  • {p.get('pipeline_name','?')}: {primary_metric_label}={score}, {extra}, CV={cv}, Issues={n_issues}"
        )
    pipeline_table = "\n".join(pipeline_rows) if pipeline_rows else "  (no pipeline data)"

    # ── Preprocessing ────────────────────────────────────────────────────────── #
    imputation = preprocessing.get("imputation_strategy", "N/A")
    removed_features = preprocessing.get("removed_features", [])
    imputed_features = list(preprocessing.get("imputation_values", {}).keys())
    rows_removed = preprocessing.get("rows_removed_by_target_na", 0)

    # ── Recommendations ──────────────────────────────────────────────────────── #
    recs = improvement_plan.get("overall_recommendations", []) or []
    needs_improvement = improvement_plan.get("needs_improvement", False)
    recs_str = "\n".join(f"  - {r}" for r in recs) if recs else "  - None flagged"

    warnings_str = "\n".join(f"  - {w}" for w in training_warnings) if training_warnings else "  - None"
    issues_str = "\n".join(f"  - {i}" for i in issues) if issues else "  - None"

    prompt = f"""You are a Senior ML Engineer and Data Scientist at a top-tier technology firm.
You have just completed a full automated ML pipeline analysis using the MetaFlow system.
Your task is to write a **comprehensive, professional, industry-grade ML Pipeline Report** based on the structured data provided below.

---

## INPUT DATA

**Task Type:** {task_type}

**Dataset Characteristics:**
  - Samples: {n_samples}
  - Features: {n_features}
  - Missing Data: {missing_pct:.2f}%
  - Feature Types: {json.dumps(feature_types, indent=4)}
  - Class Balance (if classification): {json.dumps(class_balance, indent=4)}

**Preprocessing Applied:**
  - Imputation Strategy: {imputation}
  - Features Removed (high missing): {removed_features}
  - Features Imputed: {imputed_features}
  - Rows Removed (NaN Target): {rows_removed}

**Training Warnings:**
{warnings_str}

**Best Model Selected:** {best_name}

**Best Model Performance Metrics:**
{metrics_str}

**Best Model Detected Issues:**
{issues_str}

**All Evaluated Pipelines:**
{pipeline_table}

**Improvement Plan (System-Generated):**
  - Needs Further Improvement: {needs_improvement}
  - Recommendations:
{recs_str}

---

## REPORT REQUIREMENTS

Generate a complete report with **all** of the following sections. Use Markdown formatting with headers, bullet points, tables, and bold/italic text appropriately. Be precise, technical, and professional. Do NOT invent fictional numbers — use only the data provided.

### 1. Executive Summary
A concise (3–5 sentences) high-level summary: problem type, dataset profile, best-performing model, primary metric result, and overall pipeline health. Include a deployment recommendation (production-ready / needs improvement / major rework needed).

### 2. Dataset Analysis
- Dataset size, dimensionality, memory profile
- Feature composition (numerical vs categorical)
- Data quality assessment: missing values, outliers, class imbalance
- Any preprocessing actions taken and their rationale

### 3. ML Task & Methodology
- Detected ML task and justification
- Pipeline generation strategy: how candidate models were selected
- Evaluation methodology: train/test split strategy, cross-validation approach
- Optimization iterations: hyperparameter tuning rationale

### 4. Model Performance Analysis
- A Markdown table comparing ALL evaluated pipelines with their key metrics
- Identify top 3 models and explain why they performed better
- Discuss train vs. test gap (overfitting/underfitting analysis)
- Cross-validation analysis: variance and reliability

### 5. Best Model Deep Dive
- Model architecture and hyperparameter insights
- Why this model outperformed others
- Bias-variance tradeoff assessment for this model
- Confidence in the model's results

### 6. Detected Issues & Risk Assessment
- List all detected issues from both system and human perspective
- Risk classification (Low / Medium / High) for each issue
- Impact on production deployment for each risk
- Data quality risks

### 7. Feature Engineering & Data Preprocessing
- Detailed review of all preprocessing steps applied
- Recommendations for additional feature engineering
- Dimensionality considerations

### 8. Production Deployment Recommendations
- Model serving strategy (batch vs. real-time inference)
- Monitoring plan: which metrics to track in production
- Data drift detection approach
- Model retraining trigger recommendations
- Infrastructure requirements estimate
- Rollout strategy (canary, blue-green, shadow)

### 9. Improvement Roadmap
- Short term (1–2 weeks): quick wins
- Medium term (1 month): experimentation
- Long term (3 months): architectural improvements
- Specific next experiments to try (with justification)

### 10. Conclusion
Final professional assessment with a clear production readiness verdict and the single most important action item.

---

Write the full report now. Be thorough, detailed, and industry-grade. Use proper Markdown."""

    return prompt


class GroqReportGenerator:
    """
    Generates an industry-grade ML pipeline report using the Groq LLM API.

    Usage:
        reporter = GroqReportGenerator()
        report_md = reporter.generate(results)   # results = dict from MetaFlowAgent.run()
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    MAX_TOKENS = 8192

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file or pass api_key= to GroqReportGenerator()."
            )

    def generate(self, results: Dict[str, Any]) -> str:
        """
        Generate the full industry-grade report from pipeline results.

        Args:
            results: The dict returned by MetaFlowAgent.run()

        Returns:
            A Markdown-formatted report string.
        """
        try:
            from groq import Groq  # lazy import so groq is optional until used
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required for AI report generation. "
                "Install it with: pip install groq"
            ) from exc

        client = Groq(api_key=self.api_key)
        prompt = _build_prompt(results)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior ML Engineer specializing in AutoML systems, "
                        "model evaluation, and production deployment. Always produce "
                        "structured, professional, Markdown-formatted reports."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.MAX_TOKENS,
            temperature=0.3,
        )

        report_text = response.choices[0].message.content.strip()
        return report_text

    def generate_with_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the report and return it along with usage metadata.

        Returns:
            {
                "report": str,          # Markdown report
                "model": str,           # Model used
                "tokens_used": int,     # Total tokens consumed
                "finish_reason": str,   # API finish reason
            }
        """
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "The 'groq' package is required. Install it with: pip install groq"
            ) from exc

        client = Groq(api_key=self.api_key)
        prompt = _build_prompt(results)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior ML Engineer specializing in AutoML systems, "
                        "model evaluation, and production deployment. Always produce "
                        "structured, professional, Markdown-formatted reports."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.MAX_TOKENS,
            temperature=0.3,
        )

        choice = response.choices[0]
        usage = response.usage

        return {
            "report": choice.message.content.strip(),
            "model": self.model,
            "tokens_used": usage.total_tokens if usage else 0,
            "finish_reason": choice.finish_reason,
        }
