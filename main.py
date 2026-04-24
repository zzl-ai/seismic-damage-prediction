import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import catboost as cb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tab_transformer_pytorch import TabTransformer
import warnings
import os
from typing import Dict, List, Tuple

# ===================== Configuration =====================
warnings.filterwarnings('ignore')
CONFIG = {
    "MODEL_SAVE_PATH": "tab_transformer_model.pth",
    "LOAD_SAVED_MODEL": True,
    "N_CLASSES_5": 5,
    "N_CLASSES_3": 3,
    "UNCERTAINTY_THRESHOLD": 0.0834,
    "LABEL_MAP_5TO3": {0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "SEED": 42,
    "XGB_PARAMS": {
        "n_estimators": 1000,
        "max_depth": 7,
        "learning_rate": 0.3,
        "min_child_weight": 7,
        "gamma": 0,
        "random_state": 42,
        "eval_metric": "mlogloss",
        "n_jobs": -1
    },
    "CAT_PARAMS": {
        "iterations": 1000,
        "depth": 7,
        "learning_rate": 0.3,
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "random_state": 42,
        "verbose": 100,
        "thread_count": -1,

    },
    "TAB_TRANSFORMER_PARAMS": {
        "categories": [10] * 13,
        "num_continuous": 5,
        "dim": 64,
        "depth": 3,
        "heads": 4,
        "dim_out": 5,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1
    },
    "SUBJECTIVE_LOGIC": {
        "epsilon": 1e-8,
        "model_weights": None,
        "dirichlet_prior": 5.0
    },
    "OUTPUT_PATH": {
        "sample_triple_csv": "test_sample_bu_results_weighted_ds_fusion.csv"
    }
}

np.random.seed(CONFIG["SEED"])
torch.manual_seed(CONFIG["SEED"])

# ===================== Global Validation Metrics =====================
val_metrics = {
    "xgb_val_acc": 0.0,
    "cat_val_acc": 0.0,
    "tab_val_acc": 0.0,
    "raw_weights": None
}

# ===================== Core Utilities =====================
def get_model_activations(model, X: np.ndarray, model_type: str, device: torch.device = None) -> np.ndarray:
    """Extract raw model outputs as evidence (unnormalized, non-probability)"""
    if model_type == "xgb":
        activations = model.predict(X, output_margin=True)
    elif model_type == "cat":
        activations = model.predict(X, prediction_type="RawFormulaVal")
    elif model_type == "tab":
        dataset = TabDataset(X, np.zeros(len(X)), device)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        activations = []
        model.eval()
        with torch.no_grad():
            for x_categ, x_cont, _ in dataloader:
                logits = model(x_categ, x_cont)
                activations.extend(logits.cpu().numpy())
        activations = np.array(activations)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Non-negativity correction (ReLU) for evidence
    activations = np.maximum(activations, 0.0)
    return activations


def calculate_cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, any]:
    """Calculate standard classification metrics (TP, TN, FP, FN based)"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "confusion_matrix": np.zeros((n_classes, n_classes)),
            "class_accuracy": [0.0] * n_classes,
            "class_precision": [0.0] * n_classes,
            "class_recall": [0.0] * n_classes,
            "overall_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0
        }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    class_accuracy = []
    class_precision = []
    class_recall = []

    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN

        cls_acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        cls_pre = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        cls_rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        class_accuracy.append(cls_acc)
        class_precision.append(cls_pre)
        class_recall.append(cls_rec)

    overall_accuracy = np.sum([cm[i, i] for i in range(n_classes)]) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    macro_precision = np.average(class_precision)
    macro_recall = np.average(class_recall)

    return {
        "confusion_matrix": cm,
        "class_accuracy": class_accuracy,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall
    }


def map_labels(y: np.ndarray, map_dict: Dict[int, int]) -> np.ndarray:
    """Map labels from 5-class to 3-class"""
    if len(y) == 0:
        return np.array([])
    mapper = np.vectorize(lambda x: map_dict[x])
    return mapper(y)


def calculate_weights_by_val_accuracy(
        xgb_model, cat_model, tab_model,
        X_val: np.ndarray, y_val: np.ndarray,
        device: torch.device
) -> List[float]:
    """Compute model weights from validation accuracy"""
    global val_metrics

    # XGBoost validation
    xgb_val_preds = np.argmax(xgb_model.predict_proba(X_val), axis=1)
    val_metrics["xgb_val_acc"] = accuracy_score(y_val, xgb_val_preds)

    # CatBoost validation
    cat_val_preds = np.argmax(cat_model.predict_proba(X_val), axis=1)
    val_metrics["cat_val_acc"] = accuracy_score(y_val, cat_val_preds)

    # TabTransformer validation
    val_dataset = TabDataset(X_val, y_val, device)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    tab_val_probs = []
    tab_model.eval()
    with torch.no_grad():
        for x_categ, x_cont, _ in val_dataloader:
            logits = tab_model(x_categ, x_cont)
            tab_val_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
    tab_val_probs = np.array(tab_val_probs)
    tab_val_preds = np.argmax(tab_val_probs, axis=1)
    val_metrics["tab_val_acc"] = accuracy_score(y_val, tab_val_preds)

    raw_accs = np.array([
        val_metrics["xgb_val_acc"],
        val_metrics["cat_val_acc"],
        val_metrics["tab_val_acc"]
    ])
    val_metrics["raw_weights"] = raw_accs

    print("=" * 80)
    print("Validation Accuracy (for Model Weight Correction)")
    print("=" * 80)
    print(f"XGBoost Val Acc: {val_metrics['xgb_val_acc']:.4f}")
    print(f"CatBoost Val Acc: {val_metrics['cat_val_acc']:.4f}")
    print(f"TabTransformer Val Acc: {val_metrics['tab_val_acc']:.4f}")
    print("=" * 80)

    weights = raw_accs / np.sum(raw_accs)
    return weights.tolist()

# ===================== Dataset =====================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device: torch.device):
        self.X_categ = torch.from_numpy(X[:, 5:18].astype(np.int64)).long().to(device)
        self.X_cont = torch.from_numpy(X[:, 0:5].astype(np.float32)).float().to(device)
        self.y = torch.from_numpy(y).long().to(device)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_categ[idx], self.X_cont[idx], self.y[idx]


def init_tab_transformer(device: torch.device) -> TabTransformer:
    return TabTransformer(**CONFIG["TAB_TRANSFORMER_PARAMS"]).to(device)

# ===================== Subjective Logic & DS Fusion =====================
class SubjectiveLogic:
    @staticmethod
    def evidence_to_belief_uncertainty(evidence: np.ndarray, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert evidence to belief-uncertainty (b,u) pairs (d=0)"""
        n_samples = evidence.shape[0]
        eps = CONFIG["SUBJECTIVE_LOGIC"]["epsilon"]
        c = CONFIG["SUBJECTIVE_LOGIC"]["dirichlet_prior"]

        evidence_sum = np.sum(evidence, axis=1, keepdims=True)
        S_total = evidence_sum + c
        S_total = np.maximum(S_total, eps)

        b = evidence / S_total
        u = (n_classes * c) / S_total
        u = np.tile(u, (1, n_classes))

        total = np.sum(b, axis=1, keepdims=True) + u[:, :1]
        b = b / (total + eps)
        u = u / (total + eps)

        return b, u

    @staticmethod
    def weight_correct_bu(b: np.ndarray, u: np.ndarray, weight: float) -> Tuple[np.ndarray, np.ndarray]:
        """Weight correction for belief and uncertainty using validation accuracy"""
        eps = CONFIG["SUBJECTIVE_LOGIC"]["epsilon"]
        weighted_b = b * weight
        weighted_u = u * (1 - weight + eps)

        total = np.sum(weighted_b, axis=1, keepdims=True) + weighted_u[:, :1]
        weighted_b = weighted_b / (total + eps)
        weighted_u = weighted_u / (total + eps)

        return weighted_b, weighted_u

    @staticmethod
    def ds_pair_fusion(
            b1: np.ndarray, u1: np.ndarray,
            b2: np.ndarray, u2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dempster-Shafer fusion for two evidence sources"""
        eps = CONFIG["SUBJECTIVE_LOGIC"]["epsilon"]
        n_samples, n_classes = b1.shape

        sum_b1b2 = np.sum(b1 * b2, axis=1, keepdims=True)
        K_conflict = 1 - sum_b1b2 - (np.mean(u1, axis=1, keepdims=True) * np.mean(u2, axis=1, keepdims=True))
        K_conflict = np.clip(K_conflict, 0, 1 - eps)
        denominator = 1 - K_conflict + eps

        fused_b = (b1 * b2 + b1 * np.mean(u2, axis=1, keepdims=True) + b2 * np.mean(u1, axis=1, keepdims=True)) / denominator
        fused_u = (np.mean(u1, axis=1, keepdims=True) * np.mean(u2, axis=1, keepdims=True)) / denominator
        fused_u = np.tile(fused_u, (1, n_classes))

        total = np.sum(fused_b, axis=1, keepdims=True) + fused_u[:, :1]
        fused_b = fused_b / (total + eps)
        fused_u = fused_u / (total + eps)

        return fused_b, fused_u

    @staticmethod
    def ds_multi_view_fusion(
            views_b: List[np.ndarray], views_u: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Iterative DS fusion for multiple views"""
        if len(views_b) < 2:
            raise ValueError("At least two models are required for fusion")

        fused_b, fused_u = views_b[0], views_u[0]
        for i in range(1, len(views_b)):
            fused_b, fused_u = SubjectiveLogic.ds_pair_fusion(
                fused_b, fused_u,
                views_b[i], views_u[i]
            )
        return fused_b, fused_u

    @staticmethod
    def bu_to_decision(fused_b: np.ndarray) -> np.ndarray:
        """Final decision from fused belief"""
        return np.argmax(fused_b, axis=1)

    @staticmethod
    def summarize_belief_uncertainty(fused_b: np.ndarray, fused_u: np.ndarray) -> Dict:
        """Summary statistics for belief and uncertainty"""
        return {
            "Belief (B)": {
                "mean": np.mean(fused_b),
                "std": np.std(fused_b),
                "max": np.max(fused_b),
                "min": np.min(fused_b)
            },
            "Uncertainty (U)": {
                "mean": np.mean(fused_u),
                "std": np.std(fused_u),
                "max": np.max(fused_u),
                "min": np.min(fused_u)
            },
            "Sample Examples (Top 5)": {
                "Belief": fused_b[:5].round(4),
                "Uncertainty": fused_u[:5].round(4)
            }
        }

# ===================== Weighted Average Fusion =====================
class WeightedAverageFusion:
    @staticmethod
    def weighted_average_fusion_triple(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                                       weights: List[float] = None) -> np.ndarray:
        if weights is None:
            weights = CONFIG["SUBJECTIVE_LOGIC"]["model_weights"]

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        w1, w2, w3 = weights

        n_samples, n_classes = p1.shape
        eps = CONFIG["SUBJECTIVE_LOGIC"]["epsilon"]

        fused_probs = w1 * p1 + w2 * p2 + w3 * p3
        for i in range(n_samples):
            fused_probs[i] = fused_probs[i] / (np.sum(fused_probs[i]) + eps)
        return fused_probs

    @staticmethod
    def probs_to_uncertainty(fused_probs: np.ndarray) -> np.ndarray:
        eps = CONFIG["SUBJECTIVE_LOGIC"]["epsilon"]
        n_classes = fused_probs.shape[1]
        entropy = -np.sum(fused_probs * np.log(fused_probs + eps), axis=1) / np.log(n_classes)
        return entropy

# ===================== Main =====================
if __name__ == '__main__':
    # 1. Data Loading
    print("📥 Loading data...")
    data = pd.read_csv('data.csv')
    y = data["grade"].values - 1
    X = data.drop('grade', axis=1).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=CONFIG["SEED"], stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.67, random_state=CONFIG["SEED"], stratify=y_temp
    )
    print(f"✅ Data split completed: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

    # 2. Model Training & Evidence Extraction
    print("\n🚀 Training base models and extracting evidence...")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**CONFIG["XGB_PARAMS"])
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_evidence = get_model_activations(xgb_model, X_test, model_type="xgb")
    print("✅ XGBoost trained and evidence extracted")

    # CatBoost
    cat_model = cb.CatBoostClassifier(**CONFIG["CAT_PARAMS"])
    cat_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        use_best_model=True
    )
    cat_probs = cat_model.predict_proba(X_test)
    cat_evidence = get_model_activations(cat_model, X_test, model_type="cat")
    print("✅ CatBoost trained and evidence extracted")

    # TabTransformer
    tab_model = init_tab_transformer(CONFIG["DEVICE"])
    if CONFIG["LOAD_SAVED_MODEL"] and os.path.exists(CONFIG["MODEL_SAVE_PATH"]):
        tab_model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location=CONFIG["DEVICE"]))
        print(f"✅ Loaded TabTransformer from {CONFIG['MODEL_SAVE_PATH']}")
    tab_model.eval()

    test_dataset = TabDataset(X_test, y_test, CONFIG["DEVICE"])
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    tab_probs = []
    with torch.no_grad():
        for x_categ, x_cont, _ in test_dataloader:
            logits = tab_model(x_categ, x_cont)
            tab_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
    tab_probs = np.array(tab_probs)
    tab_evidence = get_model_activations(tab_model, X_test, model_type="tab", device=CONFIG["DEVICE"])
    print("✅ TabTransformer ready and evidence extracted")

    # 3. Model Weighting
    model_weights = calculate_weights_by_val_accuracy(
        xgb_model, cat_model, tab_model,
        X_val, y_val, CONFIG["DEVICE"]
    )
    CONFIG["SUBJECTIVE_LOGIC"]["model_weights"] = model_weights
    print(f"✅ Model weights computed: normalized={model_weights} | raw={val_metrics['raw_weights']}")

    # 4. Evidence -> B-U
    print("\n🔮 Converting evidence to belief-uncertainty (B-U) pairs...")
    sl = SubjectiveLogic()
    n_classes = CONFIG["N_CLASSES_5"]
    xgb_b, xgb_u = sl.evidence_to_belief_uncertainty(xgb_evidence, n_classes)
    cat_b, cat_u = sl.evidence_to_belief_uncertainty(cat_evidence, n_classes)
    tab_b, tab_u = sl.evidence_to_belief_uncertainty(tab_evidence, n_classes)

    print("✅ Correcting B-U pairs using validation accuracy...")
    xgb_w = val_metrics["raw_weights"][0]
    cat_w = val_metrics["raw_weights"][1]
    tab_w = val_metrics["raw_weights"][2]

    xgb_bc, xgb_uc = sl.weight_correct_bu(xgb_b, xgb_u, xgb_w)
    cat_bc, cat_uc = sl.weight_correct_bu(cat_b, cat_u, cat_w)
    tab_bc, tab_uc = sl.weight_correct_bu(tab_b, tab_u, tab_w)
    print("✅ B-U correction completed")

    # 5. DS Fusion
    views_b = [xgb_bc, cat_bc, tab_bc]
    views_u = [xgb_uc, cat_uc, tab_uc]
    fused_b, fused_u = sl.ds_multi_view_fusion(views_b, views_u)
    print("✅ Weighted DS evidence fusion completed")

    # 6. Decision
    fused_preds_5cls = sl.bu_to_decision(fused_b)
    print("✅ 5-class decision made from fused belief")

    # 7. Sample Splitting by Uncertainty
    print("\n📊 Splitting samples by uncertainty...")
    u_values = np.array([fused_u[i, fused_preds_5cls[i]] for i in range(len(fused_preds_5cls))])
    certain_mask = u_values <= CONFIG["UNCERTAINTY_THRESHOLD"]
    uncertain_mask = ~certain_mask

    y_cert_true = y_test[certain_mask]
    y_cert_pred = fused_preds_5cls[certain_mask]

    y_uncert_true_5 = y_test[uncertain_mask]
    y_uncert_pred_5 = fused_preds_5cls[uncertain_mask]
    y_uncert_true_3 = map_labels(y_uncert_true_5, CONFIG["LABEL_MAP_5TO3"])
    y_uncert_pred_3 = map_labels(y_uncert_pred_5, CONFIG["LABEL_MAP_5TO3"])
    print("✅ Sample splitting completed")

    # 8. Metrics
    print("\n📈 Calculating evaluation metrics...")
    xgb_preds = np.argmax(xgb_probs, axis=1)
    cat_preds = np.argmax(cat_probs, axis=1)
    tab_preds = np.argmax(tab_probs, axis=1)

    xgb_metrics = calculate_cls_metrics(y_test, xgb_preds, CONFIG["N_CLASSES_5"])
    cat_metrics = calculate_cls_metrics(y_test, cat_preds, CONFIG["N_CLASSES_5"])
    tab_metrics = calculate_cls_metrics(y_test, tab_preds, CONFIG["N_CLASSES_5"])

    fused_metrics_5 = calculate_cls_metrics(y_test, fused_preds_5cls, CONFIG["N_CLASSES_5"])
    cert_metrics_5 = calculate_cls_metrics(y_cert_true, y_cert_pred, CONFIG["N_CLASSES_5"])
    uncert_metrics_3 = calculate_cls_metrics(y_uncert_true_3, y_uncert_pred_3, CONFIG["N_CLASSES_3"])
    print("✅ Metrics calculated")

    # 9. Output
    print("\n" + "=" * 120)
    print("5-Class Performance of Base Models")
    print("=" * 120)
    print(f"XGBoost: Acc={xgb_metrics['overall_accuracy']:.4f}, Precision={xgb_metrics['macro_precision']:.4f}, Recall={xgb_metrics['macro_recall']:.4f}")
    print(f"CatBoost: Acc={cat_metrics['overall_accuracy']:.4f}, Precision={cat_metrics['macro_precision']:.4f}, Recall={cat_metrics['macro_recall']:.4f}")
    print(f"TabTransformer: Acc={tab_metrics['overall_accuracy']:.4f}, Precision={tab_metrics['macro_precision']:.4f}, Recall={tab_metrics['macro_recall']:.4f}")

    print("\n" + "=" * 120)
    print("5-Class Performance after Weighted DS Fusion (d=0)")
    print("=" * 120)
    print(f"Raw Val Acc: XGB={xgb_w:.4f}, CatBoost={cat_w:.4f}, TabTransformer={tab_w:.4f}")
    print(f"Fused Model: Acc={fused_metrics_5['overall_accuracy']:.4f}, Precision={fused_metrics_5['macro_precision']:.4f}, Recall={fused_metrics_5['macro_recall']:.4f}")

    print("\n" + "=" * 120)
    print("Hierarchical Classification Results (Threshold = 0.0834)")
    print("=" * 120)
    if len(y_cert_true) > 0:
        print(f"✅ Certain Samples (u≤0.0834) 5-class: N={len(y_cert_true)}, Acc={cert_metrics_5['overall_accuracy']:.4f}, Precision={cert_metrics_5['macro_precision']:.4f}, Recall={cert_metrics_5['macro_recall']:.4f}")
    else:
        print("✅ Certain Samples: None")

    if len(y_uncert_true_3) > 0:
        print(f"❌ Uncertain Samples (u>0.0834) 3-class: N={len(y_uncert_true_3)}, Acc={uncert_metrics_3['overall_accuracy']:.4f}, Precision={uncert_metrics_3['macro_precision']:.4f}, Recall={uncert_metrics_3['macro_recall']:.4f}")
    else:
        print("❌ Uncertain Samples: None")

    print("\n" + "=" * 120)
    print(f"Belief & Uncertainty Summary (Dirichlet prior c = {CONFIG['SUBJECTIVE_LOGIC']['dirichlet_prior']}, d=0)")
    print("=" * 120)
    summary = sl.summarize_belief_uncertainty(fused_b, fused_u)
    for key, stats in summary.items():
        if "Examples" in key:
            print(f"\n{key}:")
            print(f"  Belief: {stats['Belief']}")
            print(f"  Uncertainty: {stats['Uncertainty']}")
        else:
            print(f"\n{key}:")
            print(f"  Mean={stats['mean']:.4f} | Std={stats['std']:.4f} | Max={stats['max']:.4f} | Min={stats['min']:.4f}")

    print("\n" + "=" * 120)
    print("Top 5 Samples: Belief & Uncertainty after Weighted DS Fusion")
    print("=" * 120)
    for i in range(min(5, len(y_test))):
        pred = fused_preds_5cls[i]
        true = y_test[i]
        b_val = fused_b[i, pred].round(4)
        u_val = fused_u[i, pred].round(4)
        sum_bu = np.sum(fused_b[i]) + u_val
        print(f"Sample {i+1}: True={true} | Pred={pred} | B={b_val} | U={u_val} | Sum(B)+U={sum_bu:.4f}")
    print("=" * 120)

    # 10. Save results
    print(f"\n📝 Saving test results to: {CONFIG['OUTPUT_PATH']['sample_triple_csv']}")
    records = []
    for idx in range(len(y_test)):
        true5 = y_test[idx]
        true3 = CONFIG["LABEL_MAP_5TO3"][true5]
        pred5 = fused_preds_5cls[idx]
        pred3 = CONFIG["LABEL_MAP_5TO3"][pred5]
        u = fused_u[idx, pred5]
        certain = "Yes" if u <= CONFIG["UNCERTAINTY_THRESHOLD"] else "No"
        b_val = fused_b[idx, pred5].round(6)
        u_val = fused_u[idx, pred5].round(6)
        b_all = ",".join([f"{v:.4f}" for v in fused_b[idx]])
        u_all = ",".join([f"{v:.4f}" for v in fused_u[idx]])

        records.append({
            "sample_idx": idx,
            "true_5class": true5,
            "true_3class": true3,
            "pred_5class": pred5,
            "pred_3class": pred3,
            "uncertainty_u": u.round(6),
            "is_certain": certain,
            "belief_b": b_val,
            "uncertainty_u_sample": u_val,
            "sum_b_u": (np.sum(fused_b[idx]) + u_val).round(6),
            "belief_all_classes": b_all,
            "uncertainty_all_classes": u_all,
            "xgb_val_acc": xgb_w,
            "cat_val_acc": cat_w,
            "tab_val_acc": tab_w,
            "xgb_pred": xgb_preds[idx],
            "cat_pred": cat_preds[idx],
            "tab_pred": tab_preds[idx],
            "dirichlet_prior_c": CONFIG["SUBJECTIVE_LOGIC"]["dirichlet_prior"],
            "d_zero": "Yes"
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(CONFIG["OUTPUT_PATH"]["sample_triple_csv"], index=False, encoding="utf-8-sig")
    print(f"✅ Results saved: {len(df_out)} records")