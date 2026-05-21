# -*- coding: utf-8 -*-
"""SECOM Streamlit dashboard — 3 tabs: experiments, features, misclassification.

Run via: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

from app.dashboard_utils import (
    compare_feature_groups,
    enrich_predictions,
    get_feature_importance,
    label_noise_candidates,
    load_final_bundle,
    load_leaderboard_with_stats,
    load_paths,
    load_raw_data,
    method_labels,
)

st.set_page_config(page_title="SECOM Defect Prediction", layout="wide", page_icon="📊")
st.title("SECOM 반도체 공정 불량 예측 대시보드")

PATHS = load_paths()
LABELS_KO = method_labels()


@st.cache_data(show_spinner="실험 결과 로딩…")
def cached_leaderboard():
    return load_leaderboard_with_stats()


@st.cache_data(show_spinner="Feature Importance 계산 중…")
def cached_importance():
    return get_feature_importance(n_repeats=6, max_samples=600)


@st.cache_data
def cached_raw():
    X, y, meta, test_idx = load_raw_data()
    return X, y, meta, test_idx


tab_exp, tab_feat, tab_err = st.tabs(
    ["① 실험 결과", "② 입력 변수 분석", "③ 오분류 심화 분석"]
)

with tab_exp:
    st.header("실험 결과 및 통계 검증")
    lb = cached_leaderboard()

    if lb.empty:
        st.warning("`run_quick_experiments.py`를 먼저 실행하세요.")
    else:
        display = lb.copy()
        for col in ["prep_id", "sample_id", "model_id"]:
            display[f"{col}_설명"] = display[col].map(LABELS_KO)
        display["significant"] = display.get("significant", False).astype(str)

        st.subheader("실험 결과 테이블 (OOF 5-Fold + Wilcoxon vs Dummy)")
        st.dataframe(
            display[
                [
                    "experiment_id",
                    "prep_id",
                    "sample_id",
                    "model_id",
                    "pr_auc_mean",
                    "pr_auc_std",
                    "recall_mean",
                    "f1_mean",
                    "balanced_acc_mean",
                    "mean_delta",
                    "p_adj",
                    "significant",
                ]
            ].round(4),
            width="stretch",
            hide_index=True,
        )

        fig = px.bar(
            lb.sort_values("pr_auc_mean"),
            x="pr_auc_mean",
            y="experiment_id",
            orientation="h",
            error_x="pr_auc_std",
            color="model_id",
            title="OOF PR-AUC (± std, 5-fold)",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("방법론 해석")
        best = lb.iloc[0]
        st.markdown(
            f"""
**최고 OOF 설정:** `{best['experiment_id']}` (PR-AUC **{best['pr_auc_mean']:.3f}** ± {best['pr_auc_std']:.3f})

| 축 | 관찰 | 해석 |
|----|------|------|
| **전처리** | P4(Winsor+Robust) > P6(MI40) > P2(결측제거) > P0 | 이상치 완화(Winsor)가 고차원 센서 노이즈에 유리. MI 40특성은 차원 축소 효과 있으나 단독으론 P4보다 약함 |
| **샘플링** | S1(weight) ≥ S2(SMOTE) (트리 모델) | 불균형(6.6% Fail)에서 가중치가 안정적. SMOTE는 Recall↑ but PR-AUC↓ 경향 (LR 기준) |
| **모델** | XGB ≈ RF > LGBM > LR >> Dummy | 비선형 트리 계열이 센서 상호작용 포착. LR은 Recall 높지만 Precision 낮아 F1/PR-AUC 열위 |
| **통계검정** | Holm 보정 후 대부분 p_adj > 0.05 | fold=5라 검정력 제한. 다만 Dummy 대비 mean Δ PR-AUC **+0.06~+0.14** 는 실질 개선 |

**주의:** P2+LR+SMOTE는 Recall **0.97**처럼 보이나 Precision/F1이 낮아 **과다 예측(FP)** 패턴 — 운영 비용 관점에서 단독 채택 비권장.
            """
        )

        cfg_path = PATHS["models"] / "final_config.json"
        if cfg_path.exists():
            tm = json.loads(cfg_path.read_text(encoding="utf-8")).get("test_metrics", {})
            st.info(
                f"Hold-out Test (최종 모델 1회): PR-AUC **{tm.get('pr_auc', 0):.3f}**, "
                f"Recall **{tm.get('recall', 0):.3f}**, F1 **{tm.get('f1', 0):.3f}**"
            )

        st.subheader("향후 실험 제안 (10회 제한 이후)")
        st.markdown(
            """
1. **비용 민감 학습** — FN 가중치 3~5배 `scale_pos_weight` 그리드 (불량 미검출 최소화)
2. **앙상블** — OOF 상위 P4-XGB + P4-RF soft voting
3. **결측 패턴 특성** — 센서 그룹별 missing rate + `MissingIndicator` (P8 확장)
4. **시간 특성** — `Time` 컬럼 기반 lag/rolling (공정 지연 반영)
5. **캘리브레이션** — Platt/Isotonic 후 임계값 운영 KPI별 튜닝
6. **하이퍼파라미터** — XGB `max_depth`, `scale_pos_weight`만 3×3 소규모 탐색
            """
        )

        with st.expander("모델 고도화 재계획 (실험 인사이트 기반)"):
            st.markdown(
                """
**Phase A (1주)** — 임계값·비용
- FN 비용 가중 KPI 정의 → threshold sweep 대시보드 연동 운영값 확정

**Phase B (2주)** — 특성
- Permutation Top-20 센서군 도메인 매핑 + 결측률 파생변수
- P4 전처리 + MI 40 hybrid 파이프라인 1회 실험

**Phase C (2주)** — 모델
- XGB + RF 앙상블, FN-weighted XGB
- 오분류 라벨 노이즈 수동 검토 큐 (탭3 suspect)

**성공 기준:** Test Recall ≥ 0.40, PR-AUC ≥ 0.22, FP rate ≤ 15%
                """
            )

with tab_feat:
    st.header("입력 변수 분석")
    cfg, preds, bundle = load_final_bundle()
    model_id = cfg.get("spec", {}).get("model_id", "")

    if "cnn" in model_id.lower() or "resnet" in model_id.lower():
        st.info("딥러닝 모델: Feature Importance 대신 **Grad-CAM** 시각화를 사용합니다. (현재 Tabular XGB — 아래 분석 적용)")
    else:
        st.caption("Tabular 모델: Permutation Importance + Pass/Fail 통계 비교 (Grad-CAM 해당 없음)")

    try:
        importance, spec = cached_importance()
        X, y, meta, test_idx = cached_raw()

        st.subheader("Permutation Importance (Top 25)")
        top = importance.head(25)
        fig_imp = px.bar(
            top.sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title="평균 Permutation Importance (PR-AUC 감소량)",
        )
        st.plotly_chart(fig_imp, width="stretch")

        st.subheader("중요 vs 비중요 변수 — Pass/Fail 통계 비교")
        cmp = compare_feature_groups(X, y, importance, top_k=12, bottom_k=12)
        if not cmp.empty:
            st.dataframe(cmp.round(4), width="stretch", hide_index=True)

            hi = cmp[cmp["group"] == "high_importance"]
            lo = cmp[cmp["group"] == "low_importance"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**고중요 변수** — Fail/Pass 평균 차이 Top 5")
                if not hi.empty:
                    t = hi.nlargest(5, "mean_diff")
                    fig_h = px.bar(t, x="feature", y="mean_diff", title="|mean_diff| (High importance)")
                    st.plotly_chart(fig_h, width="stretch")
            with c2:
                st.markdown("**저중요 변수** — 분류력 약함")
                if not lo.empty:
                    st.caption(f"평균 p-value: {lo['p_value'].mean():.3f} (높을수록 Pass/Fail 구분 약함)")

        st.subheader("심화 EDA — 상위 변수")
        feat_sel = st.selectbox(
            "탐색할 센서(특성)",
            options=top["feature"].head(15).tolist(),
        )
        if feat_sel:
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_d = px.histogram(
                    X.assign(label=y.map({0: "Pass", 1: "Fail"})),
                    x=feat_sel,
                    color="label",
                    barmode="overlay",
                    opacity=0.6,
                    title=f"{feat_sel} 분포",
                )
                st.plotly_chart(fig_d, width="stretch")
            with c2:
                miss = X[feat_sel].isna().groupby(y).mean()
                fig_m = px.bar(
                    x=["Pass", "Fail"],
                    y=[miss.get(0, 0), miss.get(1, 0)],
                    title=f"{feat_sel} 결측률",
                )
                st.plotly_chart(fig_m, width="stretch")
            with c3:
                if feat_sel in X.columns:
                    corr_cols = importance.head(8)["feature"].tolist()
                    sub = X[corr_cols].apply(pd.to_numeric, errors="coerce")
                    fig_c = px.imshow(
                        sub.corr(),
                        title="Top 특성 상관행렬",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                    )
                    st.plotly_chart(fig_c, width="stretch")

        st.subheader("추가 인사이트")
        st.markdown(
            """
- **고중요 센서**는 Fail 그룹에서 평균·분산이 크게 어긋나는 경향 → 공정 이상 직접 신호 가능성
- **저중요 센서**는 Pass/Fail 분포가 유사 → 차원 축소·노이즈 컬럼 제거 후보
- **결측 패턴**: Fail에서 결측률이 높은 변수는 별도 binary feature로 보존 가치
- **상관 클러스터** 내 중복 센서는 1개만 남기는 군집 기반 선택 검토
            """
        )

    except Exception as e:
        st.error(f"변수 분석 로드 실패: {e}")
        st.code("python run_quick_experiments.py 후 models/final_model.joblib 확인")

with tab_err:
    st.header("오분류 데이터 심화 분석")
    cfg, preds, bundle = load_final_bundle()

    if not preds.empty:
        X, y, meta, test_idx = cached_raw()
        enriched = enrich_predictions(preds, cfg)
        thr = cfg.get("threshold", 0.05)

        errors = enriched[enriched["is_error"]].copy()
        near = errors[errors["near_threshold"]].sort_values("abs_margin")
        hard = errors[errors["high_confidence_error"]].sort_values("confidence", ascending=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test 오분류", len(errors))
        c2.metric("임계값 근접 오류", len(near))
        c3.metric("고신뢰도 오류", len(hard))
        c4.metric("임계값", f"{thr:.2f}")

        st.subheader("오분류 유형별 분포")
        fig_e = px.scatter(
            enriched,
            x="y_prob",
            y="confidence",
            color="error_type",
            symbol="y_true",
            title="예측 확률 vs 신뢰도 (Test)",
            hover_data=["idx"],
        )
        fig_e.add_vline(x=thr, line_dash="dash", line_color="red")
        st.plotly_chart(fig_e, width="stretch")

        view = st.radio(
            "검토할 케이스",
            ["전체 오분류", "임계값 근접 (아쉬운 오류)", "고신뢰도 오류 (확신 있게 틀림)", "FN만", "FP만"],
            horizontal=True,
        )
        if view == "임계값 근접 (아쉬운 오류)":
            show = near
        elif view == "고신뢰도 오류 (확신 있게 틀림)":
            show = hard
        elif view == "FN만":
            show = errors[errors["error_type"] == "FN"]
        elif view == "FP만":
            show = errors[errors["error_type"] == "FP"]
        else:
            show = errors

        show = show.sort_values("abs_margin")
        st.dataframe(
            show[["idx", "y_true", "y_prob", "y_pred", "error_type", "abs_margin", "confidence"]],
            width="stretch",
            hide_index=True,
        )

        if len(show) > 0:
            sel_idx = st.selectbox("샘플 상세 보기 (idx)", show["idx"].tolist())
            row = show[show["idx"] == sel_idx].iloc[0]
            sample_X = X.loc[sel_idx].astype(float)
            st.markdown(
                f"**idx={sel_idx}** | true={'Fail' if row['y_true']==1 else 'Pass'} | "
                f"prob={row['y_prob']:.4f} | {row['error_type']}"
            )
            if "Time" in meta.columns:
                st.write(f"측정 시각: {meta.loc[sel_idx, 'Time']}")

            imp, _ = cached_importance()
            top_feats = imp.head(12)["feature"].tolist()
            vals = sample_X[top_feats].to_frame("value").T
            st.dataframe(vals, width="stretch")

            if row["error_type"] == "FN":
                st.warning(
                    "FN: 불량인데 Pass로 예측 — prob가 임계값보다 낮음. "
                    "센서가 '정상처럼' 보이거나 라벨/공정 지연 이슈 가능"
                )
            else:
                st.warning("FP: 정상인데 Fail로 예측 — 과민 반응 센서 또는 임계값 과도")

        st.subheader("근접 vs 확신 오류 비교")
        if len(near) > 0 and len(hard) > 0:
            imp, _ = cached_importance()
            top_f = imp.head(10)["feature"].tolist()
            near_mean = X.loc[near["idx"], top_f].astype(float).mean()
            hard_mean = X.loc[hard["idx"], top_f].astype(float).mean()
            comp_df = pd.DataFrame({"near_threshold": near_mean, "high_confidence": hard_mean})
            st.dataframe(comp_df.round(3))
            st.caption("Top 10 중요 변수 평균 — 근접 오류는 경계선 패턴, 확신 오류는 강한 이상 신호 가능")

        st.subheader("라벨링 오류 가능성 조사")
        suspects = label_noise_candidates(X, y, errors)
        if not suspects.empty:
            st.dataframe(suspects, width="stretch")
            st.warning(f"라벨 의심 샘플 {len(suspects)}건 — 수동 검토 권장")
        else:
            st.success("자동 휴리스틱 기준 라벨 강한 불일치 후보 없음 (오분류 subset)")

        st.subheader("모델 고도화 아이디어 (오분류 기반)")
        fn_n, fp_n = (errors["error_type"] == "FN").sum(), (errors["error_type"] == "FP").sum()
        st.markdown(
            f"""
| 관찰 | 제안 |
|------|------|
| FN **{fn_n}**건 / FP **{fp_n}**건 | FN 비용이 크면 threshold↓ 또는 `scale_pos_weight`↑ |
| 임계값 근접 **{len(near)}**건 | 확률 캘리브레이션 + 운영별 threshold 시나리오 |
| 고신뢰 오류 **{len(hard)}**건 | 해당 센서 구간 Winsor 강화 또는 센서 블랙리스트 검토 |
| 라벨 의심 | suspect idx 현장/공정 로그 대조 후 학습 제외 또는 soft label |

**다음 스프린트:** 오분류 FN 클러스터별 **전용 binary sensor rule** → XGB 잔차 학습 (stacking lite)
            """
        )

        with st.expander("Test 혼동행렬 & PR 곡선"):
            thr_sl = st.slider("Threshold (what-if)", 0.05, 0.95, float(thr), 0.01, key="err_thr")
            y_t, y_p = enriched["y_true"], enriched["y_prob"]
            y_pr = (y_p >= thr_sl).astype(int)
            cm = confusion_matrix(y_t, y_pr, labels=[0, 1])
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Pred Pass", "Pred Fail"],
                    y=["True Pass", "True Fail"],
                    text=cm,
                    texttemplate="%{text}",
                )
            )
            st.plotly_chart(fig_cm, width="stretch")
            prec, rec, _ = precision_recall_curve(y_t, y_p)
            fig_pr = go.Figure(go.Scatter(x=rec, y=prec, mode="lines"))
            fig_pr.update_layout(title="PR Curve (Test)", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, width="stretch")
    else:
        st.warning("`python run_quick_experiments.py` 실행 후 predictions_test.csv 필요")
