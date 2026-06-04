"""
통합 VaR 리스크 분석기 v2
- 탭1: 단일 종목 VaR (기존 기능 유지)
- 탭2: 포트폴리오 VaR (신규 추가)
  - 여러 종목 + 수량 입력
  - 상관관계 반영한 모수적 포트폴리오 VaR
  - 역사적 포트폴리오 VaR
  - 몬테카를로 포트폴리오 VaR (Cholesky 분해)
  - 종목별 기여 VaR
  - 상관관계 히트맵
  - 포트폴리오 구성 파이차트
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# ─────────────────────────────────────────────
# 1. 웹 페이지 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="통합 VaR 리스크 분석기", layout="wide")
st.title("📊 3대 방법론 통합 VaR 분석 플랫폼")

# ─────────────────────────────────────────────
# 2. 공통 사이드바
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ 공통 분석 조건")
start_date = st.sidebar.date_input("데이터 시작일", pd.to_datetime("2022-01-01"))
end_date   = st.sidebar.date_input("데이터 종료일", pd.Timestamp.today())
confidence_level = st.sidebar.selectbox("신뢰수준", [0.95, 0.99], index=0)
holding_period   = st.sidebar.number_input("보유 기간 (일)", min_value=1, max_value=30, value=1)

if holding_period > 1:
    st.sidebar.warning(
        f"⚠️ 보유기간 {holding_period}일 적용 시 √T 스케일링(IID 가정)이 사용됩니다. "
        "실제 시장에서는 과소 추정될 수 있습니다."
    )

# ─────────────────────────────────────────────
# 3. 공통 데이터 로딩 함수
# ─────────────────────────────────────────────
@st.cache_data
def load_single(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        return None
    close = df["Close"][ticker] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    arith  = close.pct_change().dropna()
    log_r  = np.log(close / close.shift(1)).dropna()
    return arith, log_r, close

@st.cache_data
def load_portfolio(tickers, start, end):
    """여러 종목 종가 DataFrame 반환"""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices = prices.dropna()
    return prices

# ─────────────────────────────────────────────
# 3-1. 통화 자동 감지 함수
# ─────────────────────────────────────────────
def get_currency_symbol(tickers: list) -> str:
    """
    종목 티커 목록 기반 통화 기호 자동 감지
    - .KS / .KQ → 한국 주식 → ₩
    - 그 외 → 미국 주식 → $
    - 혼합 포트폴리오 → '혼합'
    """
    kr = [t for t in tickers if t.endswith(".KS") or t.endswith(".KQ")]
    us = [t for t in tickers if not t.endswith(".KS") and not t.endswith(".KQ")]
    if kr and not us:
        return "₩"
    elif us and not kr:
        return "$"
    else:
        return "혼합"

def fmt_money(value: float, symbol: str) -> str:
    """통화 기호 + 금액 포맷 반환"""
    if symbol == "$":
        return f"${value:,.2f}"
    elif symbol == "₩":
        return f"₩{value:,.0f}"
    else:
        return f"{value:,.2f} (혼합통화)"

# ─────────────────────────────────────────────
# 4. VaR 계산 함수 (단일 종목용)
# ─────────────────────────────────────────────
def calc_parametric_var(returns, conf, days, inv):
    mu    = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z     = norm.ppf(conf)
    return float((z * sigma - mu) * np.sqrt(days) * inv)

def calc_historical_var(returns, conf, days, inv):
    pct     = (1 - conf) * 100
    var_ret = np.percentile(returns, pct)
    return float(max(-var_ret, 0) * np.sqrt(days) * inv)

def calc_monte_carlo(log_returns, conf, days, inv, iterations=10_000):
    mu    = np.mean(log_returns)
    sigma = np.std(log_returns, ddof=1)
    np.random.seed(42)
    rn   = np.random.normal(0, 1, (days, iterations))
    lr   = np.sum((mu - 0.5 * sigma**2) + sigma * rn, axis=0)
    losses = -(np.exp(lr) - 1) * inv
    return float(np.percentile(losses, conf * 100)), losses

# ─────────────────────────────────────────────
# 5. 포트폴리오 VaR 계산 함수
# ─────────────────────────────────────────────
def calc_portfolio_parametric_var(returns_df, weights, conf, days):
    """
    모수적 포트폴리오 VaR
    - 분산-공분산 행렬 + 상관관계 반영
    - weights: 금액 비중 배열 (합계=1)
    """
    cov    = returns_df.cov().values
    mu_vec = returns_df.mean().values
    port_mu    = np.dot(weights, mu_vec)
    port_sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    z = norm.ppf(conf)
    return float((z * port_sigma - port_mu) * np.sqrt(days))

def calc_portfolio_historical_var(returns_df, weights, conf, days):
    """
    역사적 포트폴리오 VaR
    - 과거 포트폴리오 수익률 직접 계산
    """
    port_returns = returns_df.dot(weights)
    pct = (1 - conf) * 100
    var_ret = np.percentile(port_returns, pct)
    return float(max(-var_ret, 0) * np.sqrt(days))

def calc_portfolio_montecarlo_var(returns_df, weights, conf, days, iterations=10_000):
    """
    몬테카를로 포트폴리오 VaR
    - Cholesky 분해로 종목 간 상관관계 반영
    """
    log_ret = np.log(1 + returns_df)
    mu_vec  = log_ret.mean().values
    cov     = log_ret.cov().values
    n       = len(weights)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # 행렬이 양정치가 아닐 경우 근사 처리
        cov += np.eye(n) * 1e-8
        L = np.linalg.cholesky(cov)

    np.random.seed(42)
    # 각 종목별 다기간 로그수익률 (Cholesky로 상관관계 반영)
    log_paths = np.zeros((n, iterations))
    drift = mu_vec[:, None] - 0.5 * np.diag(cov)[:, None]  # shape: (n, 1)
    for t in range(days):
        Z_t = np.random.normal(0, 1, (n, iterations))  # shape: (n, iterations)
        corr_Z_t = L @ Z_t                              # shape: (n, iterations)
        log_paths += drift + corr_Z_t

    asset_returns = np.exp(log_paths) - 1  # shape: (n, iterations)
    port_returns  = np.dot(weights, asset_returns)  # shape: (iterations,)
    losses = -port_returns
    var_val = np.percentile(losses, conf * 100)
    return float(var_val), losses

def calc_component_var(returns_df, weights, conf, days):
    """
    종목별 기여 VaR (Component VaR)
    - 각 종목이 포트폴리오 VaR에 기여하는 비율
    """
    cov    = returns_df.cov().values
    sigma_p = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    z = norm.ppf(conf)
    # 한계 VaR (Marginal VaR)
    marginal = z * np.dot(cov, weights) / sigma_p * np.sqrt(days)
    component = marginal * weights
    return component

# ─────────────────────────────────────────────
# 6. 탭 구성
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 단일 종목 VaR", "📁 포트폴리오 VaR"])

# ══════════════════════════════════════════════
# TAB 1: 단일 종목 VaR (기존 기능 그대로)
# ══════════════════════════════════════════════
with tab1:
    st.markdown("주식 하나를 선택해 **모수적 / 역사적 / 몬테카를로** 3가지 방식으로 VaR을 측정합니다.")

    ticker     = st.text_input("주식 티커 입력 (예: AAPL, TSLA, 005930.KS)", "AAPL", key="single_ticker")
    investment = st.number_input("총 투자 원금 (원/달러)", min_value=100_000, value=10_000_000,
                                  step=1_000_000, format="%d", key="single_inv")

    result = load_single(ticker, start_date, end_date)

    if result is not None:
        arith_returns, log_returns, prices = result

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("분석 데이터 수", f"{len(arith_returns)} 일")
        col2.metric("평균 일일 수익률 (산술)", f"{arith_returns.mean()*100:.4f}%")
        col3.metric("평균 일일 수익률 (로그)", f"{log_returns.mean()*100:.4f}%")
        col4.metric("일일 변동성 (ddof=1)", f"{log_returns.std(ddof=1)*100:.2f}%")

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=prices.index, y=prices.values, mode="lines",
                                        name="종가", line=dict(color="#38bdf8", width=2)))
        fig_price.update_layout(title=f"{ticker} 주가 추이", template="plotly_dark",
                                  xaxis_title="날짜", yaxis_title="가격")
        st.plotly_chart(fig_price, use_container_width=True)

        p_var = calc_parametric_var(arith_returns, confidence_level, holding_period, investment)
        h_var = calc_historical_var(arith_returns, confidence_level, holding_period, investment)
        m_var, mc_losses = calc_monte_carlo(log_returns, confidence_level, holding_period, investment)

        st.markdown("---")
        st.subheader("🚨 3대 방법론별 VaR 결과")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.error("**모수적 VaR**")
            st.markdown(f"### {p_var:,.0f} 원")
            st.caption("정규분포 + 표본 표준편차(ddof=1)")
        with c2:
            st.error("**역사적 VaR**")
            st.markdown(f"### {h_var:,.0f} 원")
            st.caption("과거 실제 수익률 분포 기반")
        with c3:
            st.error("**몬테카를로 VaR**")
            st.markdown(f"### {m_var:,.0f} 원")
            st.caption("로그수익률 GBM 10,000회")

        st.info(f"💡 {confidence_level*100:.0f}% 확률로 다음 {holding_period}일간 최대 손실이 위 금액을 넘지 않을 것으로 추정됩니다.")

        st.markdown("---")
        fig_bar = go.Figure(go.Bar(
            x=["모수적", "역사적", "몬테카를로"],
            y=[p_var, h_var, m_var],
            marker_color=["#38bdf8", "#34d399", "#f97316"],
            text=[f"{v:,.0f}" for v in [p_var, h_var, m_var]],
            textposition="outside"
        ))
        fig_bar.update_layout(title="3대 방법론 VaR 비교", template="plotly_dark",
                               yaxis_title="VaR (손실 추정액)", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        mu_a = arith_returns.mean()
        sig_a = arith_returns.std(ddof=1)
        x_range = np.linspace(arith_returns.min(), arith_returns.max(), 300)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=arith_returns, nbinsx=80, histnorm="probability density",
                                         name="실제 수익률", marker_color="#64748b", opacity=0.7))
        fig_dist.add_trace(go.Scatter(x=x_range, y=norm.pdf(x_range, mu_a, sig_a),
                                       mode="lines", name="정규분포 근사",
                                       line=dict(color="#facc15", width=2, dash="dash")))
        var_line = -(p_var / investment / np.sqrt(holding_period))
        fig_dist.add_vline(x=var_line, line_width=2, line_dash="dash", line_color="#ef4444")
        fig_dist.update_layout(title="수익률 분포 vs 정규분포 가정", template="plotly_dark",
                                xaxis_title="일일 수익률", yaxis_title="확률 밀도")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=mc_losses, nbinsx=100, name="손실 시나리오",
                                         marker_color="#64748b"))
        fig_hist.add_vline(x=m_var, line_width=3, line_dash="dash", line_color="#ef4444")
        fig_hist.add_annotation(x=m_var, y=10, text="MC VaR 임계점", showarrow=True,
                                  arrowhead=1, ax=120, ay=-50, font=dict(color="#ef4444"))
        fig_hist.update_layout(title="몬테카를로 손실 분포", template="plotly_dark",
                                xaxis_title="손실액", yaxis_title="시나리오 횟수")
        st.plotly_chart(fig_hist, use_container_width=True)

        with st.expander("📚 방법론 요약 및 주의사항"):
            st.markdown("""
| 방법론 | 핵심 가정 | 장점 | 한계 |
|--------|-----------|------|------|
| **모수적** | 정규분포 | 계산 단순 | 팻테일 과소 반영 |
| **역사적** | 과거=미래 | 분포 가정 없음 | 과거 데이터 종속 |
| **몬테카를로** | GBM | 복잡한 경로 시뮬레이션 | 파라미터 민감 |
> ⚠️ √T 스케일링은 IID 가정 하에서만 유효합니다.
            """)
    else:
        st.error("❌ 데이터를 불러오지 못했습니다. 티커 또는 날짜 범위를 확인해 주세요.")


# ══════════════════════════════════════════════
# TAB 2: 포트폴리오 VaR (신규 기능)
# ══════════════════════════════════════════════
with tab2:
    st.markdown("여러 종목과 보유 수량을 입력하면 **상관관계를 반영한 포트폴리오 전체 VaR**을 계산합니다.")

    st.subheader("📋 종목 구성 입력")
    st.caption("티커는 쉼표로 구분 | 수량은 각 종목 순서에 맞게 쉼표로 구분")

    col_a, col_b = st.columns(2)
    with col_a:
        tickers_input = st.text_input(
            "종목 티커 (예: AAPL, TSLA, MSFT)",
            "AAPL, TSLA, MSFT",
            key="port_tickers"
        )
    with col_b:
        qty_input = st.text_input(
            "보유 수량 (예: 10, 5, 20)",
            "10, 5, 20",
            key="port_qty"
        )

    run_btn = st.button("🚀 포트폴리오 VaR 분석 시작", type="primary")

    if run_btn:
        # 입력값 파싱
        try:
            tickers_list = [t.strip().upper() for t in tickers_input.split(",")]
            qty_list     = [float(q.strip()) for q in qty_input.split(",")]
        except Exception:
            st.error("❌ 입력 형식을 확인해 주세요. 티커와 수량은 쉼표로 구분해야 합니다.")
            st.stop()

        if len(tickers_list) != len(qty_list):
            st.error("❌ 종목 수와 수량 수가 일치하지 않습니다.")
            st.stop()

        if len(tickers_list) < 2:
            st.error("❌ 포트폴리오 VaR은 2개 이상의 종목이 필요합니다.")
            st.stop()

        with st.spinner("📡 데이터 불러오는 중..."):
            prices_df = load_portfolio(tickers_list, start_date, end_date)

        if prices_df is None or prices_df.empty:
            st.error("❌ 데이터를 불러오지 못했습니다. 티커를 확인해 주세요.")
            st.stop()

        # 실제로 데이터가 있는 종목만 필터
        valid_tickers = [t for t in tickers_list if t in prices_df.columns]
        if len(valid_tickers) < 2:
            st.error("❌ 유효한 종목이 2개 미만입니다.")
            st.stop()

        # 수량 재정렬
        qty_arr = np.array([qty_list[tickers_list.index(t)] for t in valid_tickers])
        prices_df = prices_df[valid_tickers]

        # 통화 기호 자동 감지
        currency = get_currency_symbol(valid_tickers)

        # 현재 주가 기반 종목별 평가 금액 계산
        latest_prices = prices_df.iloc[-1].values
        asset_values  = latest_prices * qty_arr          # 종목별 평가금액
        total_value   = asset_values.sum()               # 포트폴리오 총 금액
        weights       = asset_values / total_value       # 금액 비중

        # 수익률 계산
        returns_df = prices_df.pct_change().dropna()

        # ── 포트폴리오 요약 ─────────────────────────
        st.markdown("---")
        st.subheader("💼 포트폴리오 구성 요약")
        st.caption(f"💱 감지된 통화: **{'달러 ($)' if currency == '$' else '원화 (₩)' if currency == '₩' else '혼합 통화'}**")

        summary_df = pd.DataFrame({
            "종목": valid_tickers,
            "수량": qty_arr,
            f"현재 주가 ({currency})": [fmt_money(p, currency) for p in latest_prices],
            f"평가 금액 ({currency})": [fmt_money(v, currency) for v in asset_values],
            "비중 (%)": (weights * 100).round(2)
        })
        st.dataframe(summary_df.set_index("종목"), use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("포트폴리오 총 평가금액", fmt_money(total_value, currency))
        m2.metric("종목 수", f"{len(valid_tickers)} 개")

        # ── 파이차트: 포트폴리오 구성 ──────────────
        fig_pie = go.Figure(go.Pie(
            labels=valid_tickers,
            values=asset_values,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set2)
        ))
        fig_pie.update_layout(title="포트폴리오 구성 비중", template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

        # ── 상관관계 히트맵 ─────────────────────────
        st.markdown("---")
        st.subheader("🔗 종목 간 상관관계 히트맵")
        corr_matrix = returns_df.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=valid_tickers,
            y=valid_tickers,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            showscale=True
        ))
        fig_corr.update_layout(title="수익률 상관관계 (1=완전 양의 상관, -1=완전 음의 상관)",
                                template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("💡 상관관계가 낮을수록 분산 효과가 커져 포트폴리오 VaR이 개별 VaR 합계보다 작아집니다.")

        # ── 포트폴리오 VaR 계산 ─────────────────────
        st.markdown("---")
        st.subheader("🚨 포트폴리오 통합 VaR 결과")

        with st.spinner("계산 중..."):
            port_p_var = calc_portfolio_parametric_var(returns_df, weights, confidence_level, holding_period) * total_value
            port_h_var = calc_portfolio_historical_var(returns_df, weights, confidence_level, holding_period) * total_value
            port_m_var, port_mc_losses = calc_portfolio_montecarlo_var(returns_df, weights, confidence_level, holding_period)
            port_mc_var_amt = port_m_var * total_value
            port_mc_losses_amt = port_mc_losses * total_value

        r1, r2, r3 = st.columns(3)
        with r1:
            st.error("**모수적 포트폴리오 VaR**")
            st.markdown(f"### {fmt_money(port_p_var, currency)}")
            st.caption("공분산 행렬 + 상관관계 반영")
        with r2:
            st.error("**역사적 포트폴리오 VaR**")
            st.markdown(f"### {fmt_money(port_h_var, currency)}")
            st.caption("과거 포트폴리오 수익률 직접 계산")
        with r3:
            st.error("**몬테카를로 포트폴리오 VaR**")
            st.markdown(f"### {fmt_money(port_mc_var_amt, currency)}")
            st.caption("Cholesky 분해로 상관관계 반영")

        st.info(f"💡 {confidence_level*100:.0f}% 확률로 다음 {holding_period}일간 포트폴리오 최대 손실이 위 금액을 넘지 않을 것으로 추정됩니다.")

        # ── 분산 효과 비교 ──────────────────────────
        st.markdown("---")
        st.subheader("📉 분산 효과 확인")

        # 개별 종목 VaR 합계 (상관관계 무시)
        individual_vars = []
        for i, t in enumerate(valid_tickers):
            iv = calc_parametric_var(returns_df[t], confidence_level, holding_period, asset_values[i])
            individual_vars.append(iv)
        sum_individual = sum(individual_vars)
        diversification_benefit = sum_individual - port_p_var

        d1, d2, d3 = st.columns(3)
        d1.metric("개별 VaR 단순 합계", fmt_money(sum_individual, currency))
        d2.metric("포트폴리오 VaR (상관관계 반영)", fmt_money(port_p_var, currency))
        d3.metric("🎯 분산 효과 (절감액)", fmt_money(diversification_benefit, currency),
                  delta=f"-{diversification_benefit/sum_individual*100:.1f}%")

        st.caption("💡 분산 효과 = 개별 VaR 합계 - 포트폴리오 VaR. 상관관계가 낮을수록 절감액이 커집니다.")

        # ── 종목별 기여 VaR ─────────────────────────
        st.markdown("---")
        st.subheader("🔍 종목별 기여 VaR (Component VaR)")

        comp_var_rates = calc_component_var(returns_df, weights, confidence_level, holding_period)
        comp_var_amts  = comp_var_rates * total_value

        comp_df = pd.DataFrame({
            "종목": valid_tickers,
            f"기여 VaR ({currency})": [fmt_money(v, currency) for v in comp_var_amts],
            "기여 비율 (%)": (comp_var_amts / port_p_var * 100).round(2)
        }).set_index("종목")
        st.dataframe(comp_df, use_container_width=True)

        fig_comp = go.Figure(go.Bar(
            x=valid_tickers,
            y=comp_var_amts,
            marker_color=px.colors.qualitative.Set2[:len(valid_tickers)],
            text=[fmt_money(v, currency) for v in comp_var_amts],
            textposition="outside"
        ))
        fig_comp.update_layout(title="종목별 포트폴리오 VaR 기여액",
                                template="plotly_dark",
                                yaxis_title="기여 VaR",
                                showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("💡 기여 VaR이 높은 종목이 포트폴리오 리스크를 가장 많이 끌어올리는 종목입니다.")

        # ── 몬테카를로 손실 분포 ────────────────────
        st.markdown("---")
        st.subheader("🎲 몬테카를로 포트폴리오 손실 분포")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=port_mc_losses_amt, nbinsx=100,
                                       name="포트폴리오 손실 시나리오",
                                       marker_color="#64748b"))
        fig_mc.add_vline(x=port_mc_var_amt, line_width=3, line_dash="dash", line_color="#ef4444")
        fig_mc.add_annotation(x=port_mc_var_amt, y=10, text="MC VaR 임계점",
                               showarrow=True, arrowhead=1, ax=120, ay=-50,
                               font=dict(color="#ef4444"))
        fig_mc.update_layout(title="포트폴리오 미래 손실 시나리오 분포",
                              template="plotly_dark",
                              xaxis_title="손실액 (음수는 이익)",
                              yaxis_title="시나리오 횟수")
        st.plotly_chart(fig_mc, use_container_width=True)


        # ── 환율 변환기 ─────────────────────────────
        st.markdown("---")
        st.subheader("💱 실시간 환율 변환기 (USD ↔ KRW)")

        @st.cache_data(ttl=600)
        def get_exchange_rate():
            """yfinance로 실시간 USD/KRW 환율 조회"""
            try:
                fx = yf.Ticker("KRW=X")
                rate = fx.fast_info["last_price"]
                return float(rate)
            except Exception:
                return None

        # 환율 조회 버튼
        if "exchange_rate" not in st.session_state:
            st.session_state.exchange_rate = None
        if "exchange_rate_time" not in st.session_state:
            st.session_state.exchange_rate_time = None

        fetch_fx_btn = st.button("🔄 실시간 환율 불러오기", type="primary")

        if fetch_fx_btn:
            with st.spinner("환율 조회 중..."):
                rate = get_exchange_rate()
            if rate:
                st.session_state.exchange_rate = rate
                st.session_state.exchange_rate_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.error("❌ 환율 조회 실패. 잠시 후 다시 시도해 주세요.")

        if st.session_state.exchange_rate:
            rate = st.session_state.exchange_rate
            st.success(f"✅ 현재 환율: **1 USD = {rate:,.2f} KRW** (조회: {st.session_state.exchange_rate_time})")

            # 변환 방향 선택
            fx_col1, fx_col2 = st.columns(2)
            with fx_col1:
                convert_direction = st.radio("변환 방향", ["USD → KRW", "KRW → USD"], horizontal=True)
            with fx_col2:
                if convert_direction == "USD → KRW":
                    input_amount = st.number_input("금액 입력 (USD $)", min_value=0.0, value=1000.0, step=100.0, format="%.2f")
                else:
                    input_amount = st.number_input("금액 입력 (KRW ₩)", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.0f")

            convert_btn = st.button("💰 변환하기")
            if convert_btn:
                if convert_direction == "USD → KRW":
                    result_amt = input_amount * rate
                    st.success(f"### \${input_amount:,.2f}  →  ₩{result_amt:,.0f}")
                else:
                    result_amt = input_amount / rate
                    st.success(f"### ₩{input_amount:,.0f}  →  \${result_amt:,.2f}")

            # VaR 결과 환율 변환 테이블
            st.markdown("#### 📊 VaR 결과 통화 변환")
            var_results = {"모수적 VaR": port_p_var, "역사적 VaR": port_h_var, "몬테카를로 VaR": port_mc_var_amt}

            if currency == "$":
                conv_df = pd.DataFrame({
                    "방법론": list(var_results.keys()),
                    "원래 VaR (USD $)": [f"${v:,.2f}" for v in var_results.values()],
                    "변환 VaR (KRW ₩)": [f"₩{v * rate:,.0f}" for v in var_results.values()],
                })
            elif currency == "₩":
                conv_df = pd.DataFrame({
                    "방법론": list(var_results.keys()),
                    "원래 VaR (KRW ₩)": [f"₩{v:,.0f}" for v in var_results.values()],
                    "변환 VaR (USD $)": [f"${v / rate:,.2f}" for v in var_results.values()],
                })
            else:
                conv_df = None
                st.warning("혼합 통화 포트폴리오는 VaR 환율 변환을 지원하지 않습니다.")

            if conv_df is not None:
                st.dataframe(conv_df.set_index("방법론"), use_container_width=True)
                st.caption(f"💡 적용 환율: 1 USD = {rate:,.2f} KRW")
