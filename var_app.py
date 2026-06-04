"""
통합 VaR 리스크 분석기 - 수정 완료 버전
수정 사항:
  1. 모수적 VaR: ddof=1 표본 표준편차 적용
  2. 역사적 VaR: 음수 VaR 방어 코드 추가
  3. 몬테카를로: 로그수익률 기반 GBM 파라미터 추정으로 수정
  4. √T 스케일링 일관성 관련 경고 문구 추가
  5. 종료일 기본값을 오늘 날짜로 수정
  6. 3가지 VaR 비교 바차트 시각화 추가
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# ─────────────────────────────────────────────
# 1. 웹 페이지 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="통합 VaR 리스크 분석기", layout="wide")
st.title("📊 3대 방법론 통합 VaR 분석 플랫폼")
st.markdown(
    "주식을 선택하고 조건값을 입력하면 **모수적 / 역사적 / 몬테카를로** 3가지 방식으로 "
    "위험가치(VaR)를 측정합니다."
)

# ─────────────────────────────────────────────
# 2. 사이드바: 사용자 입력
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ 분석 조건 설정")

ticker = st.sidebar.text_input(
    "1. 주식 티커 입력 (예: AAPL, TSLA, 005930.KS)", "AAPL"
)
start_date = st.sidebar.date_input("2. 데이터 시작일", pd.to_datetime("2022-01-01"))

# ✅ 수정 5: 종료일 기본값을 오늘 날짜로 변경 (미래 날짜 기본값 제거)
end_date = st.sidebar.date_input("3. 데이터 종료일", pd.Timestamp.today())

confidence_level = st.sidebar.selectbox(
    "4. 신뢰수준 (Confidence Level)", [0.95, 0.99], index=0
)
holding_period = st.sidebar.number_input(
    "5. 보유 기간 (일 단위)", min_value=1, max_value=30, value=1
)
investment = st.sidebar.number_input(
    "6. 총 투자 원금 (원/달러)",
    min_value=100_000,
    value=10_000_000,
    step=1_000_000,
    format="%d",
)

# ✅ 수정 4: √T 스케일링 가정 경고 문구
if holding_period > 1:
    st.sidebar.warning(
        f"⚠️ 보유기간 {holding_period}일 적용 시, 모수적·역사적 VaR은 "
        "√T 규칙(수익률 IID 가정)으로 스케일링됩니다. "
        "변동성 클러스터링이 있는 실제 시장에서는 과소 추정될 수 있습니다."
    )

# ─────────────────────────────────────────────
# 3. 데이터 로딩
# ─────────────────────────────────────────────
@st.cache_data
def load_data(ticker: str, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        return None
    # yfinance 버전별 멀티인덱스 방어 코드
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"][ticker]
    else:
        close = df["Close"]
    # 산술 수익률 & 로그 수익률 모두 반환
    arith_returns = close.pct_change().dropna()
    log_returns = np.log(close / close.shift(1)).dropna()
    return arith_returns, log_returns, close


# ─────────────────────────────────────────────
# 4. VaR 계산 함수
# ─────────────────────────────────────────────

def calc_parametric_var(
    returns: pd.Series, conf: float, days: int, inv: float
) -> float:
    """
    모수적 VaR (Parametric / Delta-Normal)
    - 정규분포 가정
    - ✅ 수정 1: ddof=1 표본 표준편차 사용
    - √T 스케일링 적용
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)          # ✅ ddof=1 표본 표준편차
    z = norm.ppf(conf)
    daily_var = z * sigma - mu               # 손실 방향 VaR (양수)
    return float(daily_var * np.sqrt(days) * inv)


def calc_historical_var(
    returns: pd.Series, conf: float, days: int, inv: float
) -> float:
    """
    역사적 시뮬레이션 VaR (Historical Simulation)
    - 과거 실제 수익률 분포에서 하위 퍼센타일 추출
    - ✅ 수정 2: max(..., 0) 음수 방어 코드 추가
    - √T 스케일링 적용
    """
    percentile = (1 - conf) * 100
    var_return = np.percentile(returns, percentile)
    daily_var = max(-var_return, 0)          # ✅ 음수 방어
    return float(daily_var * np.sqrt(days) * inv)


def calc_monte_carlo(
    log_returns: pd.Series,
    conf: float,
    days: int,
    inv: float,
    iterations: int = 10_000,
):
    """
    몬테카를로 VaR (Monte Carlo Simulation, GBM 기반)
    - ✅ 수정 3: 로그수익률로 mu·sigma 추정 후 GBM 시뮬레이션
      (산술 수익률로 로그 공식에 대입하던 오류 수정)
    - 다기간을 직접 시뮬레이션하므로 √T 스케일링 불필요
    """
    mu = np.mean(log_returns)                # 로그수익률 평균
    sigma = np.std(log_returns, ddof=1)      # 로그수익률 표본 표준편차

    np.random.seed(42)                       # 재현성 고정
    rand_normals = np.random.normal(0, 1, (days, iterations))

    # GBM: 로그수익률 합산 → 복리 총수익률
    log_path = (mu - 0.5 * sigma ** 2) + sigma * rand_normals  # shape: (days, iter)
    simulated_total_log_return = np.sum(log_path, axis=0)       # shape: (iter,)
    simulated_returns = np.exp(simulated_total_log_return) - 1  # 산술 환산

    simulated_losses = -simulated_returns * inv                 # 손실은 양수
    var_value = np.percentile(simulated_losses, conf * 100)
    return float(var_value), simulated_losses


# ─────────────────────────────────────────────
# 5. 메인 화면 구동
# ─────────────────────────────────────────────
result = load_data(ticker, start_date, end_date)

if result is not None:
    arith_returns, log_returns, prices = result

    # ── 상단 요약 지표 ──────────────────────────
    st.subheader(f"📈 {ticker} 자산 요약 및 가격 추이")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("분석 데이터 수", f"{len(arith_returns)} 일")
    col2.metric("평균 일일 수익률 (산술)", f"{arith_returns.mean()*100:.4f}%")
    col3.metric("평균 일일 수익률 (로그)", f"{log_returns.mean()*100:.4f}%")
    col4.metric("일일 변동성 (로그, ddof=1)", f"{log_returns.std(ddof=1)*100:.2f}%")

    # ── 그래프 1: 주가 추이 ─────────────────────
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name="종가",
            line=dict(color="#38bdf8", width=2),
        )
    )
    fig_price.update_layout(
        title=f"{ticker} 분석 기간 주가 추이",
        template="plotly_dark",
        xaxis_title="날짜",
        yaxis_title="가격",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── VaR 계산 ────────────────────────────────
    p_var = calc_parametric_var(arith_returns, confidence_level, holding_period, investment)
    h_var = calc_historical_var(arith_returns, confidence_level, holding_period, investment)
    m_var, mc_losses = calc_monte_carlo(log_returns, confidence_level, holding_period, investment)

    # ── 결과 대시보드 ───────────────────────────
    st.markdown("---")
    st.subheader("🚨 3대 방법론별 VaR 결과 비교")

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.error("**모수적 VaR (Parametric)**")
        st.markdown(f"### {p_var:,.0f} 원")
        st.caption("정규분포 + 표본 표준편차(ddof=1) 기반")
    with res_col2:
        st.error("**역사적 VaR (Historical)**")
        st.markdown(f"### {h_var:,.0f} 원")
        st.caption("과거 실제 수익률 분포 기반")
    with res_col3:
        st.error("**몬테카를로 VaR (Monte Carlo)**")
        st.markdown(f"### {m_var:,.0f} 원")
        st.caption("로그수익률 GBM 10,000회 시뮬레이션")

    st.info(
        f"💡 **해석:** {confidence_level*100:.0f}%의 확률로, "
        f"다음 {holding_period}일 동안 발생할 최대 손실액은 위 금액들을 "
        "넘지 않을 것으로 예상됩니다."
    )

    # ── 그래프 2: 3대 VaR 비교 바차트 ────────────
    st.markdown("---")
    st.subheader("📊 방법론별 VaR 비교 차트")

    methods = ["모수적 (Parametric)", "역사적 (Historical)", "몬테카를로 (Monte Carlo)"]
    values = [p_var, h_var, m_var]
    colors = ["#38bdf8", "#34d399", "#f97316"]

    fig_bar = go.Figure(
        go.Bar(
            x=methods,
            y=values,
            marker_color=colors,
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        title="3대 방법론 VaR 직접 비교",
        template="plotly_dark",
        yaxis_title="VaR (손실 추정액)",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 그래프 3: 일일 수익률 분포 + 정규분포 오버레이 ──
    st.markdown("---")
    st.subheader("📉 과거 수익률 분포 (모수적 VaR 가정 검증)")

    mu_a = arith_returns.mean()
    sig_a = arith_returns.std(ddof=1)
    x_range = np.linspace(arith_returns.min(), arith_returns.max(), 300)
    normal_curve = norm.pdf(x_range, mu_a, sig_a)

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=arith_returns,
            nbinsx=80,
            histnorm="probability density",
            name="실제 수익률 분포",
            marker_color="#64748b",
            opacity=0.7,
        )
    )
    fig_dist.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_curve,
            mode="lines",
            name="정규분포 근사",
            line=dict(color="#facc15", width=2, dash="dash"),
        )
    )
    # 모수적 VaR 임계선
    var_daily_return = -(p_var / investment / np.sqrt(holding_period))
    fig_dist.add_vline(
        x=var_daily_return,
        line_width=2,
        line_dash="dash",
        line_color="#ef4444",
    )
    fig_dist.add_annotation(
        x=var_daily_return,
        y=0,
        text="모수적 VaR 임계",
        showarrow=True,
        arrowhead=1,
        ax=80,
        ay=-60,
        font=dict(color="#ef4444"),
    )
    fig_dist.update_layout(
        title="실제 수익률 분포 vs 정규분포 가정 (두 분포가 크게 다르면 모수적 VaR 신뢰도 낮음)",
        template="plotly_dark",
        xaxis_title="일일 수익률",
        yaxis_title="확률 밀도",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── 그래프 4: 몬테카를로 손실 분포 히스토그램 ──
    st.markdown("---")
    st.subheader("🎲 몬테카를로 시뮬레이션 손실 확률 분포")

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=mc_losses,
            nbinsx=100,
            name="시뮬레이션 손실액",
            marker_color="#64748b",
        )
    )
    fig_hist.add_vline(
        x=m_var, line_width=3, line_dash="dash", line_color="#ef4444"
    )
    fig_hist.add_annotation(
        x=m_var,
        y=10,
        text=f"Monte Carlo VaR 임계점",
        showarrow=True,
        arrowhead=1,
        ax=120,
        ay=-50,
        font=dict(color="#ef4444"),
    )
    fig_hist.update_layout(
        title="미래 손실 시나리오 분포 (빨간 점선 오른쪽 = 임계 손실 영역)",
        template="plotly_dark",
        xaxis_title="손실액 (음수는 이익)",
        yaxis_title="시나리오 횟수",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── 방법론 설명 ─────────────────────────────
    st.markdown("---")
    st.subheader("📚 방법론 요약 및 주의사항")
    with st.expander("각 방법론의 가정과 한계 보기"):
        st.markdown(
            """
| 방법론 | 핵심 가정 | 장점 | 한계 |
|--------|-----------|------|------|
| **모수적 (Parametric)** | 수익률이 정규분포를 따름 | 계산 단순, 해석 직관적 | 두꺼운 꼬리(팻테일) 과소 반영 |
| **역사적 (Historical)** | 과거가 미래를 반영 | 분포 가정 없음 | 과거 데이터에 종속, 데이터 부족 시 불안정 |
| **몬테카를로** | GBM + 로그 정규분포 | 복잡한 경로 시뮬레이션 가능 | 파라미터 추정 오류 민감, 계산 비용 높음 |

> ⚠️ **√T 스케일링 주의**: 모수적·역사적 VaR의 다기간 확장은 수익률 IID(독립동일분포) 가정 하에서만 유효합니다.  
> 실제 시장에는 변동성 클러스터링이 존재하므로 보유 기간이 길수록 과소 추정 가능성이 있습니다.
            """
        )

else:
    st.error(
        "❌ 데이터를 불러오지 못했습니다. "
        "티커 코드 또는 날짜 범위를 확인하고, 인터넷 연결 상태를 점검해 주세요."
    )
