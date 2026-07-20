import type {
  BacktestDailyItem,
  BacktestMonthlyItem,
  BacktestStrategyItem,
  FactorScoreItem,
  PerformanceResponse,
  PortfolioWeightItem,
  RegimeItem,
  UniverseItem
} from "./types";

export const mockUniverse: UniverseItem[] = [
  {
    ticker: "005930",
    company_name: "삼성전자",
    market: "KOSPI",
    sector: "반도체",
    industry: "종합 반도체",
    investment_theme: "AI, 메모리",
    universe_role: "Core",
    listing_date: "2010-01-01",
    is_active: true
  },
  {
    ticker: "000660",
    company_name: "SK하이닉스",
    market: "KOSPI",
    sector: "반도체",
    industry: "메모리 반도체",
    investment_theme: "HBM, AI",
    universe_role: "Core",
    listing_date: "2010-01-01",
    is_active: true
  },
  {
    ticker: "035420",
    company_name: "NAVER",
    market: "KOSPI",
    sector: "인터넷",
    industry: "플랫폼",
    investment_theme: "AI, 커머스",
    universe_role: "Core",
    listing_date: "2010-01-01",
    is_active: true
  }
];

export const mockFactors: FactorScoreItem[] = mockUniverse.map((item, index) => ({
  calculation_date: "2026-06-30",
  ticker: item.ticker,
  composite_score: 1.2 - index * 0.28,
  universe_rank: index + 1,
  momentum_score: 0.8 - index * 0.1,
  relative_strength_score: 0.7 - index * 0.05,
  quality_score: 1.1 - index * 0.2,
  growth_score: 0.6 - index * 0.08,
  low_volatility_score: 0.2 + index * 0.1,
  liquidity_score: 1.0 - index * 0.12
}));

export const mockRegimeLatest: RegimeItem = {
  date: "2026-06-30",
  regime: "Risk-On",
  kospi_close: 3180,
  moving_average: 3010,
  volatility: 0.18,
  market_breadth: 0.61,
  score: 3
};

export const mockRegimeHistory: RegimeItem[] = Array.from({ length: 18 }, (_, index) => {
  const date = new Date(2025, index, 28);
  const regime = index % 6 < 3 ? "Risk-On" : index % 6 < 5 ? "Neutral" : "Risk-Off";
  return {
    date: date.toISOString().slice(0, 10),
    regime,
    kospi_close: 2800 + index * 24 + (regime === "Risk-Off" ? -80 : 0),
    moving_average: 2760 + index * 18,
    volatility: 0.14 + (regime === "Risk-Off" ? 0.1 : 0.02),
    market_breadth: regime === "Risk-On" ? 0.62 : regime === "Neutral" ? 0.5 : 0.38,
    score: regime === "Risk-On" ? 3 : regime === "Neutral" ? 0 : -3
  };
});

export const mockPortfolio: PortfolioWeightItem[] = mockUniverse.map((item, index) => ({
  rebalance_date: "2026-07-01",
  ticker: item.ticker,
  target_weight: [0.05, 0.047, 0.041][index],
  rank: index + 1,
  regime: "Risk-On",
  selection_reason: `${item.company_name}은 종합 팩터 점수와 유동성 기준을 충족했습니다.`
}));

export const mockStrategies: BacktestStrategyItem[] = [
  {
    strategy_name: "MUST30 score_weight",
    start_date: "2014-01-01",
    end_date: "2026-06-30",
    observation_count: 3100
  }
];

export const mockDaily: BacktestDailyItem[] = Array.from({ length: 36 }, (_, index) => {
  const date = new Date(2024, 0, index + 1);
  const value = 100 + index * 0.8 + Math.sin(index / 2) * 2;
  return {
    date: date.toISOString().slice(0, 10),
    strategy_name: "MUST30 score_weight",
    daily_return: index === 0 ? 0 : 0.004 * Math.sin(index / 3),
    portfolio_value: value,
    benchmark_return: 0.002 * Math.sin(index / 4),
    benchmark_value: 100 + index * 0.45,
    drawdown: Math.min(0, Math.sin(index / 5) * -0.05),
    turnover: index % 21 === 0 ? 0.35 : 0,
    transaction_cost: index % 21 === 0 ? 0.0007 : 0,
    cash_weight: 0.2
  };
});

export const mockMonthly: BacktestMonthlyItem[] = Array.from({ length: 12 }, (_, index) => ({
  month: `2025-${String(index + 1).padStart(2, "0")}`,
  month_end_date: `2025-${String(index + 1).padStart(2, "0")}-28`,
  monthly_return: 0.015 * Math.sin(index / 2) + 0.006,
  portfolio_value: 100 + index * 2.1
}));

export const mockPerformance: PerformanceResponse = {
  strategy_name: "MUST30 score_weight",
  metrics: {
    total_return: 0.86,
    cagr: 0.108,
    sharpe_ratio: 1.12,
    maximum_drawdown: -0.183,
    calmar_ratio: 0.59,
    annualized_volatility: 0.17,
    annual_turnover: 3.1,
    total_transaction_cost: 0.022
  }
};
