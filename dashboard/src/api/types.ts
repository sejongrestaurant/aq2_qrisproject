export type RegimeName = "Risk-On" | "Neutral" | "Risk-Off";

export interface UniverseItem {
  ticker: string;
  company_name: string;
  market: string;
  sector: string;
  industry: string;
  investment_theme: string;
  universe_role: string;
  listing_date: string;
  is_active: boolean;
}

export interface FactorScoreItem {
  calculation_date: string;
  ticker: string;
  composite_score: number;
  universe_rank: number;
  momentum_score: number | null;
  relative_strength_score: number | null;
  quality_score: number | null;
  growth_score: number | null;
  low_volatility_score: number | null;
  liquidity_score: number | null;
}

export interface RegimeItem {
  date: string;
  regime: RegimeName;
  kospi_close: number;
  moving_average: number;
  volatility: number;
  market_breadth: number;
  score: number;
}

export interface PortfolioWeightItem {
  rebalance_date: string;
  ticker: string;
  target_weight: number;
  rank: number;
  regime: RegimeName;
  selection_reason: string;
}

export interface BacktestStrategyItem {
  strategy_name: string;
  start_date: string;
  end_date: string;
  observation_count: number;
}

export interface BacktestDailyItem {
  date: string;
  strategy_name: string;
  daily_return: number;
  portfolio_value: number;
  benchmark_return: number | null;
  benchmark_value: number | null;
  drawdown: number;
  turnover: number;
  transaction_cost: number;
  cash_weight: number;
}

export interface BacktestMonthlyItem {
  month: string;
  month_end_date: string;
  monthly_return: number;
  portfolio_value: number;
}

export interface PerformanceResponse {
  strategy_name: string;
  metrics: Record<string, number | string | null>;
}

export interface PortfolioRow extends PortfolioWeightItem {
  company_name?: string;
  sector?: string;
  market?: string;
  composite_score?: number;
  momentum_score?: number | null;
  relative_strength_score?: number | null;
  quality_score?: number | null;
  growth_score?: number | null;
  low_volatility_score?: number | null;
  liquidity_score?: number | null;
}
