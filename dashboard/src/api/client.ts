import {
  mockDaily,
  mockFactors,
  mockMonthly,
  mockPerformance,
  mockPortfolio,
  mockRegimeHistory,
  mockRegimeLatest,
  mockStrategies,
  mockUniverse
} from "./mockData";
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

const apiBaseUrl = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const useMocks = import.meta.env.VITE_USE_MOCKS === "true";

async function requestJson<T>(path: string, fallback: T): Promise<T> {
  if (useMocks) {
    return fallback;
  }
  const response = await fetch(`${apiBaseUrl}${path}`);
  if (!response.ok) {
    if (response.status === 404 && import.meta.env.VITE_ALLOW_EMPTY_FALLBACK === "true") {
      return fallback;
    }
    throw new Error(`API 요청 실패: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  getUniverse: (): Promise<UniverseItem[]> => requestJson("/api/universe?limit=1000", mockUniverse),
  getFactorsLatest: (): Promise<FactorScoreItem[]> => requestJson("/api/factors/latest?limit=1000", mockFactors),
  getRegimeLatest: (): Promise<RegimeItem> => requestJson("/api/regime/latest", mockRegimeLatest),
  getRegimeHistory: (): Promise<RegimeItem[]> =>
    requestJson("/api/regime/history?limit=1000", mockRegimeHistory),
  getPortfolioLatest: (): Promise<PortfolioWeightItem[]> =>
    requestJson("/api/portfolio/latest?limit=1000", mockPortfolio),
  getBacktests: (): Promise<BacktestStrategyItem[]> => requestJson("/api/backtests", mockStrategies),
  getBacktestDaily: (strategyName: string): Promise<BacktestDailyItem[]> =>
    requestJson(`/api/backtests/${encodeURIComponent(strategyName)}/daily?limit=1000`, mockDaily),
  getBacktestMonthly: (strategyName: string): Promise<BacktestMonthlyItem[]> =>
    requestJson(`/api/backtests/${encodeURIComponent(strategyName)}/monthly`, mockMonthly),
  getBacktestPerformance: (strategyName: string): Promise<PerformanceResponse> =>
    requestJson(`/api/backtests/${encodeURIComponent(strategyName)}/performance`, mockPerformance)
};

export const displayStrategyName = "MUST-30 Active Strategy";
