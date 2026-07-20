import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api, displayStrategyName } from "../api/client";
import { EmptyState, ErrorState, LoadingState } from "../components/State";
import { MetricCard } from "../components/MetricCard";
import { PageHeader } from "../components/PageHeader";
import { formatNumber, formatPercent } from "../utils/format";

export function Overview(): ReactNode {
  const backtests = useQuery({ queryKey: ["backtests"], queryFn: api.getBacktests });
  const strategyName = backtests.data?.[0]?.strategy_name ?? "MUST30 score_weight";
  const performance = useQuery({
    queryKey: ["performance", strategyName],
    queryFn: () => api.getBacktestPerformance(strategyName)
  });
  const regime = useQuery({ queryKey: ["regime-latest"], queryFn: api.getRegimeLatest });
  const daily = useQuery({ queryKey: ["daily", strategyName], queryFn: () => api.getBacktestDaily(strategyName) });

  if (backtests.isLoading || performance.isLoading || regime.isLoading || daily.isLoading) {
    return <LoadingState />;
  }
  if (backtests.isError || performance.isError || regime.isError || daily.isError) {
    return <ErrorState title="Overview 데이터를 불러오지 못했습니다" description="API URL 또는 mock 모드 설정을 확인하세요." />;
  }
  if (!performance.data || !regime.data) {
    return <EmptyState title="표시할 성과 데이터가 없습니다" />;
  }

  const metrics = performance.data.metrics;
  const latestReturn = daily.data?.[daily.data.length - 1]?.daily_return ?? null;
  const equityWeight = regime.data.regime === "Risk-On" ? 1 : regime.data.regime === "Neutral" ? 0.8 : 0.5;

  return (
    <div className="page">
      <PageHeader
        title={displayStrategyName}
        description="팩터 기반 상위 종목 선정, 시장 국면별 주식/현금 배분, 월별 리밸런싱을 결합한 연구용 대시보드입니다."
      />
      <div className="metric-grid">
        <MetricCard label="최신 일간 수익률" value={formatPercent(Number(latestReturn ?? 0), 2)} />
        <MetricCard label="CAGR" value={formatPercent(Number(metrics.cagr ?? metrics.annualized_return ?? 0), 1)} />
        <MetricCard label="Sharpe" value={formatNumber(Number(metrics.sharpe_ratio ?? 0), 2)} />
        <MetricCard label="MDD" value={formatPercent(Number(metrics.maximum_drawdown ?? metrics.max_drawdown ?? 0), 1)} />
        <MetricCard label="현재 시장 국면" value={regime.data.regime} detail={regime.data.date} />
        <MetricCard label="주식 / 현금" value={`${formatPercent(equityWeight, 0)} / ${formatPercent(1 - equityWeight, 0)}`} />
      </div>
      <section className="panel">
        <h2>포트폴리오 가치 흐름</h2>
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={daily.data ?? []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={28} />
            <YAxis />
            <Tooltip formatter={(value) => formatNumber(Number(value), 2)} />
            <Area type="monotone" dataKey="portfolio_value" stroke="#2563eb" fill="#dbeafe" name="전략 가치" />
          </AreaChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}
