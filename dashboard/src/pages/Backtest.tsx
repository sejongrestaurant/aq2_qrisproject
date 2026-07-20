import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { api, displayStrategyName } from "../api/client";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState, ErrorState, LoadingState } from "../components/State";
import { PageHeader } from "../components/PageHeader";
import { formatNumber, formatPercent } from "../utils/format";

interface CompareRow {
  metric: string;
  strategy: string;
  value: string;
}

export function Backtest(): ReactNode {
  const strategies = useQuery({ queryKey: ["backtests"], queryFn: api.getBacktests });
  const [selected, setSelected] = useState<string[]>([]);
  const strategyName = selected[0] ?? strategies.data?.[0]?.strategy_name ?? "MUST30 score_weight";
  const daily = useQuery({ queryKey: ["daily", strategyName], queryFn: () => api.getBacktestDaily(strategyName) });
  const monthly = useQuery({ queryKey: ["monthly", strategyName], queryFn: () => api.getBacktestMonthly(strategyName) });
  const performance = useQuery({ queryKey: ["performance", strategyName], queryFn: () => api.getBacktestPerformance(strategyName) });

  const comparison = useMemo<CompareRow[]>(() => {
    const metrics = performance.data?.metrics ?? {};
    return [
      { metric: "CAGR", strategy: displayStrategyName, value: formatPercent(Number(metrics.cagr ?? 0), 1) },
      { metric: "Sharpe", strategy: displayStrategyName, value: formatNumber(Number(metrics.sharpe_ratio ?? 0), 2) },
      { metric: "MDD", strategy: displayStrategyName, value: formatPercent(Number(metrics.maximum_drawdown ?? 0), 1) },
      { metric: "Turnover", strategy: displayStrategyName, value: formatPercent(Number(metrics.annual_turnover ?? 0), 1) }
    ];
  }, [performance.data]);
  const yearly = useMemo(() => {
    const groups = new Map<string, number>();
    for (const item of monthly.data ?? []) {
      const year = item.month.slice(0, 4);
      groups.set(year, (1 + (groups.get(year) ?? 0)) * (1 + item.monthly_return) - 1);
    }
    return Array.from(groups.entries()).map(([year, annual_return]) => ({ year, annual_return }));
  }, [monthly.data]);
  const rollingSharpe = useMemo(() => {
    const rows = daily.data ?? [];
    return rows.map((item, index) => {
      const window = rows.slice(Math.max(0, index - 11), index + 1);
      const mean = window.reduce((sum, row) => sum + row.daily_return, 0) / window.length;
      const variance = window.reduce((sum, row) => sum + (row.daily_return - mean) ** 2, 0) / window.length;
      const sharpe = variance > 0 ? (mean / Math.sqrt(variance)) * Math.sqrt(252) : 0;
      return { date: item.date, rolling_sharpe: sharpe };
    });
  }, [daily.data]);

  if (strategies.isLoading || daily.isLoading || monthly.isLoading || performance.isLoading) return <LoadingState />;
  if (strategies.isError || daily.isError || monthly.isError || performance.isError) return <ErrorState title="백테스트 데이터를 불러오지 못했습니다" />;
  if (!daily.data?.length) return <EmptyState title="백테스트 데이터가 없습니다" />;

  const columns: Column<CompareRow>[] = [
    { key: "metric", label: "지표", render: (row) => row.metric },
    { key: "strategy", label: "전략", render: (row) => row.strategy },
    { key: "value", label: "값", render: (row) => row.value, align: "right" }
  ];

  return (
    <div className="page">
      <PageHeader title="Backtest" description="전략과 벤치마크의 수익률, 손실 구간, 월간 성과를 비교합니다." />
      <div className="check-list">
        {(strategies.data ?? []).map((item) => (
          <label key={item.strategy_name}>
            <input
              type="checkbox"
              checked={selected.includes(item.strategy_name)}
              onChange={(event) => setSelected(event.target.checked ? [item.strategy_name] : [])}
            />
            {item.strategy_name}
          </label>
        ))}
      </div>
      <section className="panel">
        <h2>누적 수익률</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={daily.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={24} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line dataKey="portfolio_value" stroke="#2563eb" dot={false} name="전략" />
            <Line dataKey="benchmark_value" stroke="#64748b" dot={false} name="벤치마크" />
          </LineChart>
        </ResponsiveContainer>
      </section>
      <div className="grid-2">
        <section className="panel">
          <h2>Drawdown</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={daily.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" hide />
              <YAxis tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <Tooltip formatter={(value) => formatPercent(Number(value), 1)} />
              <Line dataKey="drawdown" stroke="#dc2626" dot={false} name="Drawdown" />
            </LineChart>
          </ResponsiveContainer>
        </section>
        <section className="panel">
          <h2>월간 수익률</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={monthly.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <Tooltip formatter={(value) => formatPercent(Number(value), 1)} />
              <Bar dataKey="monthly_return" fill="#16a34a" name="월간 수익률" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>
      <div className="grid-2">
        <section className="panel">
          <h2>연도별 수익률</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={yearly}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis tickFormatter={(value) => formatPercent(Number(value), 0)} />
              <Tooltip formatter={(value) => formatPercent(Number(value), 1)} />
              <Bar dataKey="annual_return" fill="#0f766e" name="연도별 수익률" />
            </BarChart>
          </ResponsiveContainer>
        </section>
        <section className="panel">
          <h2>Rolling Sharpe</h2>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={rollingSharpe}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" hide />
              <YAxis />
              <Tooltip formatter={(value) => formatNumber(Number(value), 2)} />
              <Line dataKey="rolling_sharpe" stroke="#7c3aed" dot={false} name="Rolling Sharpe" />
            </LineChart>
          </ResponsiveContainer>
        </section>
      </div>
      <section className="panel">
        <h2>성과 비교 테이블</h2>
        <DataTable rows={comparison} columns={columns} />
      </section>
    </div>
  );
}
