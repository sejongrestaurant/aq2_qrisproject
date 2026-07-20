import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState } from "../components/State";
import { MetricCard } from "../components/MetricCard";
import { PageHeader } from "../components/PageHeader";
import { formatNumber, formatPercent } from "../utils/format";

export function MarketRegime(): ReactNode {
  const latest = useQuery({ queryKey: ["regime-latest"], queryFn: api.getRegimeLatest });
  const history = useQuery({ queryKey: ["regime-history"], queryFn: api.getRegimeHistory });

  if (latest.isLoading || history.isLoading) return <LoadingState />;
  if (latest.isError || history.isError) return <ErrorState title="시장 국면 데이터를 불러오지 못했습니다" />;
  if (!latest.data || !history.data?.length) return <EmptyState title="시장 국면 데이터가 없습니다" />;

  const equityWeight = latest.data.regime === "Risk-On" ? 1 : latest.data.regime === "Neutral" ? 0.8 : 0.5;
  const intervals = history.data.map((item) => ({ date: item.date, regime: item.regime }));

  return (
    <div className="page">
      <PageHeader title="Market Regime" description="월말 확정 지표 기반 국면과 위험 노출을 확인합니다." />
      <div className="metric-grid">
        <MetricCard label="현재 국면" value={latest.data.regime} detail={latest.data.date} />
        <MetricCard label="KOSPI" value={formatNumber(latest.data.kospi_close, 1)} />
        <MetricCard label="200일 이동평균" value={formatNumber(latest.data.moving_average, 1)} />
        <MetricCard label="시장 폭" value={formatPercent(latest.data.market_breadth, 1)} />
        <MetricCard label="변동성" value={formatPercent(latest.data.volatility, 1)} />
        <MetricCard label="주식 / 현금" value={`${formatPercent(equityWeight, 0)} / ${formatPercent(1 - equityWeight, 0)}`} />
      </div>
      <section className="panel">
        <h2>KOSPI와 200일 이동평균</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={history.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={28} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line dataKey="kospi_close" stroke="#2563eb" dot={false} name="KOSPI" />
            <Line dataKey="moving_average" stroke="#f97316" dot={false} name="200일 이동평균" />
          </LineChart>
        </ResponsiveContainer>
      </section>
      <section className="panel">
        <h2>과거 국면 구간</h2>
        <div className="regime-strip">
          {intervals.map((item) => (
            <span key={item.date} className={`regime-pill ${item.regime.toLowerCase().replace("-", "")}`}>
              {item.date} · {item.regime}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}
