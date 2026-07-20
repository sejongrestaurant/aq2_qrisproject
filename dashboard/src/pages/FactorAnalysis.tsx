import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../api/client";
import type { FactorScoreItem } from "../api/types";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState, ErrorState, LoadingState } from "../components/State";
import { PageHeader } from "../components/PageHeader";
import { formatNumber } from "../utils/format";

export function FactorAnalysis(): ReactNode {
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const factors = useQuery({ queryKey: ["factors"], queryFn: api.getFactorsLatest });
  const universe = useQuery({ queryKey: ["universe"], queryFn: api.getUniverse });

  const rows = useMemo(() => {
    const universeMap = new Map(universe.data?.map((item) => [item.ticker, item]));
    return (factors.data ?? []).map((item) => ({ ...item, ...universeMap.get(item.ticker) }));
  }, [factors.data, universe.data]);

  if (factors.isLoading || universe.isLoading) return <LoadingState />;
  if (factors.isError || universe.isError) return <ErrorState title="팩터 데이터를 불러오지 못했습니다" />;
  if (!rows.length) return <EmptyState title="팩터 데이터가 없습니다" />;

  const sectorAverage = Array.from(new Set(rows.map((row) => row.sector))).map((sector) => {
    const group = rows.filter((row) => row.sector === sector);
    return {
      sector,
      composite_score: group.reduce((sum, row) => sum + row.composite_score, 0) / group.length
    };
  });
  const factorContribution = [
    { factor: "Momentum", score: avg(rows, "momentum_score") },
    { factor: "Quality", score: avg(rows, "quality_score") },
    { factor: "Growth", score: avg(rows, "growth_score") },
    { factor: "Low Vol", score: avg(rows, "low_volatility_score") },
    { factor: "Liquidity", score: avg(rows, "liquidity_score") }
  ];
  const correlationRows = factorContribution.map((left) => ({
    factor: left.factor,
    Momentum: formatNumber(correlation(rows, left.factor, "Momentum"), 2),
    Quality: formatNumber(correlation(rows, left.factor, "Quality"), 2),
    LowVol: formatNumber(correlation(rows, left.factor, "Low Vol"), 2)
  }));
  const selected = rows.find((row) => row.ticker === selectedTicker);
  const columns: Column<typeof rows[number]>[] = [
    { key: "rank", label: "순위", render: (row) => row.universe_rank, align: "right" },
    { key: "name", label: "종목", render: (row) => row.company_name ?? row.ticker },
    { key: "ticker", label: "티커", render: (row) => row.ticker },
    { key: "sector", label: "섹터", render: (row) => row.sector ?? "-" },
    { key: "composite", label: "종합", render: (row) => formatNumber(row.composite_score, 2), align: "right" },
    {
      key: "detail",
      label: "상세",
      render: (row) => <button className="link-button" onClick={() => setSelectedTicker(row.ticker)}>보기</button>
    }
  ];

  return (
    <div className="page">
      <PageHeader title="Factor Analysis" description="팩터 점수, 섹터 평균, 종목 상세를 확인합니다." />
      <div className="grid-2">
        <section className="panel">
          <h2>팩터별 평균 점수</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={factorContribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="factor" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="score" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </section>
        <section className="panel">
          <h2>섹터별 평균 종합 점수</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={sectorAverage}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="sector" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="composite_score" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>
      <section className="panel">
        <h2>종목별 팩터 점수</h2>
        <DataTable rows={rows} columns={columns} />
      </section>
      <section className="panel">
        <h2>팩터 상관관계</h2>
        <DataTable
          rows={correlationRows}
          columns={[
            { key: "factor", label: "팩터", render: (row) => row.factor },
            { key: "momentum", label: "Momentum", render: (row) => row.Momentum, align: "right" },
            { key: "quality", label: "Quality", render: (row) => row.Quality, align: "right" },
            { key: "lowvol", label: "Low Vol", render: (row) => row.LowVol, align: "right" }
          ]}
        />
      </section>
      {selected ? (
        <div className="modal-backdrop" onClick={() => setSelectedTicker(null)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <h2>{selected.company_name ?? selected.ticker}</h2>
            <p>{selected.ticker} · {selected.sector ?? "-"} · {selected.market ?? "-"}</p>
            <dl className="detail-grid">
              <dt>종합</dt><dd>{formatNumber(selected.composite_score, 2)}</dd>
              <dt>모멘텀</dt><dd>{formatNumber(selected.momentum_score, 2)}</dd>
              <dt>상대강도</dt><dd>{formatNumber(selected.relative_strength_score, 2)}</dd>
              <dt>퀄리티</dt><dd>{formatNumber(selected.quality_score, 2)}</dd>
              <dt>성장</dt><dd>{formatNumber(selected.growth_score, 2)}</dd>
              <dt>저변동성</dt><dd>{formatNumber(selected.low_volatility_score, 2)}</dd>
              <dt>유동성</dt><dd>{formatNumber(selected.liquidity_score, 2)}</dd>
            </dl>
            <button onClick={() => setSelectedTicker(null)}>닫기</button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function avg(rows: FactorScoreItem[], key: keyof FactorScoreItem): number {
  const values = rows.map((row) => row[key]).filter((value): value is number => typeof value === "number");
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function factorKey(label: string): keyof FactorScoreItem {
  const map: Record<string, keyof FactorScoreItem> = {
    Momentum: "momentum_score",
    Quality: "quality_score",
    Growth: "growth_score",
    "Low Vol": "low_volatility_score",
    Liquidity: "liquidity_score"
  };
  return map[label] ?? "composite_score";
}

function correlation(rows: FactorScoreItem[], left: string, right: string): number {
  const leftKey = factorKey(left);
  const rightKey = factorKey(right);
  const pairs = rows
    .map((row) => [row[leftKey], row[rightKey]])
    .filter((pair): pair is [number, number] => typeof pair[0] === "number" && typeof pair[1] === "number");
  if (pairs.length < 2) return 0;
  const leftMean = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const rightMean = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  const numerator = pairs.reduce((sum, pair) => sum + (pair[0] - leftMean) * (pair[1] - rightMean), 0);
  const leftStd = Math.sqrt(pairs.reduce((sum, pair) => sum + (pair[0] - leftMean) ** 2, 0));
  const rightStd = Math.sqrt(pairs.reduce((sum, pair) => sum + (pair[1] - rightMean) ** 2, 0));
  return leftStd > 0 && rightStd > 0 ? numerator / (leftStd * rightStd) : 0;
}
