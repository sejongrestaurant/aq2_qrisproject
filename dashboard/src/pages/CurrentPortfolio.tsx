import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";

import type { PortfolioRow } from "../api/types";
import { api } from "../api/client";
import { DataTable, type Column } from "../components/DataTable";
import { EmptyState, ErrorState, LoadingState } from "../components/State";
import { PageHeader } from "../components/PageHeader";
import { formatNumber, formatWeight } from "../utils/format";

export function CurrentPortfolio(): ReactNode {
  const [query, setQuery] = useState("");
  const [sector, setSector] = useState("전체");
  const [sortKey, setSortKey] = useState<"rank" | "target_weight" | "composite_score">("rank");
  const portfolio = useQuery({ queryKey: ["portfolio"], queryFn: api.getPortfolioLatest });
  const universe = useQuery({ queryKey: ["universe"], queryFn: api.getUniverse });
  const factors = useQuery({ queryKey: ["factors"], queryFn: api.getFactorsLatest });

  const rows = useMemo<PortfolioRow[]>(() => {
    const universeMap = new Map(universe.data?.map((item) => [item.ticker, item]));
    const factorMap = new Map(factors.data?.map((item) => [item.ticker, item]));
    return (portfolio.data ?? [])
      .map((item) => ({ ...item, ...universeMap.get(item.ticker), ...factorMap.get(item.ticker) }))
      .filter((item) => {
        const text = `${item.company_name ?? ""} ${item.ticker}`.toLowerCase();
        return text.includes(query.toLowerCase()) && (sector === "전체" || item.sector === sector);
      })
      .sort((a, b) => (
        sortKey === "rank"
          ? Number(a.rank) - Number(b.rank)
          : Number(b[sortKey] ?? 0) - Number(a[sortKey] ?? 0)
      ));
  }, [factors.data, portfolio.data, query, sector, sortKey, universe.data]);

  if (portfolio.isLoading || universe.isLoading || factors.isLoading) return <LoadingState />;
  if (portfolio.isError || universe.isError || factors.isError) return <ErrorState title="포트폴리오 데이터를 불러오지 못했습니다" />;

  const sectors = ["전체", ...Array.from(new Set(rows.map((row) => row.sector).filter(Boolean)))];
  const columns: Column<PortfolioRow>[] = [
    { key: "rank", label: "순위", render: (row) => row.rank, align: "right" },
    { key: "company", label: "종목명", render: (row) => row.company_name ?? "-" },
    { key: "ticker", label: "티커", render: (row) => row.ticker },
    { key: "weight", label: "비중", render: (row) => formatWeight(row.target_weight), align: "right" },
    { key: "sector", label: "섹터", render: (row) => row.sector ?? "-" },
    { key: "market", label: "시장", render: (row) => row.market ?? "-" },
    { key: "score", label: "종합", render: (row) => formatNumber(row.composite_score, 2), align: "right" },
    { key: "momentum", label: "모멘텀", render: (row) => formatNumber(row.momentum_score, 2), align: "right" },
    { key: "quality", label: "퀄리티", render: (row) => formatNumber(row.quality_score, 2), align: "right" },
    { key: "reason", label: "선정 이유", render: (row) => row.selection_reason }
  ];

  return (
    <div className="page">
      <PageHeader title="Current Portfolio" description="최신 목표 비중과 종목별 팩터 점수를 확인합니다." />
      <div className="toolbar">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="종목명 또는 티커 검색" />
        <select value={sector} onChange={(event) => setSector(event.target.value)}>
          {sectors.map((item) => <option key={item}>{item}</option>)}
        </select>
        <select value={sortKey} onChange={(event) => setSortKey(event.target.value as typeof sortKey)}>
          <option value="rank">순위</option>
          <option value="target_weight">비중</option>
          <option value="composite_score">종합 점수</option>
        </select>
      </div>
      {rows.length ? <DataTable rows={rows} columns={columns} /> : <EmptyState title="조건에 맞는 종목이 없습니다" />}
    </div>
  );
}
