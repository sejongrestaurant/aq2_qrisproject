import type { ReactNode } from "react";

import { PageHeader } from "../components/PageHeader";

const sections = [
  ["유니버스", "한국 주식 100개 후보군을 기반으로 하되, 과거 백테스트에서는 상장일과 point-in-time 사용 가능성을 우선 확인합니다."],
  ["팩터 계산법", "모멘텀, 상대강도, 퀄리티, 성장, 저변동성, 유동성 팩터를 표준화하고 종합 점수로 결합합니다."],
  ["상위 30개 선정", "종합 팩터 점수 순으로 순회하면서 섹터, 시장, 역할, 유동성 제약을 만족하는 종목을 편입합니다."],
  ["국면 분류", "KOSPI 200일선, 60일 모멘텀, 시장 폭 조건으로 Risk-On, Neutral, Risk-Off를 분류합니다."],
  ["비중 결정", "equal, score, rank 방식으로 초기 비중을 계산하고 종목/섹터/시장 상한과 현금 비중을 반영합니다."],
  ["리밸런싱", "월말 확정 신호를 사용하고 실제 거래는 다음 거래일 기본 next_open 기준으로 실행합니다."],
  ["거래비용", "매수와 매도 각각 편도 거래비용을 반영하며 기본 가정은 수수료와 시장충격 합산 0.20%입니다."],
  ["백테스트 한계", "현재 유니버스는 생존자 편향 위험이 있으며, 데이터 정합성·상장폐지·거래정지 이력 보강이 필요합니다."]
];

export function Methodology(): ReactNode {
  return (
    <div className="page">
      <PageHeader title="Methodology" description="MUST-30 Active Strategy의 연구 절차와 한계를 정리합니다." />
      <div className="method-list">
        {sections.map(([title, body]) => (
          <section className="panel" key={title}>
            <h2>{title}</h2>
            <p>{body}</p>
          </section>
        ))}
      </div>
    </div>
  );
}
