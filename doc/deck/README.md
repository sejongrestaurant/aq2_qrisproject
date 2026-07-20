# 발표자료 재현 절차

`헬름_IRP세븐서티액티브_발표자료.pptx` 와 `투자제안서_v1.3.md` 는 **같은 한 벌의 숫자**
(`metrics.json`)를 쓴다. 아래 순서로 전부 재생성된다.

## 1. 동결 브랜치 워크트리 준비

지표·백테스트 엔진(`run_v2.py`, `analysis/`, `satellite/backtester_v2.py`)은
`origin/v2-tier2a-freeze` 에만 있다. main 에는 없으므로 워크트리를 만든다.

```bash
git worktree add /path/to/wt-freeze origin/v2-tier2a-freeze
cp datasets/ohlcv/*.parquet /path/to/wt-freeze/datasets/ohlcv/   # 시세는 gitignore 대상
```

## 2. 수치·차트 생성 (워크트리 안에서 실행)

```bash
cd /path/to/wt-freeze
python run_exposure.py        # reports/exposure_monthly.csv 선행 생성
python gen_metrics.py         # _deck/metrics.json + _deck/*.png
```

구간은 `config.json` 의 `end`(2026-06-30)를 따른다. **`end=None` 을 주면 데이터 끝
(현재 2026-07-15)까지 돌아 제안서와 다른 수치가 나오므로 주의한다.**

## 3. PPTX 조립

```bash
python build_deck.py          # doc/헬름_IRP세븐서티액티브_발표자료.pptx
```

`ASSETS` 경로가 1단계 워크트리의 `_deck/` 을 가리키도록 맞춘 뒤 실행한다.

## 검증된 재현 결과 (2020-01-01 ~ 2026-06-30 · 왕복 비용 0.10%)

| | 본 상품 | KODEX TRF7030 |
|---|---|---|
| CAGR | 13.6% | 13.2% |
| MDD | −12.7% | −22.1% |
| Sharpe | 1.20 | 1.18 |
| Calmar | 1.07 | 0.60 |
| 2026 제외 Calmar | 1.09 | 0.56 |

`metrics.json` 에는 원설계(이진 게이트 60) 대조군 수치도 함께 들어 있으나, 이는
**내부 검증용**이며 제안서·발표자료에는 노출하지 않는다 — 최종 상품은 하나이고,
비교 대상은 사내 프로토타입이 아니라 시장 대안(KODEX TRF7030)이다.

## 알려진 이력

- 제안서 **v1.2 까지의 수치는 시세가 2026-06-15 까지만 적재된 상태**에서 산출된 것이다.
  `python run_v2.py --end 2026-06-15` 로 정확히 재현된다(V1 +44.7% / V2 +27.5%).
  v1.3 부터는 6월 전체가 적재된 현재 데이터 기준이다(V1 +46.9% / V2 +29.3%).
- 2020~2025 연도별 수익과 2025년말 컷 지표는 **두 시점에서 완전히 동일**하다.
