# EST Indicator — 일봉 TrendScore Swing Backtest

일봉 **TrendScore**(0~100)를 신호로 쓰는 롱-플랫 **스윙 전략**을 백테스트하고, 결과를
자체완결형 **HTML 리포트**로 생성하는 프로젝트다. 참고 자동매매 시스템(`temp/AutoTrade`)의
TrendScore·스윙 로직을 **OOP 모듈 구조**로 재구성했으며, 지표·전략·엔진·리포터를 인터페이스
단위로 분리해 개별 교체·확장이 쉽다.

## 모듈 구조

```
Indicator/
├── data/                  # 시세 로딩 (표준 스키마로 정규화)
│   └── loader.py          #   DataLoader(ABC) → ParquetDataLoader, PriceData
├── indicator/             # 기술적 지표 (표준 스키마 → pd.Series)
│   ├── base.py            #   Indicator(ABC)
│   ├── rsi.py             #   RSIIndicator
│   ├── adx.py             #   ADXIndicator (부호 있는 방향성 ADX)
│   └── trend_score.py     #   TrendScoreIndicator (EWMAC+TSMOM+RSI+ADX penalty, 0~100)
├── indicator/
│   └── supertrend.py      #   SuperTrendIndicator (ATR 밴드 추세추종, 가격 오버레이형)
├── strategy/              # 매매 전략 (시세 → 봉별 목표 보유상태)
│   ├── base.py            #   Strategy(ABC), Signals(indicators/overlays)
│   ├── swing_trend_score.py  #  TrendScoreSwingStrategy (히스테리시스 + ADX 게이트)
│   └── swing_supertrend.py   #  SuperTrendSwingStrategy (밴드 방향 추종)
├── backtest/              # 백테스트 (신호 → 자산곡선·성과)
│   ├── trade.py           #   Trade (왕복 거래 기록)
│   ├── result.py          #   BacktestResult (+ 성과지표 계산)
│   └── engine.py          #   Backtester (익일 시가 체결, 룩어헤드 방지)
├── report/                # 리포트 (결과 → 산출물)
│   ├── base.py            #   Reporter(ABC)
│   └── html_report.py     #   HTMLReporter (matplotlib 차트 base64 임베드)
│   └── yfinance_loader.py #   YFinanceDataLoader (온라인 다운로드 + 워밍업 확장 + 캐시)
├── datasets/ohlcv/        # OHLCV parquet (미국 섹터/테마 ETF 14년)
├── config.json            # ★ 편집 대상: 유니버스·임계·ADX·비용·구간·소스
├── config.py              # config.json 로더 (누락 키는 기본값 폴백)
├── main.py                # 파이프라인 오케스트레이터 (Pipeline)
└── reports/               # 생성된 HTML 리포트 출력 위치
```

### 데이터 흐름

```
DataLoader → TrendScoreSwingStrategy(TrendScoreIndicator) → Backtester → HTMLReporter
  PriceData          Signals(target_long)                   BacktestResult      HTML
```

## 실행

```bash
pip install -r requirements.txt
python main.py
# → reports/index.html (유니버스 비교표 + 링크)
#   reports/<코드>.html (종목별 개별 리포트: 전체기간·연도별 성과 + 매수/매도 마커 차트)
```

리포트는 **종목별로 개별 HTML** 이 생성되고, `index.html` 에서 유니버스 비교표(종목 클릭 → 상세)로
연결된다. 종목 페이지 상단에는 **전략 비교**(TrendScore vs SuperTrend vs Buy&Hold 지표표 + 자산곡선
오버레이)가 있고, 그 아래 각 전략 상세(가격+매수▲/매도▼ 마커, 지표, 연도별 성과)가 이어진다.
`config.json` 의 `strategies` 로 비교에 넣을 전략을 켜고 끌 수 있다.

## 설정 (config.json)

코드 수정 없이 **`config.json`** 값만 바꿔 재실행하면 반영된다(누락 키는 내장 기본값 폴백).

| 섹션 | 키 | 설명 |
|---|---|---|
| `data` | `source` | `parquet`(로컬) \| `yfinance`(온라인 다운로드) |
| | `start` / `end` | 백테스트 구간(`"YYYY-MM-DD"` 또는 `null`). start 이전은 워밍업 전용 |
| | `warmup_bars` | 백테스트 시작 이전에 **추가 확보할 워밍업 봉 수**(지표 예열) |
| | `cache` | yfinance 다운로드를 `data_dir` 에 parquet 캐시(오프라인 폴백) |
| `universe` | `{코드: 표시명}` | **테스트 종목 리스트** — 줄 추가/삭제로 편집 |
| `indicator.trend_score` | `entry`·`exit` 외 | `adx_penalty_max`(추세 약할 때 최대 차감), `adx_full_strength`(**ADX 임계값** — 이상이면 페널티 0), 가중치·기간 |
| `strategy` | `entry` / `exit` | TrendScore 진입/청산 임계(entry > exit) |
| `backtest` | `cost` / `min_bars` | 왕복 거래비용, 최소 봉 수 |

### 워밍업 자동 확장

`start` 를 지정하면 지표(TrendScore 252봉)가 **시작 시점에 이미 예열**되도록, `warmup_bars` 만큼
시작일 이전 데이터를 추가로 확보한다(yfinance 는 그만큼 더 다운로드, parquet 은 로컬 이전 구간 사용).
따라서 백테스트 시작 구간이 워밍업으로 버려지지 않고 지정 구간 전체가 유효 매매 구간이 된다.

예) yfinance, 2018~2023 백테스트:
```json
"data": { "source": "yfinance", "start": "2018-01-01", "end": "2023-01-01", "warmup_bars": 260 }
```
→ 실제 다운로드는 2016-08 부터(워밍업 351봉 > 252) 시작, 매매·성과는 2018-01 부터 집계.

## 전략 로직

- **TrendScore**(일봉, 0~100): EWMAC 앙상블(0.55) + TSMOM(0.25) + RSI(0.20) 합성 후
  ADX soft-penalty(추세 약하면 최대 −15) 차감. 252봉 미만은 워밍업(NaN).
- **스윙 진입/청산**(히스테리시스): TrendScore ≥ `entry`(기본 60) 진입, `exit`(기본 45) 미만 청산.
  그 사이 구간은 직전 상태 유지 → 휩쏘 억제.
- **체결**: 신호는 봉 i 종가에 확정, 실제 체결은 봉 i+1 **시가**(룩어헤드 방지). 롱-플랫(숏 없음),
  전액 투입/전액 청산. 청산 시 왕복 거래비용(기본 0.10%) 1회 차감.

## 유니버스 (데이터 제약)

로컬 한국 위성 ETF 일봉은 ~280봉(2025~)이라 TrendScore 252봉 워밍업 후 유효구간이 부족하다.
따라서 14년 이력(2012~) **미국 섹터/테마 ETF**(SMH·XLY·XLP·VNQ·GLD·EWY·QQQ·SPY·RSP·SCHD)를
스윙 타이밍 신호 품질 프록시로 사용한다. 신호 로직은 종목 무관하게 동일 적용된다.

## 확장 가이드 (OOP)

| 추가하고 싶은 것 | 방법 |
|---|---|
| 새 지표 (SuperTrend, MACD…) | `indicator.Indicator` 상속 → `compute()` 구현 |
| 새 전략 (교차·모멘텀·복합…) | `strategy.Strategy` 상속 → `generate_signals()` 구현 |
| 다른 데이터 소스 (DB·API) | `data.DataLoader` 상속 → `load()` 구현 |
| 다른 리포트 (PDF·MD) | `report.Reporter` 상속 → `generate()` 구현 |
| 파라미터 실험 | `config.Config` 값만 변경 후 재실행 |

각 계층은 인터페이스만 맞으면 나머지 파이프라인과 그대로 결합된다.
```python
from config import Config
from main import Pipeline
Pipeline(Config(entry=55, exit=40, cost=0.0005)).run()
```
```
