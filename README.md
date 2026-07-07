# EST Indicator — 일봉 스윙 전략 백테스트 (SMA 각도 국면 + ROC 가속)

일봉 시세로 **시장 국면(상승·횡보·하락)과 진입 시점**을 판단하는 롱-플랫 **스윙 전략**을 백테스트하고,
결과를 자체완결형 **HTML 리포트**로 생성하는 프로젝트다. KDT AI퀀트 "나만의 인디케이터" 프로젝트의
산출물이며, 지표·전략·엔진·리포터를 **OOP 인터페이스 단위로 분리**해 개별 교체·확장이 쉽다.

대표 전략은 **`SMA 각도 국면 + ROC 가속`** 이고, 대조군인 SuperTrend·TrendScore 등과 동일 엔진에서
헤드투헤드로 비교한다. (발표자료·대본·차트: `doc/`)

## 대표 전략 — SMA 각도 국면 + ROC 가속

"살 수 있는 **국면**인가"와 "지금 진입할 **시점**인가"를 분리해 설계했다.

- **국면 (각도)**: 20일 이동평균의 **정규화 기울기 = 각도**(`(SMA_t − SMA_{t-1}) / SMA × 100`, %/일).
  스케일 무관하게 종목 공통 문턱 적용 → **상승(> +0.03) / 횡보(±0.03) / 하락(< −0.03)** 3분할.
- **시점 (ROC 가속)**: ROC(10일, 3봉 평활)의 **기울기 ≥ 0** (모멘텀 레벨이 아니라 *가속*).
- **매수** = 상승국면 **AND** ROC 가속 · **매도** = 하락국면 진입(횡보국면은 **보유 유지**로 휩쏘 방지).

검증(미국 4 + 한국 6 = 10종목, 2018~2026): 유니버스 평균 **Sharpe 0.59** (SuperTrend 0.44), **10종목 중 9종목 우위**,
MDD −34%로 Buy&Hold(−48%)보다 얕다. 구현: `strategy/swing_sma_slope.py`(`SMASlopeROCStrategy`).

## 모듈 구조

```
Indicator/
├── data/                       # 시세 로딩 (표준 스키마로 정규화)
│   ├── loader.py               #   DataLoader(ABC) → ParquetDataLoader, PriceData
│   └── yfinance_loader.py      #   YFinanceDataLoader (온라인 다운로드 + 워밍업 확장 + 캐시)
├── indicator/                  # 기술적 지표 (표준 스키마 → pd.Series)
│   ├── base.py                 #   Indicator(ABC)
│   ├── rsi.py                  #   RSIIndicator
│   ├── adx.py                  #   ADXIndicator (부호 있는 방향성 ADX, dm_mode 변형)
│   ├── trend_score.py          #   TrendScoreIndicator (EWMAC+TSMOM+RSI+ADX penalty, 0~100)
│   └── supertrend.py           #   SuperTrendIndicator (ATR 밴드 추세추종, 가격 오버레이형)
├── strategy/                   # 매매 전략 (시세 → 봉별 목표 보유상태)
│   ├── base.py                 #   Strategy(ABC), Signals(indicators/overlays)
│   ├── swing_sma_slope.py      #   ★ SMASlopeROCStrategy (각도 국면 3분할 + ROC 가속)
│   ├── swing_supertrend.py     #   SuperTrendSwingStrategy (밴드 방향 추종) — 비교군
│   ├── swing_trend_score.py    #   TrendScoreSwingStrategy (히스테리시스 + ADX 게이트 + ATR 손절)
│   └── swing_regime_trend_score.py  # RegimeGatedTrendScoreStrategy (진입=TS+ADX, 청산=lifeline)
├── backtest/                   # 백테스트 (신호 → 자산곡선·성과)
│   ├── trade.py                #   Trade (왕복 거래 기록)
│   ├── result.py               #   BacktestResult (+ 성과지표 계산, 표시명 label)
│   └── engine.py               #   Backtester (익일 시가 체결, 룩어헤드 방지)
├── report/                     # 리포트 (결과 → 산출물)
│   ├── base.py                 #   Reporter(ABC)
│   └── html_report.py          #   HTMLReporter (matplotlib 차트 base64 임베드)
├── datasets/ohlcv/             # OHLCV parquet (미국 ETF + 한국 종목)
├── doc/                        # 프로젝트 발표자료(.pptx)·대본·차트(assets/)
├── config.json                 # ★ 편집 대상: 유니버스·전략 on/off·임계·비용·구간·소스
├── config.py                   # config.json 로더 (누락 키는 기본값 폴백)
├── main.py                     # 파이프라인 오케스트레이터 (Pipeline)
└── reports/                    # 생성된 HTML 리포트 출력 위치
```

### 데이터 흐름

```
DataLoader → SMASlopeROCStrategy(+ 비교 전략들) → Backtester → HTMLReporter
  PriceData        Signals(target_long)             BacktestResult      HTML
```

## 실행

```bash
pip install -r requirements.txt
python main.py
# → reports/index.html (유니버스 비교표 + 링크)
#   reports/<코드>.html (종목별 개별 리포트: 전체기간·연도별 성과 + 매수/매도 마커 차트)
```

리포트는 **종목별 개별 HTML** 이 생성되고, `index.html` 유니버스 비교표(종목 클릭 → 상세)로 연결된다.
종목 페이지 상단에 **전략 비교**(내 지표 vs SuperTrend vs Buy&Hold 지표표 + 자산곡선 오버레이),
그 아래 각 전략 상세(가격 + 매수▲/매도▼ 마커, 지표 패널, 연도별 성과)가 이어진다.
표시명은 `config.json` 의 `universe` 값을 사용한다(예: `005930·삼성전자`).

## 설정 (config.json)

코드 수정 없이 **`config.json`** 값만 바꿔 재실행하면 반영된다(누락 키는 내장 기본값 폴백).

| 섹션 | 키 | 설명 |
|---|---|---|
| `data` | `source` | `parquet`(로컬) \| `yfinance`(온라인 다운로드) |
| | `start` / `end` | 백테스트 구간(`"YYYY-MM-DD"` \| `null`). start 이전은 워밍업 전용 |
| | `warmup_bars` | 시작 이전에 **추가 확보할 워밍업 봉 수**(지표 예열) |
| | `cache` | yfinance 다운로드를 `data_dir` 에 parquet 캐시(오프라인 폴백) |
| `universe` | `{코드: 표시명}` | **테스트 종목 리스트** — 줄 추가/삭제로 편집(리포트에 표시명 사용) |
| `strategies` | `sma_slope`·`supertrend`·`trend_score`·`regime_ts` | **비교에 넣을 전략 on/off** |
| `strategy` | `entry`/`exit`/`adx_gate` | TrendScore 계열 전략의 진입/청산 임계·ADX 게이트 |
| `supertrend` | `atr_period`/`multiplier` | SuperTrend 파라미터 |
| `stops` | `enabled`/`trailing_atr`/`stop_loss_atr` | TrendScore 전략의 ATR 손절 변형(비교용) |
| `backtest` | `cost` / `min_bars` | 왕복 거래비용(기본 0.10%), 최소 봉 수 |

> **대표 전략(SMASlopeROCStrategy)의 파라미터**(각도 문턱 ±0.03, ROC 10일·3봉평활 등)는 현재
> `main.py`의 `Pipeline._build_strategies()`에 고정되어 있다. 켜고 끄는 것은 `strategies.sma_slope`.

### 워밍업 자동 확장

`start` 를 지정하면 지표가 **시작 시점에 이미 예열**되도록 `warmup_bars` 만큼 시작일 이전 데이터를
추가 확보한다(yfinance 는 그만큼 더 다운로드, parquet 은 로컬 이전 구간 사용). 따라서 지정 구간
전체가 유효 매매 구간이 된다(앞부분이 워밍업으로 버려지지 않음).

## 전략 로직

- **체결(공통)**: 신호는 봉 i 종가에 확정, 실제 체결은 봉 i+1 **시가**(룩어헤드 방지). 롱-플랫(숏 없음),
  전액 투입/전액 청산. 청산 시 왕복 거래비용(기본 0.10%) 1회 차감.
- **SMASlopeROCStrategy (대표)**: 위 *대표 전략* 절 참고. 국면=각도 3분할, 시점=ROC 가속, 횡보 홀드.
- **SuperTrendSwingStrategy (비교군)**: ATR 밴드(기본 10×3) 방향 전환에 진입/청산.
- **TrendScoreSwingStrategy**: TrendScore(0~100) 히스테리시스 진입/청산 + ADX 게이트 + 선택적 ATR 손절.
- **RegimeGatedTrendScoreStrategy**: 진입=TrendScore+ADX 게이트, 청산=스윙-로우 lifeline(급락 방어형).

## 유니버스

기본 유니버스는 **미국 ETF 4**(SPY·QQQ·IWM·DIA) + **한국 종목 6**(KODEX200·삼성전자·네이버·알테오젠·
KB금융·KT&G)이다. 한국 종목은 yfinance(`.KS`/`.KQ`)로 받아 KRX 코드로 parquet 저장한다. 신호 로직은
**종목 무관하게 동일** 적용되며, 저변동(미국 ETF)·고변동(한국 개별주)에서의 강건성을 함께 본다.

> 거래비용 주의: 왕복 0.10%는 미국 ETF·KODEX200 ETF엔 적정하나 **한국 개별주엔 과소**하다
> (매도 증권거래세만 0.15%+ → 현실 0.25~0.40%). 고회전 전략은 비용 민감도가 크다.

## 확장 가이드 (OOP)

| 추가하고 싶은 것 | 방법 |
|---|---|
| 새 지표 (MACD·볼린저…) | `indicator.Indicator` 상속 → `compute()` 구현 |
| 새 전략 (교차·모멘텀·복합…) | `strategy.Strategy` 상속 → `generate_signals()` 구현 후 `main.py`에 배선 |
| 다른 데이터 소스 (DB·API) | `data.DataLoader` 상속 → `load()` 구현 |
| 다른 리포트 (PDF·MD) | `report.Reporter` 상속 → `generate()` 구현 |
| 파라미터 실험 | `config.json` 값 변경 후 재실행 |

각 계층은 인터페이스만 맞으면 나머지 파이프라인과 그대로 결합된다.
```python
from config import Config
from main import Pipeline
Pipeline(Config.load()).run()   # config.json 로드 후 전체 파이프라인 실행
```
