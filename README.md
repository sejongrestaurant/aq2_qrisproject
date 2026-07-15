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
│   └── html_report.py          #   HTMLReporter (차트 base64 임베드 + 섹터 로테이션 내역 표)
├── portfolio/                  # 자산배분 포트폴리오 (고정비중 + 리밸런싱, TrendScore 비중 틸트)
├── satellite/                  # 모멘텀 로테이션 (Top-N 동일가중 + 트레일링 스탑 + 현금대용)
├── irp/                        # ★ IRP 전략 (채권 30% 고정 + 사테라이트 70% · 분기 리밸런싱)
├── datasets/ohlcv/             # OHLCV parquet (미국 ETF + 한국 종목/ETF)
├── doc/                        # 프로젝트 발표자료(.pptx)·대본·차트(assets/)
├── config.json                 # ★ 편집 대상: 유니버스·전략 on/off·임계·비용·구간·소스
├── config/                     # 자산배분 설정: portfolio.json · satellite.json · irp.json
├── config.py                   # config.json 로더 (누락 키는 기본값 폴백)
├── main.py                     # 파이프라인 오케스트레이터 (Pipeline)
└── reports/                    # 생성된 HTML 리포트 출력 위치
```

### 데이터 흐름

```
DataLoader → SMASlopeROCStrategy(+ 비교 전략들) → Backtester → HTMLReporter
  PriceData        Signals(target_long)             BacktestResult      HTML
```

## 실행 방법

```bash
# 1) 의존성 설치(최초 1회)
pip install -r requirements.txt

# 2) 실행 — config.json + config/*.json 을 읽어 전체 파이프라인을 돌린다
python main.py

# 산출물
# → reports/index.html      유니버스 비교표 + 각 종목/전략 페이지 링크
#   reports/<코드>.html     종목/전략별 상세(성과 카드·연도별·차트·거래·로테이션 내역)
```

- **재실행**: 코드 수정 없이 `config.json`·`config/*.json` 값만 바꾸고 `python main.py` 를 다시 돌리면 반영된다.
- **데이터 소스**: 기본은 로컬 parquet(`config.json` 의 `data.source="parquet"`). 로컬에 없는 종목은
  `"yfinance"` 로 바꾸면 온라인 다운로드하며, `data.cache=true` 면 `datasets/ohlcv/<코드>.parquet` 로 캐시된다
  (다음 실행부터 오프라인 사용). 한국 종목/ETF 는 `.KS`/`.KQ` 로 받아 6자리 코드 parquet 로 저장한다.
- **일부만 실행**: 각 전략은 설정에서 개별로 끌 수 있다 — `config.json` 의 `strategies.*` on/off,
  자산배분은 `config/<전략>.json` 의 `enabled=false`(또는 파일 삭제 시 조용히 생략).

리포트는 **종목별 개별 HTML** 이 생성되고, `index.html` 유니버스 비교표(종목 클릭 → 상세)로 연결된다.
종목 페이지 상단에 **전략 비교**(내 지표 vs SuperTrend vs Buy&Hold 지표표 + 자산곡선 오버레이),
그 아래 각 전략 상세(가격 + 매수▲/매도▼ 마커, 지표 패널, 연도별 성과)가 이어진다.
자산배분(사테라이트·IRP) 페이지에는 **섹터 로테이션 내역**(교체 시점별 선정 종목, 신규 편입은 붉은색)도 실린다.
표시명은 `config.json` 의 `universe` 값을 사용한다(예: `005930·삼성전자`).

## 설정 변경 방법

설정은 두 갈래다. **스윙 전략 파이프라인**은 루트 `config.json`, **자산배분 전략**(포트폴리오·
사테라이트·IRP)은 `config/*.json`. 모두 코드 수정 없이 값만 바꿔 재실행하면 반영된다(누락 키는 내장
기본값 폴백, `_` 로 시작하는 키는 주석용으로 무시).

### 1) 스윙 전략 파이프라인 — `config.json`

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

### 2) 자산배분 전략 — `config/*.json`

각 파일은 `enabled=false` 로 끄거나 파일을 지우면 해당 전략을 조용히 생략한다.

| 파일 | 핵심 키 | 설명 |
|---|---|---|
| `config/portfolio.json` | `holdings`·`rebalance`·`weighting` | 고정비중 다자산 보유 + 주기/임계 리밸런싱(+TrendScore 비중 틸트) |
| `config/satellite.json` | `top_n`·`check_period`·`entry_score`/`exit_score`·`trailing`·`cash_ticker`·`universe` | 점수 상위 Top-N 동일가중 모멘텀 로테이션 + 트레일링 스탑 + 빈 슬롯 현금대용 |
| `config/irp.json` | `bonds`·`rebalance_period`·`rebalance_threshold`·`satellite`·`start`/`end` | 채권 고정 + 70% 사테라이트 슬리브를 분기 + 임계(±7%p) 리밸런싱(IRP) |

**IRP 설정 예시** (`config/irp.json` — 채권 30% + 사테라이트 70%):

```jsonc
{
  "enabled": true,
  "rebalance_period": "Q",          // 상위 리밸런싱 주기: M/Q/Y/none
  "rebalance_threshold": 0.07,      // 임계 안전망: 사테라이트 비중이 목표±7%p 이탈 시 리밸런싱(null=끔)
  "start": "2020-01-01",            // IRP 전용 구간(글로벌·미국섹터 상장이 늦어 2020+ 권장)
  "bonds": {                         // 채권 슬리브(각 비중, 합=채권 총배분). 사테라이트=1−합
    "153130": 0.10, "114260": 0.10, "273130": 0.10
  },
  "satellite": {
    "check_period": "M",            // 로테이션 주기
    "top_n": 7,                      // 매 체크에서 채울 슬롯 수(동일가중)
    "entry_score": 0, "exit_score": 0,  // TrendScore 편입/청산 문턱(0=게이트 끔·항상 채움)
    "cash_ticker": "153130",        // 빈 슬롯 현금대용
    "universe": ["379800", "379810", "..."]   // 후보 티커
  }
}
```

### 자주 하는 설정 변경 (레시피)

| 하고 싶은 것 | 어디를 | 어떻게 |
|---|---|---|
| 백테스트 구간 변경 | `config.json` `data.start/end` (IRP는 `config/irp.json` `start/end`) | `"YYYY-MM-DD"` 지정 또는 `null`(전체/최신) |
| 테스트 종목 추가·삭제 | `config.json` `universe` | `"코드": "표시명"` 줄 추가/삭제(로컬 parquet 없으면 `data.source="yfinance"`) |
| 특정 전략만 실행 | `config.json` `strategies.*` / `config/*.json` `enabled` | 불필요한 항목 `false` |
| IRP 채권 비중 조정 | `config/irp.json` `bonds` | 비중 합이 채권 총배분(예 40%면 각 0.13…) → 사테라이트=1−합 |
| 로테이션 개수 변경 | `config/irp.json` `satellite.top_n` / `config/satellite.json` `top_n` | 정수(슬롯당 1/top_n 동일가중) |
| 모멘텀 품질 필터 | `config/satellite.json` `entry_score`/`exit_score` | 진입≥/청산< TrendScore 문턱(히스테리시스). 0/0=끔 |
| 트레일링 스탑 조정 | `config/satellite.json` `trailing.fixed_pcts` | 비교할 고정 후퇴 비율 목록(예 `[0.15, 0.07]`) |
| 거래비용 변경 | `config.json` `backtest.cost` | 왕복 비율(예 0.0010=0.10%) |

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

## IRP ETF 전략 (채권 30% + 섹터 로테이션 70%)

개인형 퇴직연금(IRP)을 겨냥한 **자산배분 전략**. **채권 3종을 각 10%(합 30%)로 고정**해 하방을
받치고, 나머지 **70% 는 사테라이트(월간 Top-N 모멘텀 로테이션)** 로 글로벌·미국섹터·원자재·한국섹터
ETF 를 굴린다. 상위 리밸런싱(채권/사테라이트 30/70 복원)은 **분기(Q)** 마다 + 사테라이트 비중이
목표에서 **±7%p** 넘게 틀어지면(임계 안전망) 수행한다.

- **구성(composition)**: 70% 슬리브는 `satellite` 패키지를 그대로 재사용하고, IRP 는 그 위에
  '채권 바닥 + 분기 리밸런싱' 만 얹는다. 사테라이트 자산곡선을 **하나의 합성 자산**으로 보고 채권 3종과
  4-슬리브로 묶어 → 월간 로테이션(슬리브 내부)과 분기 리밸런싱(슬리브 간)이 자연히 2단으로 분리된다.
- **채권(30%)**: KODEX 단기채권(153130) · 국고채3년(114260) · 종합채권(AA-이상) 액티브(273130) 각 10%.
- **사테라이트(70%)**: 체크주기 M · **Top7** · 후보 37종(글로벌 9 + 미국섹터 9 + 원자재/리츠 4 + 한국섹터 15).
  **TrendScore 게이트 60/45**(진입≥60·청산<45)로 약세 추세 종목은 빼고, 빈 슬롯은 단기채권(153130)으로
  대피시킨다 → 하락장에서 실효 채권비중이 30%↑로 올라가는 **동적 위기 대응**(급락·회복 스트레스 완화).
- **구간**: 글로벌·미국섹터 ETF 상장이 2021~2023 이라, 유니버스가 갖춰지는 **2020년 이후**로 시작한다
  (`config/irp.json` 의 `start`). 그래야 로테이션이 대표성을 갖는다.
- **리포트**: 종목 페이지에 **섹터 로테이션 내역** 표(교체 시점별 편입일·종목수·구간수익·선정 종목)를
  실어 "그때그때 어떤 종목을 골랐는지" 를 보여준다(사테라이트 결과도 동일).

검증(2020~2026, 게이트 60/45): **CAGR 13.6% · Sharpe 1.15 · MDD −12.7% · Calmar 1.07 · Ulcer 2.90**.
채권 바닥 + 게이트 위기대응으로 벤치마크 **KODEX TRF7030**(CAGR 13.2%·MDD −22.1%·Calmar 0.60)을
수익·낙폭·회복 대부분 지표에서 앞선다. 구현: `irp/`(`IRPBacktester`·`IRPConfig`).

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
