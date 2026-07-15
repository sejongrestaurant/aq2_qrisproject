"""IRP(개인형 퇴직연금) ETF 전략 패키지.

채권 30% 고정 + 사테라이트(섹터/자산 모멘텀 로테이션) 70% 를 분기마다 목표비중으로 되돌리는
자산배분 전략을 담는다. 70% 슬리브는 `satellite` 패키지를 재사용하고, IRP 는 그 위에 '채권 바닥 +
분기 리밸런싱' 을 얹는다. 산출물은 기존 `BacktestResult` 로 포장되어 report 계층을 그대로 재사용한다.
"""
from .backtester import IRPBacktester
from .config import IRPConfig

__all__ = ["IRPBacktester", "IRPConfig"]
