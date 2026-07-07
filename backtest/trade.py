"""단일 왕복 거래 기록.

진입~청산 한 사이클의 체결 정보와 순손익을 담는 값 객체. 리포트의 거래 테이블·승률·평균손익 집계에 쓰인다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    """롱-플랫 왕복 거래 하나.

    Attributes:
        entry_date / exit_date: 체결 봉의 날짜(익일 시가 체결 기준).
        entry_px / exit_px: 체결가.
        ret: 왕복 비용을 반영한 순수익률(예: 0.05 = +5%).
        bars_held: 보유 봉 수.
        exit_reason: 청산 사유("signal"=신호 청산, "eod"=마지막 봉 강제 청산).
    """
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_px: float
    exit_px: float
    ret: float
    bars_held: int
    exit_reason: str = "signal"

    @property
    def is_win(self) -> bool:
        """순수익률이 0 초과이면 승리 거래."""
        return self.ret > 0

    def as_dict(self) -> dict:
        """리포트 테이블용 딕셔너리로 직렬화한다."""
        return {
            "entry_date": self.entry_date.strftime("%Y-%m-%d"),
            "exit_date": self.exit_date.strftime("%Y-%m-%d"),
            "entry_px": round(self.entry_px, 2),
            "exit_px": round(self.exit_px, 2),
            "ret_pct": round(self.ret * 100, 2),
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason,
        }
