"""리포트 산출물의 공통 뼈대 — 디렉터리 보장·차트 저장·CSV(BOM) 쓰기.

적립식(`report.DCAReport`)과 노출(`exposure_report.ExposureReport`)이 같은 저장 관용구를
쓴다: `reports/` 생성, `tight_layout→savefig(dpi=150)→close`, 엑셀 호환 BOM(utf-8-sig) CSV.
이걸 각 리포트에 복붙해 두면 한쪽만 고쳐진 채 다른 쪽이 어긋난다(fonts 설정을 한곳으로 모은
것과 같은 이유). 그래서 공통부를 이 베이스에 두고, 각 리포트는 표·차트의 '내용'만 맡는다.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from .fonts import apply_korean_font

apply_korean_font()  # 백엔드(Agg)·한글 폰트 설정은 fonts 모듈이 한곳에서 맡는다
import matplotlib.pyplot as plt  # noqa: E402 — apply_korean_font() 이후여야 백엔드가 먹는다

logger = logging.getLogger(__name__)


class ReportWriter:
    """CSV·PNG 를 `out_dir` 에 떨구는 리포트의 공통 베이스.

    Args (생성자):
        out_dir: 산출 디렉터리(없으면 만든다).
    """

    def __init__(self, out_dir: str = "reports"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _save(self, fig, name: str) -> str:
        """차트를 `{name}.png` 로 저장하고 경로를 돌려준다."""
        path = os.path.join(self.out_dir, f"{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"차트 저장 · {path}")
        return path

    def _write_csv(self, df: pd.DataFrame, name: str, *, index: bool = False) -> str:
        """표를 `{name}.csv` 로 저장한다(엑셀에서 한글 안 깨지게 BOM).

        Args:
            index: 행 인덱스를 함께 쓸지. 날짜 인덱스 표(노출)는 True, 행 번호가 의미 없는
                표(적립식 요약)는 False.
        """
        path = os.path.join(self.out_dir, f"{name}.csv")
        df.to_csv(path, index=index, encoding="utf-8-sig")
        logger.info(f"CSV 저장 · {path} ({len(df)}행)")
        return path
