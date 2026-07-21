"""유니버스 무결성 사전 점검 — 조용한 이탈을 시끄러운 실패로 바꾼다.

이 저장소의 로딩 경로는 **실패해도 멈추지 않도록** 설계돼 있다. 개별 종목이 안 읽히면
후보에서 빼고 경고만 남기고, 현금 대용이 없으면 무이자 현금으로, 벤치마크가 없으면
무리밸런싱 드리프트로 대체한다. 파이프라인이 끊기지 않는다는 장점이 있지만, 대가가 크다:
**설정과 다른 전략이 조용히 돌아가고 결과는 그럴듯하게 나온다.**

실제로 겪은 사고 — `411060`(금) parquet 이 없는 환경에서 돌리면 WARNING 한 줄만 남기고
금이 후보에서 빠져 CAGR 13.3 / Calmar 1.05 가 나온다. 동결 수치(13.4 / 1.05)와 비슷한 데다
오히려 좋아 보여서 아무도 이상하다고 느끼지 않는다. 재현이 깨졌는데 아무도 모르는 상태가
가장 나쁘다.

그래서 **실행 전에** 설정이 요구하는 모든 종목을 실제로 읽어 보고, 하나라도 실패하면 멈춘다.
빠진 채로 굴리는 건 `allow_missing=True` 라는 **명시적 의사표시**로만 가능하다 —
'모르고 빠지는' 경로를 없애는 것이 목적이지 유연성을 없애는 게 아니다.

계층: 이 모듈은 코드 목록과 로더만 안다(설정 타입을 모른다). 어떤 종목이 필수인지 조립하는
책임은 호출자(백테스터)에 둔다 — data 계층이 irp/satellite 를 알면 의존이 거꾸로 선다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .loader import DataLoader

logger = logging.getLogger(__name__)


class UniverseIntegrityError(RuntimeError):
    """설정이 요구한 종목을 읽지 못해 백테스트를 중단한다."""


@dataclass(frozen=True)
class TickerStatus:
    """종목 하나의 점검 결과.

    Attributes:
        code: 티커.
        ok: 로드 성공 여부.
        n_bars: 읽힌 봉 수(실패면 0).
        error: 실패 사유(성공이면 None).
    """
    code: str
    ok: bool
    n_bars: int
    error: Optional[str] = None


class UniverseGuard:
    """설정이 요구하는 종목이 전부 읽히는지 실행 전에 확인한다.

    Args (생성자):
        loader: 점검에 쓸 `DataLoader`(백테스트가 쓸 것과 같은 인스턴스를 넘긴다).
        min_bars: 이 봉 수 미만이면 '죽은 티커' 로 경고한다(지표 워밍업 봉 수를 넣는다).
            None 이면 봉 수 점검을 건너뛴다. 워밍업 미달은 **경고**지 오류가 아니다 —
            상장이 늦은 종목은 나중에 후보가 되는 게 정상이기 때문이다. 다만 **전체 봉 수**가
            워밍업에 못 미치면 그 종목은 전 구간에서 단 한 번도 후보가 될 수 없다(죽은 슬롯).
    """

    def __init__(self, loader: DataLoader, min_bars: Optional[int] = None):
        self.loader = loader
        self.min_bars = min_bars

    # ── public ──────────────────────────────────────────────────────
    def check(self, codes: Iterable[str], allow_missing: bool = False) -> List[TickerStatus]:
        """모든 종목을 실제로 읽어 보고, 실패가 있으면 (원칙적으로) 중단한다.

        Args:
            codes: 필수 종목 티커들(중복은 알아서 걸러진다).
            allow_missing: True 면 실패해도 진행한다(경고만). **명시적 의사표시 전용** —
                기본값 False 를 바꾸지 말고, 필요할 때 호출부에서 플래그로만 켠다.
        Returns:
            종목별 점검 결과 리스트.
        Raises:
            UniverseIntegrityError: 하나라도 못 읽었고 `allow_missing=False` 일 때.
        """
        uniq = list(dict.fromkeys(c for c in codes if c))  # 순서 유지 중복 제거
        results = [self._probe(c) for c in uniq]
        failed = [r for r in results if not r.ok]

        for r in results:
            if r.ok and self.min_bars is not None and r.n_bars < self.min_bars:
                logger.warning(
                    f"{r.code}: 전체 {r.n_bars}봉 < 워밍업 {self.min_bars}봉 → 전 구간에서 "
                    f"후보가 될 수 없는 죽은 티커입니다(유니버스에서 빼거나 상장 이력을 확인하세요).")

        if not failed:
            logger.info(f"유니버스 무결성 확인 · {len(results)}종목 전부 로드 가능")
            return results

        detail = "\n".join(f"  · {r.code}: {r.error}" for r in failed)
        if allow_missing:
            logger.warning(f"유니버스 {len(failed)}종목 로드 실패 — allow_missing 지정으로 "
                           f"빠진 채 진행합니다. **결과는 설정과 다른 전략입니다**:\n{detail}")
            return results

        raise UniverseIntegrityError(
            f"설정이 요구한 {len(failed)}/{len(results)}종목을 읽지 못했습니다. 이대로 돌리면 "
            f"해당 종목이 조용히 빠진 채 그럴듯한 결과가 나와 재현이 깨집니다.\n{detail}\n"
            f"해결: 시세를 받아 캐시에 넣거나(ulimit -n 4096 후 yfinance, start 명시 필수), "
            f"유니버스에서 빼거나, 의도한 것이라면 --allow-missing 으로 명시하세요.")

    # ── 내부 ────────────────────────────────────────────────────────
    def _probe(self, code: str) -> TickerStatus:
        """종목 하나를 실제로 읽어 본다(성공/실패·봉 수)."""
        try:
            df = self.loader.load(code).df
        except Exception as exc:  # noqa: BLE001 — 사유를 모아 한 번에 보고하려고 삼킨다
            return TickerStatus(code=code, ok=False, n_bars=0, error=str(exc))
        return TickerStatus(code=code, ok=True, n_bars=len(df))
