"""차트 한글 폰트 설정 — 한곳에서만.

한글 라벨이 깨지면 제안서에 그대로 실린다. 그런데 폰트 설정을 차트 모듈마다 복붙해 두면
한쪽만 고쳐진 채 다른 쪽이 조용히 깨진다(그림은 나오되 글자가 □ 로). 그래서 설정은 이 함수
하나로 모으고, 차트를 그리는 모듈은 이걸 부르기만 한다.

`matplotlib.use("Agg")` 도 여기서 한다 — 헤드리스(터미널 실행)에서 창을 띄우려다 죽지 않게
하려면 `pyplot` 임포트 **전에** 백엔드를 잡아야 하는데, 그 순서를 모듈마다 지키게 하는 것보다
한곳에 가둬 두는 편이 안전하다.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # 헤드리스에서 창 없이 저장만 — pyplot 임포트보다 먼저여야 먹는다
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

_applied = False


def apply_korean_font() -> None:
    """한글 폰트를 적용한다(여러 번 불러도 한 번만 실제로 적용).

    `koreanize_matplotlib` 가 없으면 macOS 기본 고딕으로 폴백한다 — 팀원 환경이 제각각이라
    폰트가 없다고 파이프라인이 죽으면 곤란하고, 대신 경고를 남겨 깨진 채 지나가지 않게 한다.
    """
    global _applied
    if _applied:
        return
    try:
        import koreanize_matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover — 환경 의존
        plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False
        logger.warning("koreanize_matplotlib 없음 → AppleGothic 폴백(한글 깨지면 설치 필요)")
    _applied = True
