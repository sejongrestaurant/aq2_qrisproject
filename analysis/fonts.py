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

# 폴백 후보 — macOS · Windows · Linux 순으로 흔한 한글 고딕. 위에서부터 설치 여부를 확인한다.
_FALLBACK_FONTS = ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR",
                   "Noto Sans KR", "Gulim", "Batang")


def apply_korean_font() -> None:
    """한글 폰트를 적용한다(여러 번 불러도 한 번만 실제로 적용).

    `koreanize_matplotlib` 가 없으면 **설치된 한글 폰트를 찾아** 폴백한다 — 팀원 환경이
    제각각이라 폰트가 없다고 파이프라인이 죽으면 곤란하고, 대신 경고를 남겨 깨진 채 지나가지
    않게 한다.

    폴백을 목록으로 두는 이유: 예전엔 `AppleGothic` 하나만 걸었는데, 그건 macOS 전용이라
    Windows 팀원 환경에서는 **없는 폰트를 지정한 채로 그림이 나왔다**(경고만 남고 한글이 전부
    □). 폰트가 깨진 그림은 제안서에 그대로 실리므로, 실제로 설치된 것을 골라 잡는다.
    """
    global _applied
    if _applied:
        return
    try:
        import koreanize_matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover — 환경 의존
        picked = _first_installed(_FALLBACK_FONTS)
        if picked:
            plt.rcParams["font.family"] = picked
            logger.warning(f"koreanize_matplotlib 없음 → 설치된 '{picked}' 로 폴백")
        else:
            logger.warning("koreanize_matplotlib 없고 한글 폰트도 못 찾음 → 차트 한글이 "
                           "깨집니다(pip install koreanize-matplotlib 권장)")
        plt.rcParams["axes.unicode_minus"] = False
    _applied = True


def _first_installed(names: tuple[str, ...]) -> str | None:
    """후보 중 이 환경에 **실제로 설치된** 첫 폰트 이름을 돌려준다(없으면 None)."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in installed), None)
