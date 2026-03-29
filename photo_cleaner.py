"""
Google Drive 여행 사진 정리기

3단계 프로세스:
  [1단계] 중복/유사 사진 그룹핑 (Perceptual Hash)
           → 거의 같은 사진끼리 묶기
  [2단계] 그룹 내 베스트 선별 (Gemini Vision)
           → 얼굴 제대로 나온 것만, 나머지 삭제 후보
  [3단계] 단독 사진 품질 필터 (OpenCV + Gemini)
           → 블러/노출 불량, 표정 나쁜 사진 삭제 후보

기본 동작: 삭제 미리보기 (dry-run)
실제 삭제: --execute 플래그 필요 (Drive 휴지통으로 이동)
"""

import os
import io
import base64
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import imagehash
    from PIL import Image
    HASH_AVAILABLE = True
except ImportError:
    HASH_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# 유사 사진 판단 임계값 (낮을수록 엄격)
HASH_THRESHOLD_DUPLICATE = 6   # 거의 동일한 사진 (연속 촬영)
HASH_THRESHOLD_SIMILAR   = 12  # 비슷한 구도

BLUR_THRESHOLD  = 80
BRIGHTNESS_MIN  = 30
BRIGHTNESS_MAX  = 225


# ─────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class PhotoInfo:
    file_id: str       # Google Drive file ID
    name: str          # 파일명
    local_path: str    # 로컬 다운로드 경로
    phash: object = None         # perceptual hash
    blur_score: float = 999.0
    brightness: float = 128.0
    keep: bool = True            # 유지 여부
    delete_reason: str = ''      # 삭제 이유


# ─────────────────────────────────────────────
# 1단계: 중복 그룹핑
# ─────────────────────────────────────────────

def _compute_hash(photo: PhotoInfo) -> Optional[object]:
    if not HASH_AVAILABLE:
        return None
    try:
        img = Image.open(photo.local_path).convert('RGB')
        return imagehash.phash(img)
    except Exception:
        return None


def group_duplicates(photos: list[PhotoInfo]) -> list[list[PhotoInfo]]:
    """
    perceptual hash로 유사 사진 그룹핑.
    반환: [[photo, photo, ...], [...], ...]  (단독 사진은 1개짜리 리스트)
    """
    if not HASH_AVAILABLE:
        print("  [imagehash] 미설치 → 중복 그룹핑 스킵")
        return [[p] for p in photos]

    # 해시 계산
    for p in photos:
        p.phash = _compute_hash(p)

    groups = []
    assigned = set()

    for i, p in enumerate(photos):
        if i in assigned or p.phash is None:
            continue

        group = [p]
        assigned.add(i)

        for j, q in enumerate(photos):
            if j in assigned or q.phash is None:
                continue
            dist = p.phash - q.phash
            if dist <= HASH_THRESHOLD_SIMILAR:
                group.append(q)
                assigned.add(j)

        groups.append(group)

    # 해시 없는 사진 처리
    for i, p in enumerate(photos):
        if i not in assigned:
            groups.append([p])

    dup_groups = [g for g in groups if len(g) > 1]
    singles    = [g for g in groups if len(g) == 1]
    print(f"  [중복 그룹핑] 중복 그룹 {len(dup_groups)}개, 단독 {len(singles)}개")
    return groups


# ─────────────────────────────────────────────
# 2단계: Gemini로 그룹 내 베스트 선별
# ─────────────────────────────────────────────

def _encode(path: str, max_size=900) -> Optional[str]:
    try:
        img = Image.open(path).convert('RGB')
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


_GROUP_PROMPT = """\
거의 동일하거나 비슷한 구도로 찍힌 여행 사진 {n}장입니다.

아래 기준으로 남길 사진을 골라주세요:

[반드시 제거]
- 눈을 감은 사람이 있는 사진
- 얼굴이 흔들리거나 초점이 안 맞는 사진
- 표정이 어색하거나 찡그린 사진
- 입이 어색하게 벌어진 사진

[우선 유지]
- 모든 사람이 눈을 뜨고 자연스럽게 웃는 사진
- 구도와 빛이 가장 좋은 사진

{n}장 중 최대 {keep}장만 남기세요.
선택한 인덱스(0부터)만 콤마로 답하세요. 예: 0,2
"""

_SINGLE_PROMPT = """\
여행 사진 {n}장의 품질을 평가해주세요.

[삭제할 사진 기준]
- 눈을 감은 사람이 있음
- 표정이 어색하거나 찡그림
- 얼굴 초점 불량
- 구도가 크게 실패한 사진 (피사체 잘림, 심하게 기울어짐 등)
- 여행 내용과 무관한 단순 실수 사진 (바닥, 손 등)

[유지할 사진]
- 명백히 품질 문제가 없는 사진은 유지

삭제할 사진의 인덱스(0부터)만 콤마로 답하세요.
삭제할 게 없으면 "없음"이라고 답하세요.
"""


def _gemini_pick_best_in_group(model, group: list[PhotoInfo], max_keep: int) -> list[PhotoInfo]:
    """그룹 내에서 Gemini로 최대 max_keep장 선택"""
    if len(group) <= max_keep:
        return group

    encoded, valid = [], []
    for p in group:
        data = _encode(p.local_path)
        if data:
            encoded.append(data)
            valid.append(p)

    if not encoded:
        return group[:max_keep]

    prompt = _GROUP_PROMPT.format(n=len(encoded), keep=max_keep)
    content = [prompt] + [{'inline_data': {'mime_type': 'image/jpeg', 'data': d}} for d in encoded]

    try:
        resp = model.generate_content(content)
        raw = resp.text.strip()
        indices = [int(t) for t in raw.replace(' ', '').split(',')
                   if t.strip().isdigit() and int(t.strip()) < len(valid)]
        if not indices:
            return valid[:max_keep]
        return [valid[i] for i in sorted(set(indices))[:max_keep]]
    except Exception as e:
        print(f"    [Gemini] 그룹 선별 실패: {e}")
        return group[:max_keep]


def _gemini_filter_singles(model, photos: list[PhotoInfo]) -> list[str]:
    """단독 사진 중 품질 불량 인덱스 반환"""
    BATCH = 12
    delete_paths = []

    for i in range(0, len(photos), BATCH):
        batch = photos[i:i + BATCH]
        encoded, valid = [], []
        for p in batch:
            data = _encode(p.local_path)
            if data:
                encoded.append(data)
                valid.append(p)

        if not encoded:
            continue

        prompt = _SINGLE_PROMPT.format(n=len(encoded))
        content = [prompt] + [{'inline_data': {'mime_type': 'image/jpeg', 'data': d}} for d in encoded]

        try:
            resp = model.generate_content(content)
            raw = resp.text.strip()
            if '없음' in raw:
                continue
            for t in raw.replace(' ', '').split(','):
                if t.strip().isdigit():
                    idx = int(t.strip())
                    if idx < len(valid):
                        delete_paths.append(valid[idx].local_path)
        except Exception as e:
            print(f"    [Gemini] 단독 필터 실패: {e}")

    return delete_paths


# ─────────────────────────────────────────────
# 3단계: OpenCV 기술 품질 필터
# ─────────────────────────────────────────────

def _opencv_filter(photos: list[PhotoInfo]) -> list[PhotoInfo]:
    """블러/밝기 불량 사진에 플래그 세팅"""
    if not CV2_AVAILABLE:
        return photos

    for p in photos:
        try:
            img = cv2.imread(p.local_path)
            if img is None and HASH_AVAILABLE:
                pil = Image.open(p.local_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            if img is None:
                continue

            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            center = gray[h//4:3*h//4, w//4:3*w//4]
            p.blur_score = min(
                cv2.Laplacian(gray, cv2.CV_64F).var(),
                cv2.Laplacian(center, cv2.CV_64F).var()
            )
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            p.brightness = float(hsv[:, :, 2].mean())
        except Exception:
            pass

    return photos


# ─────────────────────────────────────────────
# 메인: 전체 정리 로직
# ─────────────────────────────────────────────

def analyze_for_cleanup(
    photos: list[PhotoInfo],
    api_key: str = None,
    max_keep_per_group: int = 2,
) -> tuple[list[PhotoInfo], list[PhotoInfo]]:
    """
    전체 정리 분석.

    반환: (유지할 사진 리스트, 삭제할 사진 리스트)
    """
    if not photos:
        return [], []

    # ── OpenCV 기술 품질 측정 ──
    print(f"\n  [1단계] 블러/밝기 측정 중... ({len(photos)}장)")
    _opencv_filter(photos)

    # 명백한 기술 불량 먼저 표시
    tech_bad = []
    tech_ok  = []
    for p in photos:
        if p.blur_score < BLUR_THRESHOLD:
            p.keep = False
            p.delete_reason = f'블러 (score={p.blur_score:.0f})'
            tech_bad.append(p)
        elif p.brightness < BRIGHTNESS_MIN:
            p.keep = False
            p.delete_reason = f'너무 어두움 (brightness={p.brightness:.0f})'
            tech_bad.append(p)
        elif p.brightness > BRIGHTNESS_MAX:
            p.keep = False
            p.delete_reason = f'과노출 (brightness={p.brightness:.0f})'
            tech_bad.append(p)
        else:
            tech_ok.append(p)

    print(f"     기술 불량 {len(tech_bad)}장 발견")

    # ── 중복 그룹핑 ──
    print(f"\n  [2단계] 중복/유사 사진 그룹핑 중...")
    groups = group_duplicates(tech_ok)

    # Gemini 없으면 그룹당 상위 max_keep_per_group만 유지
    if not GENAI_AVAILABLE or not (api_key or os.getenv('GEMINI_API_KEY')):
        print("  [Gemini] API 키 없음, 그룹당 앞쪽 사진 유지")
        to_keep, to_delete = list(tech_ok), list(tech_bad)
        for g in groups:
            if len(g) > max_keep_per_group:
                for p in g[max_keep_per_group:]:
                    p.keep = False
                    p.delete_reason = '중복 (AI 없이 자동 제거)'
                    to_delete.append(p)
        to_keep = [p for p in to_keep if p.keep]
        return to_keep, to_delete

    key = api_key or os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # ── Gemini: 중복 그룹 내 베스트 선별 ──
    dup_groups   = [g for g in groups if len(g) > 1]
    single_photos = [g[0] for g in groups if len(g) == 1]

    group_deleted = []
    group_kept    = []

    print(f"\n  [3단계] 중복 그룹 {len(dup_groups)}개 Gemini 분석 중...")
    for gi, g in enumerate(dup_groups):
        print(f"    그룹 {gi+1}/{len(dup_groups)} ({len(g)}장) → 최대 {max_keep_per_group}장 유지", end='', flush=True)
        best = _gemini_pick_best_in_group(model, g, max_keep_per_group)
        best_paths = {p.local_path for p in best}

        kept_n = 0
        for p in g:
            if p.local_path in best_paths:
                group_kept.append(p)
                kept_n += 1
            else:
                p.keep = False
                p.delete_reason = f'중복 그룹 내 비선택 (그룹크기={len(g)})'
                group_deleted.append(p)

        print(f" → {kept_n}장 유지, {len(g)-kept_n}장 삭제")

    # ── Gemini: 단독 사진 품질 필터 ──
    print(f"\n  [4단계] 단독 사진 {len(single_photos)}장 품질 필터 중...")
    bad_paths = set(_gemini_filter_singles(model, single_photos))

    single_kept    = []
    single_deleted = []
    for p in single_photos:
        if p.local_path in bad_paths:
            p.keep = False
            p.delete_reason = '품질 불량 (표정/구도/초점)'
            single_deleted.append(p)
        else:
            single_kept.append(p)

    print(f"     단독 사진 {len(single_deleted)}장 삭제 예정")

    # 최종 집계
    all_delete = tech_bad + group_deleted + single_deleted
    all_keep   = group_kept + single_kept

    return all_keep, all_delete


def print_cleanup_summary(keep: list[PhotoInfo], delete: list[PhotoInfo], folder_name: str):
    """삭제 미리보기 출력"""
    total = len(keep) + len(delete)
    print(f"\n{'='*55}")
    print(f"  정리 결과 미리보기: {folder_name}")
    print(f"{'='*55}")
    print(f"  전체:   {total}장")
    print(f"  유지:   {len(keep)}장")
    print(f"  삭제:   {len(delete)}장")
    print()

    if delete:
        # 이유별 집계
        reasons: dict[str, int] = {}
        for p in delete:
            key = p.delete_reason.split('(')[0].strip()
            reasons[key] = reasons.get(key, 0) + 1

        print("  삭제 이유:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count}장")

        print("\n  삭제 예정 파일 (일부):")
        for p in delete[:10]:
            print(f"    {p.name:40s}  ({p.delete_reason})")
        if len(delete) > 10:
            print(f"    ... 외 {len(delete)-10}장")

    print()


def execute_cleanup(drive_client, delete: list[PhotoInfo], dry_run: bool = True):
    """실제 Drive 휴지통 이동 실행"""
    if dry_run:
        print("  [dry-run] 실제 삭제하려면 --execute 플래그를 추가하세요.")
        return

    print(f"  Drive 휴지통으로 이동 중... ({len(delete)}장)")
    success, fail = 0, 0
    for p in delete:
        try:
            drive_client.trash_file(p.file_id)
            success += 1
        except Exception as e:
            print(f"  실패: {p.name} ({e})")
            fail += 1

    print(f"  완료: {success}장 이동, {fail}장 실패")
    print("  Drive 휴지통에서 30일 내 복구 가능합니다.")
