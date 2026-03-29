"""
Gemini Vision API + OpenCV 2단계 베스트컷 선별

[1단계] OpenCV 기술 품질 필터 (사전 제거)
  - 블러 점수 (Laplacian 분산): 선명도 측정
  - 밝기 검사: 너무 어둡거나 과노출인 사진 제거

[2단계] Gemini Vision AI 선별 (얼굴/표정 중심)
  - 눈 감음 감지
  - 표정 자연스러움 (억지웃음, 무표정 등)
  - 사람이 여럿일 때 모두 잘 나온 사진 우선
  - 중복 구도 중 최선 선택
  - 여행 흐름 다양성 확보
"""

import os
import io
import base64
import math
from pathlib import Path

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


BATCH_SIZE = 12  # 얼굴 분석은 더 세밀하게 → 배치 크기 줄임

# 기술 품질 임계값
BLUR_THRESHOLD = 80      # Laplacian 분산 (낮을수록 흐림, 80 미만 제거)
BRIGHTNESS_MIN = 30      # 너무 어두운 사진 제거 (0~255)
BRIGHTNESS_MAX = 225     # 과노출 사진 제거


# ─────────────────────────────────────────────
# 1단계: OpenCV 기술 품질 필터
# ─────────────────────────────────────────────

def _blur_score(image_path):
    """
    Laplacian 분산으로 블러 점수 계산.
    높을수록 선명, 낮을수록 흐림.
    """
    if not CV2_AVAILABLE:
        return 999  # opencv 없으면 통과
    try:
        img = cv2.imread(image_path)
        if img is None:
            # HEIC 등 OpenCV가 못 읽는 포맷 → Pillow로 변환
            if PIL_AVAILABLE:
                pil = Image.open(image_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            else:
                return 999

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 전체 블러 점수
        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 사람이 있는 영역(중앙)은 더 엄격하게 체크
        h, w = gray.shape
        center = gray[h//4:3*h//4, w//4:3*w//4]
        center_score = cv2.Laplacian(center, cv2.CV_64F).var()

        # 전체와 중앙 점수 중 낮은 쪽 반환 (더 엄격)
        return min(score, center_score)
    except Exception:
        return 999


def _brightness_score(image_path):
    """평균 밝기 반환 (0~255)"""
    if not CV2_AVAILABLE:
        return 128
    try:
        img = cv2.imread(image_path)
        if img is None:
            if PIL_AVAILABLE:
                pil = Image.open(image_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            else:
                return 128
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean())
    except Exception:
        return 128


def technical_filter(photo_paths, verbose=True):
    """
    OpenCV로 기술적으로 나쁜 사진 사전 제거.
    블러 / 과노출 / 과암 사진을 걸러냄.
    """
    if not CV2_AVAILABLE:
        if verbose:
            print("  [OpenCV] 미설치 → 기술 필터 스킵")
        return photo_paths

    passed, removed = [], []

    for path in photo_paths:
        blur = _blur_score(path)
        brightness = _brightness_score(path)

        if blur < BLUR_THRESHOLD:
            removed.append((path, f"블러 score={blur:.0f}"))
        elif brightness < BRIGHTNESS_MIN:
            removed.append((path, f"너무 어두움 brightness={brightness:.0f}"))
        elif brightness > BRIGHTNESS_MAX:
            removed.append((path, f"과노출 brightness={brightness:.0f}"))
        else:
            passed.append(path)

    if verbose and removed:
        print(f"  [OpenCV] {len(removed)}장 기술 품질 제거 "
              f"(블러/밝기) → {len(passed)}장 남음")

    # 너무 많이 제거됐으면 원본 반환
    if len(passed) < max(3, len(photo_paths) * 0.3):
        if verbose:
            print("  [OpenCV] 너무 많이 제거됨, 필터 취소")
        return photo_paths

    return passed


# ─────────────────────────────────────────────
# 2단계: Gemini Vision 얼굴/표정 특화 선별
# ─────────────────────────────────────────────

def _encode_image(path, max_size=1024):
    """사진을 base64로 인코딩 (얼굴 분석용으로 해상도 높임)"""
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(path).convert('RGB')
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


_GEMINI_PROMPT = """\
여행 사진 {n}장을 분석해서 베스트컷을 골라주세요.

## 반드시 제거할 사진
- 눈을 감은 사람이 있는 사진
- 얼굴이 흔들리거나 초점이 안 맞는 사진
- 표정이 어색하거나 찡그린 사진
- 여러 명 중 한 명이라도 눈 감거나 표정이 나쁜 단체 사진
- 거의 동일한 구도로 연속 촬영된 사진 중 가장 덜 좋은 것들

## 우선 선택할 사진
- 모든 사람이 자연스럽게 웃고 눈을 뜬 사진
- 여행지의 분위기와 감성이 잘 담긴 사진
- 풍경, 음식, 인물, 활동 등 다양한 순간을 골고루 포함
- 빛이 예쁘게 들어온 사진 (골든아워, 자연광 등)

## 선별 목표
{target}

선택한 사진의 인덱스 번호(0부터 시작)만, 콤마로 구분해서 답해주세요.
다른 설명 없이 숫자만 답하세요.
예시: 0,2,3,5,7,10
"""


def _select_batch(model, photo_paths, target_count):
    """배치 단위 선별"""
    encoded, valid_paths = [], []

    for path in photo_paths:
        data = _encode_image(path)
        if data:
            encoded.append(data)
            valid_paths.append(path)

    if not encoded:
        return photo_paths

    target_str = f"약 {target_count}장" if target_count else "전체의 70% 수준"
    prompt = _GEMINI_PROMPT.format(n=len(encoded), target=target_str)

    content = [prompt]
    for data in encoded:
        content.append({'inline_data': {'mime_type': 'image/jpeg', 'data': data}})

    try:
        response = model.generate_content(content)
        raw = response.text.strip()

        indices = []
        for token in raw.replace(' ', '').split(','):
            token = token.strip()
            if token.isdigit():
                idx = int(token)
                if 0 <= idx < len(valid_paths):
                    indices.append(idx)

        selected = [valid_paths[i] for i in sorted(set(indices))]

        # 너무 적게 선별됐으면 원본 반환
        if len(selected) < max(2, len(valid_paths) * 0.3):
            return valid_paths

        return selected

    except Exception as e:
        print(f"  [Gemini] 배치 분석 실패: {e}")
        return valid_paths


def select_best_shots(photo_paths, target_count=None, api_key=None):
    """
    2단계 베스트컷 선별

    photo_paths: 사진 경로 리스트 (시간순 정렬 상태)
    target_count: 목표 사진 수 (None이면 자동)
    api_key: Gemini API 키
    """
    if not photo_paths:
        return photo_paths

    # ── 1단계: OpenCV 기술 필터 ──
    print(f"  [1단계] OpenCV 기술 품질 필터 ({len(photo_paths)}장)")
    after_tech = technical_filter(photo_paths)

    # ── 2단계: Gemini 얼굴/표정 선별 ──
    if not GENAI_AVAILABLE:
        print("  [2단계] google-generativeai 미설치, 스킵")
        return after_tech

    key = api_key or os.getenv('GEMINI_API_KEY')
    if not key:
        print("  [2단계] Gemini API 키 없음, 스킵")
        return after_tech

    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    total = len(after_tech)
    num_batches = math.ceil(total / BATCH_SIZE)
    selected_all = []

    print(f"  [2단계] Gemini 얼굴/표정 선별 ({total}장, {num_batches}배치)")

    for i in range(num_batches):
        batch = after_tech[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_target = max(2, round(target_count * len(batch) / total)) if target_count else None

        print(f"    배치 {i+1}/{num_batches} 분석 중... ({len(batch)}장)", end='', flush=True)
        selected = _select_batch(model, batch, batch_target)
        selected_all.extend(selected)
        print(f" → {len(selected)}장 선택")

    print(f"  [선별 완료] {len(photo_paths)}장 → {len(selected_all)}장")
    return selected_all
