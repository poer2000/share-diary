"""
Gemini Vision API를 사용해 여행 사진 베스트컷 선별
- 흔들림/노출 불량 사진 제거
- 중복 유사 사진 중 최선 선택
- 여행 흐름에 맞는 장면 다양성 확보
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


BATCH_SIZE = 16  # Gemini에 한 번에 보낼 최대 사진 수


def _encode_image(path, max_size=800):
    """사진을 base64로 인코딩 (리사이즈 포함)"""
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(path).convert('RGB')
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=75)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


def _select_batch(model, photo_paths, target_count):
    """배치 단위 선별 - Gemini에게 인덱스 반환 요청"""
    encoded = []
    valid_paths = []

    for path in photo_paths:
        data = _encode_image(path)
        if data:
            encoded.append(data)
            valid_paths.append(path)

    if not encoded:
        return photo_paths

    target_str = f"약 {target_count}장" if target_count else f"전체의 70% 수준"
    prompt = (
        f"여행 사진 {len(encoded)}장을 분석해주세요.\n\n"
        "다음 기준으로 좋은 사진을 골라주세요:\n"
        "1. 초점이 맞고 흔들림이 없는 사진\n"
        "2. 노출이 적절한 사진 (너무 밝거나 어둡지 않음)\n"
        "3. 거의 동일한 구도의 사진이 여러 장이면 가장 좋은 1장만\n"
        "4. 여행의 다양한 순간을 골고루 포함\n\n"
        f"선별 목표: {target_str}\n\n"
        "선택한 사진의 인덱스 번호(0부터 시작)만 콤마로 구분해서 답해주세요.\n"
        "예시: 0,2,3,5,7,10"
    )

    content = [prompt]
    for data in encoded:
        content.append({'inline_data': {'mime_type': 'image/jpeg', 'data': data}})

    try:
        response = model.generate_content(content)
        raw = response.text.strip()
        indices = []
        for token in raw.replace(' ', '').split(','):
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
    Gemini Vision으로 베스트컷 선별

    photo_paths: 사진 경로 리스트 (시간순 정렬 상태)
    target_count: 목표 사진 수 (None이면 자동)
    api_key: Gemini API 키 (없으면 환경변수 GEMINI_API_KEY 사용)
    """
    if not GENAI_AVAILABLE:
        print("  [Gemini] google-generativeai 미설치, 선별 스킵")
        return photo_paths

    key = api_key or os.getenv('GEMINI_API_KEY')
    if not key:
        print("  [Gemini] API 키 없음, 선별 스킵")
        return photo_paths

    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    total = len(photo_paths)
    if total == 0:
        return photo_paths

    # 배치 수에 맞게 target 분배
    num_batches = math.ceil(total / BATCH_SIZE)
    selected_all = []

    for i in range(num_batches):
        batch = photo_paths[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        if target_count:
            batch_target = max(2, round(target_count * len(batch) / total))
        else:
            batch_target = None

        print(f"  [Gemini] 배치 {i+1}/{num_batches} 분석 중... ({len(batch)}장)")
        selected = _select_batch(model, batch, batch_target)
        selected_all.extend(selected)

    print(f"  [Gemini] 선별 완료: {total}장 → {len(selected_all)}장")
    return selected_all
