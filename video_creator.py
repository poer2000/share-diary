"""
FFmpeg 기반 여행 영상 생성기

기능:
- 사진: Ken Burns 효과 (랜덤 줌/패닝)
- 영상: 하이라이트 구간 자동 추출
- 날짜별 자막 자동 삽입
- 인트로/아웃트로 (도시명 + 날짜)
- BGM 추가 (영상 길이에 맞게 루프)
- 장면 전환 효과
"""

import os
import json
import random
import shutil
import shutil as _shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

from media_processor import (
    is_image, is_video, get_media_datetime, get_video_duration, group_by_date
)

# 출력 해상도
WIDTH, HEIGHT = 1920, 1080
FPS = 25

# 한글 폰트 후보 경로 (OS별)
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    '/Library/Fonts/AppleGothic.ttf',
    'C:/Windows/Fonts/malgun.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
]

FONT_SMALL_CANDIDATES = [
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]


def find_font(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None  # FFmpeg 기본 폰트 사용


FONT_BOLD = find_font(FONT_CANDIDATES)
FONT_REGULAR = find_font(FONT_SMALL_CANDIDATES)

# Fix 6: module-level FFmpeg availability flag, checked once at import
_FFMPEG_OK = _shutil.which('ffmpeg') is not None


def _ffmpeg(args, desc=''):
    """FFmpeg 실행 헬퍼"""
    if not _FFMPEG_OK:
        print("  [FFmpeg] ffmpeg가 설치되어 있지 않습니다.")
        return False
    cmd = ['ffmpeg', '-y'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FFmpeg] {desc} 실패:\n{result.stderr[-500:]}")
        return False
    return True


def _font_filter(text, font_path, size, color, x, y, shadow=True, alpha_fade=None):
    """drawtext 필터 문자열 생성"""
    parts = [f"text='{text}'"]
    if font_path:
        parts.append(f"fontfile='{font_path}'")
    parts += [f"fontsize={size}", f"fontcolor={color}", f"x={x}", f"y={y}"]
    if shadow:
        parts += ["shadowcolor=black@0.7", "shadowx=2", "shadowy=2"]
    if alpha_fade:
        parts.append(f"alpha='{alpha_fade}'")
    return "drawtext=" + ":".join(parts)


def create_intro(output_path, city, start_date, end_date, duration=4):
    """검정 배경에 도시명 + 날짜 인트로"""
    d = duration
    fade = f"if(lt(t,0.8),t/0.8,if(lt(t,{d-0.8}),1,({d}-t)/0.8))"

    title_filter = _font_filter(
        city, FONT_BOLD, 90, 'white',
        '(w-text_w)/2', '(h-text_h)/2-50',
        shadow=True, alpha_fade=fade
    )
    sub_filter = _font_filter(
        f"{start_date}  -  {end_date}", FONT_REGULAR, 38, 'lightgray',
        '(w-text_w)/2', '(h-text_h)/2+60',
        shadow=True, alpha_fade=fade
    )

    # Fix 1: removed dead line; keep only the correct assignment
    vf = f"{title_filter},{sub_filter}"

    return _ffmpeg([
        '-f', 'lavfi',
        '-i', f'color=c=black:size={WIDTH}x{HEIGHT}:rate={FPS}',
        '-t', str(d),
        '-vf', vf,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-an', output_path
    ], 'intro')


def create_date_title(output_path, date_str, duration=2):
    """날짜 전환 카드 (반투명 검정 + 날짜 텍스트)"""
    fade = f"if(lt(t,0.4),t/0.4,if(lt(t,{duration-0.4}),1,({duration}-t)/0.4))"
    title_filter = _font_filter(
        date_str, FONT_BOLD, 60, 'white',
        '(w-text_w)/2', '(h-text_h)/2',
        shadow=True, alpha_fade=fade
    )
    return _ffmpeg([
        '-f', 'lavfi',
        '-i', f'color=c=0x111111:size={WIDTH}x{HEIGHT}:rate={FPS}',
        '-t', str(duration),
        '-vf', title_filter,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-an', output_path
    ], f'date_title_{date_str}')


def create_outro(output_path, city, duration=3):
    """아웃트로"""
    fade = f"if(lt(t,0.8),t/0.8,if(lt(t,{duration-0.6}),1,({duration}-t)/0.6))"
    f1 = _font_filter(city, FONT_BOLD, 70, 'white',
                      '(w-text_w)/2', '(h-text_h)/2-30', alpha_fade=fade)
    f2 = _font_filter('끝', FONT_REGULAR, 36, 'lightgray',
                      '(w-text_w)/2', '(h-text_h)/2+50', alpha_fade=fade)
    return _ffmpeg([
        '-f', 'lavfi',
        '-i', f'color=c=black:size={WIDTH}x{HEIGHT}:rate={FPS}',
        '-t', str(duration),
        '-vf', f"{f1},{f2}",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-an', output_path
    ], 'outro')


def photo_to_clip(photo_path, output_path, duration=4, date_label=None):
    """사진 → Ken Burns 효과 영상 클립"""

    # Fix 5: pre-scale to cap input at 4K before zoompan to avoid memory issues
    scale_filter = (
        f"scale='min(iw,3840)':'min(ih,2160)':force_original_aspect_ratio=decrease,"
        f"scale={WIDTH*2}:{HEIGHT*2}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH*2}:{HEIGHT*2}"
    )

    # Ken Burns 종류 랜덤 선택
    frames = duration * FPS
    effects = [
        # 줌인 (중앙 기준)
        f"zoompan=z='min(zoom+0.0015,1.5)':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        # 줌아웃
        f"zoompan=z='if(eq(on,1),1.5,max(zoom-0.0015,1.0))':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
        # 왼쪽→오른쪽 패닝 + 약한 줌
        f"zoompan=z=1.2:d={frames}:x='if(eq(on,1),0,min(x+iw/{frames*4},iw/2))':y='ih/2-(ih/zoom/2)'",
        # 오른쪽→왼쪽 패닝 + 약한 줌
        f"zoompan=z=1.2:d={frames}:x='if(eq(on,1),iw/2,max(x-iw/{frames*4},0))':y='ih/2-(ih/zoom/2)'",
        # 하단→상단 패닝
        f"zoompan=z=1.2:d={frames}:x='iw/2-(iw/zoom/2)':y='if(eq(on,1),ih/2,max(y-ih/{frames*4},0))'",
    ]
    kb_filter = random.choice(effects)

    # Fix 2: build filter list and join, avoiding trailing comma when date_label is None
    filters = [scale_filter, kb_filter, f"scale={WIDTH}:{HEIGHT}"]
    if date_label:
        filters.append(_font_filter(
            date_label, FONT_REGULAR, 34, 'white@0.9',
            '40', f'{HEIGHT}-75', shadow=True
        ))
    vf = ",".join(filters)

    ok = _ffmpeg([
        '-loop', '1',
        '-i', photo_path,
        '-vf', vf,
        '-t', str(duration),
        '-r', str(FPS),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-an', output_path
    ], f'photo {Path(photo_path).name}')

    return ok


def video_to_clip(video_path, output_path, max_clip_sec=12, date_label=None):
    """영상 → 하이라이트 클립 추출"""
    total = get_video_duration(video_path)
    if total <= 0:
        return False

    if total <= max_clip_sec:
        start = 0
        clip_dur = total
    else:
        # 초반 10% 건너뛰고, 최대 max_clip_sec 사용
        start = total * 0.08
        clip_dur = min(max_clip_sec, total * 0.7)

    # Fix 4: guard against zero or negative clip duration
    if clip_dur <= 0:
        return False

    # Fix 2: build filter list and join, avoiding trailing comma when date_label is None
    base_filters = [
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease",
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2",
    ]
    if date_label:
        base_filters.append(_font_filter(
            date_label, FONT_REGULAR, 34, 'white@0.9',
            '40', f'{HEIGHT}-75', shadow=True
        ))
    vf = ",".join(base_filters)

    return _ffmpeg([
        '-ss', f'{start:.2f}',
        '-i', video_path,
        '-t', f'{clip_dur:.2f}',
        '-vf', vf,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-ar', '44100', '-ac', '2',
        '-preset', 'fast',
        output_path
    ], f'video {Path(video_path).name}')


def concatenate_clips(clip_paths, output_path):
    """클립 이어붙이기 (concat demuxer)"""
    list_file = output_path + '.list.txt'
    valid = [p for p in clip_paths if os.path.exists(p) and os.path.getsize(p) > 0]

    if not valid:
        return False

    with open(list_file, 'w', encoding='utf-8') as f:
        for p in valid:
            f.write(f"file '{p}'\n")

    ok = _ffmpeg([
        '-f', 'concat', '-safe', '0',
        '-i', list_file,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-ar', '44100', '-ac', '2',
        '-preset', 'fast',
        output_path
    ], 'concat')

    if os.path.exists(list_file):
        os.remove(list_file)

    return ok


def add_bgm(video_path, bgm_path, output_path):
    """BGM 추가 (영상 길이에 맞게 루프, 끝에서 3초 페이드아웃)"""
    total = get_video_duration(video_path)
    fade_start = max(0, total - 3)

    return _ffmpeg([
        '-i', video_path,
        '-stream_loop', '-1', '-i', bgm_path,
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'copy',
        '-c:a', 'aac', '-ar', '44100', '-ac', '2',
        '-af', f'afade=t=out:st={fade_start:.2f}:d=3,volume=0.6',
        '-shortest',
        output_path
    ], 'add_bgm')


def mix_original_and_bgm(video_path, bgm_path, output_path, bgm_volume=0.4):
    """원본 음성 + BGM 믹싱"""
    total = get_video_duration(video_path)
    fade_start = max(0, total - 3)

    return _ffmpeg([
        '-i', video_path,
        '-stream_loop', '-1', '-i', bgm_path,
        '-filter_complex',
        f'[1:a]afade=t=out:st={fade_start:.2f}:d=3,volume={bgm_volume}[bgm];'
        '[0:a][bgm]amix=inputs=2:duration=first[aout]',
        '-map', '0:v:0', '-map', '[aout]',
        '-c:v', 'copy', '-c:a', 'aac', '-ar', '44100',
        '-shortest',
        output_path
    ], 'mix_bgm')


def calculate_photo_duration(total_photos, total_videos, target_seconds):
    """
    전체 길이 목표에 맞춰 사진 1장당 노출 시간 계산
    영상은 평균 10초로 가정
    """
    video_seconds = total_videos * 10
    remaining = max(target_seconds - video_seconds, total_photos * 3)
    per_photo = remaining / max(total_photos, 1)
    return max(3.0, min(6.0, per_photo))  # 3~6초 범위로 제한


class VideoCreator:
    def __init__(self, output_dir='output', target_minutes=5, keep_original_audio=True):
        self.output_dir = output_dir
        self.target_seconds = target_minutes * 60
        self.keep_original_audio = keep_original_audio
        os.makedirs(output_dir, exist_ok=True)

    def create(self, media_files, trip_info, bgm_path=None):
        """
        메인 함수: 미디어 파일 → 여행 영상

        media_files: 시간순 정렬된 로컬 파일 경로 리스트
        trip_info: {'city': str, 'start': datetime, 'end': datetime}
        bgm_path: BGM 파일 경로 (없으면 원본 소리만)
        반환: 완성 영상 경로 or None
        """
        city = trip_info['city']
        start_date = trip_info['start'].strftime('%Y.%m.%d')
        end_date = trip_info['end'].strftime('%Y.%m.%d')

        photos = [f for f in media_files if is_image(f)]
        videos = [f for f in media_files if is_video(f)]

        photo_duration = calculate_photo_duration(
            len(photos), len(videos), self.target_seconds
        )

        print(f"  사진 {len(photos)}장 x {photo_duration:.1f}s, 영상 {len(videos)}개")

        with tempfile.TemporaryDirectory() as tmp:
            clips = []

            # 인트로
            intro = os.path.join(tmp, 'intro.mp4')
            if create_intro(intro, city, start_date, end_date):
                clips.append(intro)

            # 날짜별로 묶어서 날짜 카드 삽입
            date_groups = group_by_date(media_files)

            clip_idx = 0
            for date_str, files in date_groups.items():
                # 날짜 전환 카드 (날짜미상 제외)
                if date_str != '날짜미상' and len(date_groups) > 1:
                    date_card = os.path.join(tmp, f'date_{clip_idx}.mp4')
                    if create_date_title(date_card, date_str):
                        clips.append(date_card)
                    clip_idx += 1

                # 해당 날짜 파일들
                for f in files:
                    out = os.path.join(tmp, f'clip_{clip_idx:04d}.mp4')
                    label = date_str if date_str != '날짜미상' else None

                    if is_image(f):
                        ok = photo_to_clip(f, out, duration=photo_duration, date_label=label)
                    else:
                        ok = video_to_clip(f, out, date_label=label)

                    if ok and os.path.exists(out):
                        clips.append(out)
                    # Fix 3: warn if clip function returned False and file was not created
                    elif not ok and not os.path.exists(out):
                        print(f"  [경고] 클립 생성 실패: {f}")
                    clip_idx += 1

            # 아웃트로
            outro = os.path.join(tmp, 'outro.mp4')
            if create_outro(outro, city):
                clips.append(outro)

            if not clips:
                print("  생성된 클립 없음")
                return None

            # 이어붙이기
            concat = os.path.join(tmp, 'concat.mp4')
            if not concatenate_clips(clips, concat):
                return None

            # BGM
            out_name = f"{city}_{trip_info['start'].strftime('%Y%m%d')}.mp4"
            out_path = os.path.join(self.output_dir, out_name)

            if bgm_path and os.path.exists(bgm_path):
                if self.keep_original_audio:
                    ok = mix_original_and_bgm(concat, bgm_path, out_path)
                else:
                    ok = add_bgm(concat, bgm_path, out_path)
                if not ok:
                    shutil.copy(concat, out_path)
            else:
                shutil.copy(concat, out_path)

        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            dur = get_video_duration(out_path)
            print(f"  완성: {out_path} ({size_mb:.0f}MB, {dur/60:.1f}분)")
            return out_path

        return None
