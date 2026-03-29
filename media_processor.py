import os
import re
import json
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.tiff', '.bmp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.3gp', '.mts', '.m2ts'}


def parse_folder_name(folder_name):
    """
    '뉴욕)231212-231220' 형식 파싱
    반환: {'city': '뉴욕', 'start': datetime, 'end': datetime}
    """
    match = re.match(r'(.+)\)(\d{6})-(\d{6})', folder_name)
    if not match:
        return None

    city = match.group(1).strip()
    start_str = match.group(2)
    end_str = match.group(3)

    try:
        start = datetime.strptime('20' + start_str, '%Y%m%d')
        end = datetime.strptime('20' + end_str, '%Y%m%d')
    except ValueError:
        return None

    return {'city': city, 'start': start, 'end': end}


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def get_photo_datetime(file_path):
    """사진 EXIF에서 촬영 시간 추출"""
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(file_path)
        exif_data = img._getexif()
        if not exif_data:
            return None
        # DateTimeOriginal(36867) 우선, 없으면 DateTime(306)
        for tag_id in (36867, 36868, 306):
            if tag_id in exif_data:
                dt_str = exif_data[tag_id]
                try:
                    return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    pass
    except Exception:
        pass
    return None


def get_video_datetime(file_path):
    """ffprobe로 영상 촬영 시간 추출"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})

        for key in ('creation_time', 'com.apple.quicktime.creationdate', 'date'):
            if key in tags:
                raw = tags[key]
                for fmt in ('%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
                            '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S%z'):
                    try:
                        dt = datetime.strptime(raw[:26], fmt[:len(raw)])
                        return dt.replace(tzinfo=None)
                    except ValueError:
                        pass
    except Exception:
        pass
    return None


def get_datetime_from_filename(file_path):
    """파일명에서 날짜 추출 (IMG_20231212_143022.jpg 형식)"""
    basename = os.path.basename(file_path)
    patterns = [
        r'(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})',  # 20231212_143022
        r'(\d{4})-(\d{2})-(\d{2})',                           # 2023-12-12
    ]
    for pattern in patterns:
        m = re.search(pattern, basename)
        if m:
            try:
                groups = m.groups()
                if len(groups) == 6:
                    return datetime(*[int(g) for g in groups])
                elif len(groups) == 3:
                    return datetime(*[int(g) for g in groups])
            except ValueError:
                pass
    return None


def get_media_datetime(file_path):
    """미디어 파일의 촬영 시간 반환 (우선순위: EXIF > ffprobe > 파일명)"""
    if is_image(file_path):
        dt = get_photo_datetime(file_path)
    elif is_video(file_path):
        dt = get_video_datetime(file_path)
    else:
        dt = None

    if dt is None:
        dt = get_datetime_from_filename(file_path)

    return dt


def sort_media_by_time(media_files):
    """시간순 정렬. 날짜 없는 파일은 뒤로"""
    dated, undated = [], []

    for f in media_files:
        dt = get_media_datetime(f)
        if dt:
            dated.append((dt, f))
        else:
            undated.append(f)

    dated.sort(key=lambda x: x[0])
    return [f for _, f in dated] + undated


def group_by_date(media_files):
    """날짜별로 그룹핑 {date_str: [file, ...]}"""
    groups = {}
    for f in media_files:
        dt = get_media_datetime(f)
        key = dt.strftime('%Y.%m.%d') if dt else '날짜미상'
        groups.setdefault(key, []).append(f)
    return groups


def get_video_duration(file_path):
    """영상 길이(초) 반환"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception:
        return 0
