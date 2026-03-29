"""
여행 사진/영상 자동 정리 & 영상 생성기

[영상 생성]
  python main.py                           # 전체 여행 폴더 처리
  python main.py --trip 뉴욕               # 특정 여행만
  python main.py --trip 뉴욕 --bgm bgm.mp3
  python main.py --no-ai                   # Gemini 선별 없이 전체 사용
  python main.py --minutes 7               # 목표 영상 길이 7분
  python main.py --upload                  # 완성 영상 Drive에 업로드

[사진 정리 (Drive에서 불량/중복 사진 삭제)]
  python main.py --clean --trip 뉴욕       # 뉴욕 폴더 정리 미리보기
  python main.py --clean --trip 뉴욕 --execute   # 실제 삭제 (휴지통 이동)
  python main.py --clean                   # 전체 여행 폴더 정리 미리보기
  python main.py --clean --keep 3          # 중복 그룹당 최대 3장 유지
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from drive_client import DriveClient
from media_processor import parse_folder_name, sort_media_by_time, is_image, is_video
from gemini_selector import select_best_shots
from video_creator import VideoCreator
from photo_cleaner import PhotoInfo, analyze_for_cleanup, print_cleanup_summary, execute_cleanup

TRAVEL_FOLDER_NAME = '여행'
DOWNLOAD_DIR = 'downloads'
OUTPUT_DIR = 'output'


# ─────────────────────────────────────────────
# 공통: 파일 다운로드
# ─────────────────────────────────────────────

def download_trip_media(drive_client, folder, download_dir, return_meta=False):
    """
    여행 폴더 미디어 파일 다운로드.
    return_meta=True이면 DriveFile 메타 포함한 (local_path, drive_file) 리스트 반환
    """
    folder_name = folder['name']
    folder_id   = folder['id']
    trip_dir    = os.path.join(download_dir, folder_name)
    os.makedirs(trip_dir, exist_ok=True)

    print(f"  파일 목록 가져오는 중...")
    drive_files = drive_client.list_media_files(folder_id)

    if not drive_files:
        return []

    print(f"  총 {len(drive_files)}개 파일 발견")

    results = []
    for f in tqdm(drive_files, desc='  다운로드', unit='파일'):
        local_path = os.path.join(trip_dir, f['name'])
        if not os.path.exists(local_path):
            try:
                drive_client.download_file(f['id'], local_path)
            except Exception as e:
                print(f"\n  다운로드 실패: {f['name']} ({e})")
                continue
        results.append((local_path, f) if return_meta else local_path)

    return results


# ─────────────────────────────────────────────
# 영상 생성
# ─────────────────────────────────────────────

def process_trip_video(drive_client, folder, args):
    """단일 여행 폴더 → 영상 생성"""
    trip_info = parse_folder_name(folder['name'])
    if not trip_info:
        print(f"  폴더명 형식 불일치 (건너뜀): {folder['name']}")
        return None

    city  = trip_info['city']
    start = trip_info['start'].strftime('%Y.%m.%d')
    end   = trip_info['end'].strftime('%Y.%m.%d')

    print(f"\n{'='*55}")
    print(f"  {city}  ({start} ~ {end})")
    print(f"{'='*55}")

    local_files = download_trip_media(drive_client, folder, DOWNLOAD_DIR)
    if not local_files:
        print("  미디어 파일 없음, 건너뜀")
        return None

    print("  시간순 정렬 중...")
    sorted_files = sort_media_by_time(local_files)

    photos = [f for f in sorted_files if is_image(f)]
    videos = [f for f in sorted_files if is_video(f)]
    print(f"  사진: {len(photos)}장 / 영상: {len(videos)}개")

    if not args.no_ai and photos:
        target_photos = max(5, int(args.minutes * 60 * 0.7 / 4.5))
        print(f"  AI 베스트컷 선별 중... (목표 {target_photos}장)")
        photos = select_best_shots(
            photos,
            target_count=target_photos,
            api_key=os.getenv('GEMINI_API_KEY')
        )

    final_media = sort_media_by_time(photos + videos)

    print("  영상 생성 중...")
    creator = VideoCreator(
        output_dir=OUTPUT_DIR,
        target_minutes=args.minutes,
        keep_original_audio=True
    )
    output_path = creator.create(final_media, trip_info, bgm_path=args.bgm)

    if output_path and args.upload:
        print("  Drive에 업로드 중...")
        travel_folder = drive_client.find_folder(TRAVEL_FOLDER_NAME)
        if travel_folder:
            folder_id = drive_client.get_or_create_folder('완성영상', travel_folder['id'])
            fid = drive_client.upload_file(output_path, folder_id)
            print(f"  업로드 완료 (id: {fid})")

    return output_path


# ─────────────────────────────────────────────
# 사진 정리
# ─────────────────────────────────────────────

def process_trip_clean(drive_client, folder, args):
    """단일 여행 폴더 사진 정리 (중복/불량 → 휴지통)"""
    trip_info = parse_folder_name(folder['name'])
    label = trip_info['city'] if trip_info else folder['name']

    print(f"\n{'='*55}")
    print(f"  정리 대상: {folder['name']}")
    print(f"{'='*55}")

    # 사진만 다운로드 (영상은 정리 대상 아님)
    raw = download_trip_media(drive_client, folder, DOWNLOAD_DIR, return_meta=True)
    if not raw:
        print("  파일 없음, 건너뜀")
        return

    photos = [
        PhotoInfo(file_id=meta['id'], name=meta['name'], local_path=local)
        for local, meta in raw
        if is_image(local)
    ]

    if not photos:
        print("  사진 없음, 건너뜀")
        return

    print(f"  사진 {len(photos)}장 분석 시작")

    keep, delete = analyze_for_cleanup(
        photos,
        api_key=os.getenv('GEMINI_API_KEY'),
        max_keep_per_group=args.keep,
    )

    print_cleanup_summary(keep, delete, folder['name'])

    if delete:
        execute_cleanup(drive_client, delete, dry_run=not args.execute)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Google Drive 여행 사진 정리 & 영상 자동 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 공통
    parser.add_argument('--trip', '-t',  help="특정 여행 폴더명 (예: 뉴욕)")
    parser.add_argument('--list', '-l',  action='store_true', help="여행 폴더 목록만 출력")

    # 영상 생성
    parser.add_argument('--bgm',     '-b', help="BGM 파일 경로")
    parser.add_argument('--minutes', '-m', type=float, default=5.0, help="목표 영상 길이(분)")
    parser.add_argument('--no-ai',         action='store_true', help="Gemini AI 비활성화")
    parser.add_argument('--upload',  '-u', action='store_true', help="완성 영상 Drive 업로드")

    # 사진 정리
    parser.add_argument('--clean',         action='store_true',
                        help="사진 정리 모드 (중복/불량 제거)")
    parser.add_argument('--execute',       action='store_true',
                        help="[--clean 전용] 실제 삭제 실행 (기본: 미리보기만)")
    parser.add_argument('--keep',          type=int, default=2,
                        help="[--clean 전용] 중복 그룹당 유지할 최대 사진 수 (기본: 2)")

    args = parser.parse_args()

    # 사전 점검
    if not os.path.exists('credentials.json'):
        print("[오류] credentials.json 파일이 없습니다.")
        print("  Google Cloud Console에서 OAuth2 클라이언트 ID를 다운로드해 주세요.")
        sys.exit(1)

    if not args.clean and args.bgm and not os.path.exists(args.bgm):
        print(f"[경고] BGM 파일 없음: {args.bgm}")
        args.bgm = None

    if args.clean and args.execute:
        print("⚠️  실제 삭제 모드입니다. Drive 휴지통으로 이동됩니다 (복구 가능).")

    # Drive 연결
    print("Google Drive 연결 중...")
    drive_client = DriveClient()

    travel_folder = drive_client.find_folder(TRAVEL_FOLDER_NAME)
    if not travel_folder:
        print(f"[오류] Drive에서 '{TRAVEL_FOLDER_NAME}' 폴더를 찾을 수 없습니다.")
        sys.exit(1)

    trip_folders = drive_client.list_subfolders(travel_folder['id'])
    if not trip_folders:
        print(f"'{TRAVEL_FOLDER_NAME}' 폴더 안에 하위 폴더가 없습니다.")
        sys.exit(0)

    # 폴더 목록 출력
    if args.list:
        print(f"\n[여행 폴더 목록] ({len(trip_folders)}개)")
        for f in trip_folders:
            info = parse_folder_name(f['name'])
            if info:
                print(f"  {info['city']:12s}  {info['start'].strftime('%Y.%m.%d')} ~ {info['end'].strftime('%Y.%m.%d')}")
            else:
                print(f"  {f['name']}  (형식 불일치)")
        return

    # 폴더 필터링
    if args.trip:
        trip_folders = [f for f in trip_folders if args.trip in f['name']]
        if not trip_folders:
            print(f"'{args.trip}'이 포함된 폴더를 찾을 수 없습니다.")
            sys.exit(1)

    print(f"\n처리할 여행: {len(trip_folders)}개")
    for f in trip_folders:
        print(f"  - {f['name']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # 실행
    if args.clean:
        for folder in trip_folders:
            process_trip_clean(drive_client, folder, args)
        print(f"\n정리 완료.")
        if not args.execute:
            print("실제 삭제하려면 --execute 플래그를 추가하세요.")
    else:
        results = []
        for folder in trip_folders:
            result = process_trip_video(drive_client, folder, args)
            if result:
                results.append(result)
        print(f"\n{'='*55}")
        print(f"완료! {len(results)}개 영상 생성됨:")
        for r in results:
            print(f"  {r}")


if __name__ == '__main__':
    main()
