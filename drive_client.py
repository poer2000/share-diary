import os
import io
import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.file',
]


class DriveClient:
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_file, 'wb') as f:
                pickle.dump(creds, f)

        return build('drive', 'v3', credentials=creds)

    def find_folder(self, name, parent_id=None):
        """이름으로 폴더 찾기"""
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            q += f" and '{parent_id}' in parents"

        results = self.service.files().list(
            q=q,
            fields='files(id, name)'
        ).execute()

        files = results.get('files', [])
        return files[0] if files else None

    def list_subfolders(self, parent_id):
        """하위 폴더 목록"""
        results = self.service.files().list(
            q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields='files(id, name)',
            orderBy='name'
        ).execute()
        return results.get('files', [])

    def list_media_files(self, folder_id):
        """폴더 내 사진/영상 파일 목록 (재귀 포함)"""
        media = []
        page_token = None

        while True:
            query = (
                f"'{folder_id}' in parents and trashed=false and "
                "(mimeType contains 'image/' or mimeType contains 'video/')"
            )
            kwargs = {
                'q': query,
                'fields': 'nextPageToken, files(id, name, mimeType, createdTime, size)',
                'pageSize': 1000,
            }
            if page_token:
                kwargs['pageToken'] = page_token

            results = self.service.files().list(**kwargs).execute()
            media.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break

        return media

    def download_file(self, file_id, dest_path):
        """파일 다운로드"""
        request = self.service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with io.FileIO(dest_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def upload_file(self, file_path, parent_folder_id, file_name=None):
        """Drive에 파일 업로드"""
        file_name = file_name or os.path.basename(file_path)
        file_metadata = {'name': file_name, 'parents': [parent_folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        return file.get('id')

    def get_or_create_folder(self, name, parent_id):
        """폴더 찾거나 없으면 생성"""
        folder = self.find_folder(name, parent_id)
        if folder:
            return folder['id']

        metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = self.service.files().create(body=metadata, fields='id').execute()
        return folder['id']
