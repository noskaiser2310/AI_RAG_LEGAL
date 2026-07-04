"""
Script tải index data từ Google Drive về thư mục data/indexes/.

Folder ID mặc định: 1FSFKamm_fwss0sI4wBnBOxOko1gIGMH9
Link: https://drive.google.com/drive/folders/1FSFKamm_fwss0sI4wBnBOxOko1gIGMH9

Cách dùng:
    # Tải toàn bộ folder (mặc định)
    python scripts/download_index_gdrive.py

    # Tải từ folder ID tùy chỉnh
    python scripts/download_index_gdrive.py --folder_id "YOUR_FOLDER_ID"

    # Tải 1 file .zip (nếu đã đóng gói thành 1 file)
    python scripts/download_index_gdrive.py --file_id "YOUR_FILE_ID"
"""
import os
import zipfile
import argparse
from pathlib import Path

# === CẤU HÌNH MẶC ĐỊNH ===
DEFAULT_FOLDER_ID = "1FSFKamm_fwss0sI4wBnBOxOko1gIGMH9"
DEFAULT_DEST = "data/indexes"


def download_folder(folder_id: str, dest_dir: str) -> None:
    """
    Tải toàn bộ thư mục từ Google Drive về máy.

    Args:
        folder_id: ID của Folder trên Google Drive.
        dest_dir: Đường dẫn thư mục đích (sẽ được tạo nếu chưa tồn tại).
    """
    try:
        import gdown
    except ImportError:
        print("Chưa cài gdown. Đang cài...")
        os.system("pip install gdown -q")
        import gdown

    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"[INFO] Đang tải folder từ Google Drive...")
    print(f"[INFO] Folder ID : {folder_id}")
    print(f"[INFO] Đích      : {dest_dir}")

    gdown.download_folder(url=url, output=dest_dir, quiet=False, use_cookies=False)
    print(f"\n[OK] Tải xong! Dữ liệu đã lưu vào: {dest_dir}")


def download_file_and_extract(file_id: str, dest_dir: str) -> None:
    """
    Tải 1 file .zip từ Google Drive và giải nén vào thư mục đích.

    Args:
        file_id: ID của file trên Google Drive.
        dest_dir: Đường dẫn thư mục đích.
    """
    try:
        import gdown
    except ImportError:
        print("Chưa cài gdown. Đang cài...")
        os.system("pip install gdown -q")
        import gdown

    os.makedirs(dest_dir, exist_ok=True)
    download_path = "index_data.zip"
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"[INFO] Đang tải file zip từ Google Drive (ID: {file_id})...")
    gdown.download(url, download_path, quiet=False)

    if not os.path.exists(download_path):
        print("[ERROR] Tải thất bại. Kiểm tra lại File ID và quyền chia sẻ.")
        return

    print(f"[INFO] Đang giải nén vào {dest_dir}...")
    try:
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print("[OK] Giải nén xong!")
    except zipfile.BadZipFile:
        print("[ERROR] File tải về không phải định dạng zip hợp lệ.")
    finally:
        if os.path.exists(download_path):
            os.remove(download_path)
            print("[INFO] Đã xóa file zip tạm.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu index từ Google Drive về data/indexes/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--folder_id",
        type=str,
        default=DEFAULT_FOLDER_ID,
        help=f"Google Drive Folder ID (mặc định: {DEFAULT_FOLDER_ID})",
    )
    parser.add_argument(
        "--file_id",
        type=str,
        default=None,
        help="Google Drive File ID nếu index được đóng gói thành 1 file .zip",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_DEST,
        help=f"Thư mục đích (mặc định: {DEFAULT_DEST})",
    )

    args = parser.parse_args()

    if args.file_id:
        # Nếu người dùng cung cấp file_id cụ thể → tải file zip
        download_file_and_extract(args.file_id, args.dest)
    else:
        # Mặc định: tải toàn bộ folder
        download_folder(args.folder_id, args.dest)

