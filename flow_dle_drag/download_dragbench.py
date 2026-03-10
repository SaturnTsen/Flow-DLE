import argparse
import logging
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ZIP_URL = "https://github.com/Yujun-Shi/DragDiffusion/releases/download/v0.1.1/DragBench.zip"
ZIP_FILENAME = ZIP_URL.split("/")[-1]


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download DragBench benchmark dataset"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='drag_bench_data',
        help='Directory to download the dataset (default: drag_bench_data)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if directory exists'
    )
    return parser.parse_args()


def download_file(url, filepath):
    """Download a file from URL with progress bar."""
    logger.info(f"Starting download: {filepath}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=filepath.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Download completed: {filepath}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory."""
    logger.info(f"Starting extraction: {zip_path.name} -> {extract_to}")
    
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Show progress for extraction
            file_list = zip_ref.namelist()
            for file in tqdm(file_list, desc="Extracting", unit="file"):
                zip_ref.extract(file, extract_to)
        
        logger.info(f"Extraction completed: {len(file_list)} files extracted")
        return True
        
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file: {e}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def cleanup(zip_path):
    """Remove the downloaded zip file after extraction."""
    if zip_path.exists():
        zip_path.unlink()
        logger.info(f"Cleaned up temporary file: {zip_path.name}")


def main():
    """Main execution flow."""
    logger.info("=" * 50)
    logger.info("Starting zip download and extraction process")
    logger.info("=" * 50)
    
    try:

        args = parse_arguments()
        download_dir = Path(args.output_dir)
        
        # Create directory if it doesn't exist
        if args.force and download_dir.exists():
            import shutil
            shutil.rmtree(download_dir)
            logger.info(f"Removed existing directory: {download_dir}")
        
        download_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Download directory: {download_dir.absolute()}")
        
        zip_path = download_dir / ZIP_FILENAME
        
        # Step 1: Download
        if not download_file(ZIP_URL, zip_path):
            logger.critical("Download failed. Aborting.")
            return
        
        # Step 2: Extract
        if not extract_zip(zip_path, download_dir):
            logger.critical("Extraction failed. Aborting.")
            return
        
        # Step 3: Cleanup (optional)
        # cleanup(zip_path)
        
        logger.info("=" * 50)
        logger.info(f"Process completed successfully!")
        logger.info(f"Data extracted to: {download_dir.absolute()}")
        logger.info("=" * 50)
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    main()