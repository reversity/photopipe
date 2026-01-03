"""
File system watcher for PhotoPipe.

Uses watchdog to monitor scanner output folder for new scans.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Callable
from queue import Queue

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from photopipe.config import get_config


# Common image extensions to watch for
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


class ScannerWatchHandler(FileSystemEventHandler):
    """Handler for new scanner files."""

    def __init__(
        self,
        callback: Callable[[Path], None],
        stable_time: float = 2.0,
    ):
        """
        Initialize the watch handler.

        Args:
            callback: Function to call when a new file is stable
            stable_time: Time in seconds to wait for file to stabilize
        """
        super().__init__()
        self.callback = callback
        self.stable_time = stable_time
        self._pending_files: dict[Path, float] = {}
        self._lock = threading.Lock()
        self._checker_thread: Optional[threading.Thread] = None
        self._running = False

    def start_checker(self):
        """Start the background file stability checker."""
        self._running = True
        self._checker_thread = threading.Thread(target=self._check_stability, daemon=True)
        self._checker_thread.start()

    def stop_checker(self):
        """Stop the background file stability checker."""
        self._running = False
        if self._checker_thread:
            self._checker_thread.join(timeout=2)

    def _check_stability(self):
        """Background thread to check file stability."""
        while self._running:
            time.sleep(0.5)

            with self._lock:
                now = time.time()
                stable_files = []

                for path, first_seen in list(self._pending_files.items()):
                    if now - first_seen >= self.stable_time:
                        # File has been stable long enough
                        if path.exists():
                            stable_files.append(path)
                        del self._pending_files[path]

            # Process stable files outside the lock
            for path in stable_files:
                try:
                    self.callback(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

    def on_created(self, event):
        """Handle new file creation."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Check if it's an image file
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return

        with self._lock:
            self._pending_files[path] = time.time()

    def on_modified(self, event):
        """Handle file modification (reset stability timer)."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        with self._lock:
            if path in self._pending_files:
                # Reset the timer as file is still being written
                self._pending_files[path] = time.time()


class FolderWatcher:
    """
    Watch a folder for new scanner files.

    Provides both callback-based and queue-based interfaces.
    """

    def __init__(
        self,
        watch_folder: Optional[Path] = None,
        callback: Optional[Callable[[Path], None]] = None,
    ):
        """
        Initialize the folder watcher.

        Args:
            watch_folder: Folder to watch (from config if not provided)
            callback: Optional callback for new files
        """
        config = get_config()
        self.watch_folder = watch_folder or config.paths.input_folder
        self.callback = callback
        self.file_queue: Queue[Path] = Queue()

        # Set up watchdog
        self._observer: Optional[Observer] = None
        self._handler: Optional[ScannerWatchHandler] = None
        self._running = False

    def _on_new_file(self, path: Path):
        """Internal callback for new files."""
        # Add to queue
        self.file_queue.put(path)

        # Call user callback if provided
        if self.callback:
            self.callback(path)

    def start(self):
        """Start watching the folder."""
        if self._running:
            return

        # Ensure folder exists
        self.watch_folder.mkdir(parents=True, exist_ok=True)

        # Create handler and observer
        config = get_config()
        self._handler = ScannerWatchHandler(
            callback=self._on_new_file,
            stable_time=config.scanner.watch_interval_seconds,
        )
        self._handler.start_checker()

        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.watch_folder), recursive=False)
        self._observer.start()
        self._running = True

    def stop(self):
        """Stop watching the folder."""
        if not self._running:
            return

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

        if self._handler:
            self._handler.stop_checker()

        self._running = False

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def get_pending_count(self) -> int:
        """Get number of files pending in queue."""
        return self.file_queue.qsize()

    def get_next_file(self, timeout: Optional[float] = None) -> Optional[Path]:
        """
        Get next file from queue.

        Args:
            timeout: Maximum time to wait (None = non-blocking)

        Returns:
            Path to new file, or None if queue empty/timeout
        """
        try:
            return self.file_queue.get(block=timeout is not None, timeout=timeout)
        except Exception:
            return None

    def get_all_pending(self) -> list[Path]:
        """
        Get all pending files from queue.

        Returns:
            List of file paths
        """
        files = []
        while not self.file_queue.empty():
            try:
                files.append(self.file_queue.get_nowait())
            except Exception:
                break
        return files

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class BatchWatcher:
    """
    High-level watcher that integrates with batch processing.

    Monitors a folder and automatically ingests new photos into a batch.
    """

    def __init__(
        self,
        watch_folder: Path,
        batch_id: str,
        db,  # Database instance
        on_new_photo: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize batch watcher.

        Args:
            watch_folder: Folder to watch
            batch_id: Batch to add photos to
            db: Database instance
            on_new_photo: Optional callback when photo is ingested
        """
        self.watch_folder = watch_folder
        self.batch_id = batch_id
        self.db = db
        self.on_new_photo = on_new_photo

        self._watcher = FolderWatcher(
            watch_folder=watch_folder,
            callback=self._handle_new_file,
        )
        self._pending_fronts: dict[int, Path] = {}
        self._lock = threading.Lock()

    def _handle_new_file(self, path: Path):
        """Handle new file from watcher."""
        from photopipe.pairing import (
            extract_sequence_number,
            is_back_image,
        )
        from photopipe.models import PhotoPair
        from photopipe.config import get_config

        config = get_config()
        front_pattern = config.scanner.front_pattern.replace(".jpg", "").replace(".JPG", "")
        back_pattern = config.scanner.back_pattern.replace(".jpg", "").replace(".JPG", "")

        filename_stem = path.stem

        with self._lock:
            if is_back_image(filename_stem, back_pattern):
                # This is a back image - find matching front
                seq_num = extract_sequence_number(
                    filename_stem.replace("_back", "").replace("_BACK", ""),
                    front_pattern,
                )

                if seq_num and seq_num in self._pending_fronts:
                    # We have the front, create pair
                    front_path = self._pending_fronts.pop(seq_num)
                    self._create_photo_pair(front_path, path)
            else:
                # This is a front image
                seq_num = extract_sequence_number(filename_stem, front_pattern)

                if seq_num:
                    # Wait briefly for potential back
                    self._pending_fronts[seq_num] = path

                    # Schedule check for orphaned front after delay
                    threading.Timer(
                        5.0,  # Wait 5 seconds for back
                        self._check_orphaned_front,
                        args=[seq_num],
                    ).start()

    def _check_orphaned_front(self, seq_num: int):
        """Check if front is still waiting for back."""
        with self._lock:
            if seq_num in self._pending_fronts:
                # No back came, create pair without it
                front_path = self._pending_fronts.pop(seq_num)
                self._create_photo_pair(front_path, None)

    def _create_photo_pair(self, front_path: Path, back_path: Optional[Path]):
        """Create a new photo pair in the database."""
        from photopipe.models import PhotoPair

        # Check if already exists
        if self.db.check_photo_exists(front_path):
            return

        photo = PhotoPair(
            batch_id=self.batch_id,
            sequence_num=self.db.get_next_sequence_num(self.batch_id),
            front_path=front_path,
            back_path=back_path,
        )

        self.db.create_photo(photo)
        self.db.log_action(
            photo_id=photo.id,
            batch_id=self.batch_id,
            action="auto_ingested",
            details={
                "front_path": str(front_path),
                "back_path": str(back_path) if back_path else None,
            },
        )

        if self.on_new_photo:
            self.on_new_photo(photo.id)

    def start(self):
        """Start watching."""
        self._watcher.start()

    def stop(self):
        """Stop watching."""
        self._watcher.stop()

        # Process any remaining pending fronts
        with self._lock:
            for seq_num, front_path in list(self._pending_fronts.items()):
                self._create_photo_pair(front_path, None)
            self._pending_fronts.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
