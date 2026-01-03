"""
Scanner control module for PhotoPipe.

Provides control of SANE-compatible scanners (including Epson FastFoto FF-680W)
using the python-sane library or scanimage command-line tool.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from photopipe.config import get_config


@dataclass
class ScannerDevice:
    """Represents a detected scanner device."""
    name: str
    vendor: str
    model: str
    device_type: str

    def __str__(self) -> str:
        return f"{self.vendor} {self.model} ({self.name})"


@dataclass
class ScanResult:
    """Result of a scan operation."""
    front_path: Path
    back_path: Optional[Path]
    sequence_num: int
    timestamp: datetime


def check_sane_installed() -> bool:
    """Check if SANE (scanimage) is installed."""
    return shutil.which("scanimage") is not None


def check_python_sane_available() -> bool:
    """Check if python-sane library is available."""
    try:
        import sane
        return True
    except ImportError:
        return False


def list_scanners() -> list[ScannerDevice]:
    """
    List all available SANE-compatible scanners.

    Returns:
        List of ScannerDevice objects
    """
    devices = []

    # Try python-sane first
    if check_python_sane_available():
        try:
            import sane
            sane.init()
            for device in sane.get_devices():
                devices.append(ScannerDevice(
                    name=device[0],
                    vendor=device[1],
                    model=device[2],
                    device_type=device[3],
                ))
            sane.exit()
            return devices
        except Exception:
            pass

    # Fall back to scanimage command
    if check_sane_installed():
        try:
            result = subprocess.run(
                ["scanimage", "-L"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse output like: device `epson2:libusb:001:005' is a Epson FF-680W sheetfed scanner
            for line in result.stdout.strip().split("\n"):
                if line.startswith("device"):
                    # Extract device name between backticks
                    start = line.find("`") + 1
                    end = line.find("'", start)
                    if start > 0 and end > start:
                        name = line[start:end]
                        # Extract description after "is a "
                        desc_start = line.find("is a ") + 5
                        if desc_start > 4:
                            desc = line[desc_start:]
                            parts = desc.split()
                            vendor = parts[0] if parts else "Unknown"
                            model = " ".join(parts[1:-2]) if len(parts) > 2 else "Unknown"
                            device_type = parts[-1] if parts else "scanner"
                            devices.append(ScannerDevice(
                                name=name,
                                vendor=vendor,
                                model=model,
                                device_type=device_type,
                            ))
        except Exception:
            pass

    return devices


def find_fastfoto() -> Optional[ScannerDevice]:
    """
    Find an Epson FastFoto scanner.

    Returns:
        ScannerDevice if found, None otherwise
    """
    devices = list_scanners()
    for device in devices:
        if "fastfoto" in device.model.lower() or "ff-680" in device.model.lower():
            return device
        if "epson" in device.vendor.lower() and "ff" in device.model.lower():
            return device
    return None


class Scanner:
    """
    Scanner control interface.

    Supports both python-sane library and scanimage command-line tool.
    """

    def __init__(self, device_name: Optional[str] = None):
        """
        Initialize scanner interface.

        Args:
            device_name: SANE device name. If None, auto-detects FastFoto.
        """
        self.device_name = device_name
        self._sane_device = None
        self._use_python_sane = check_python_sane_available()

        if not self.device_name:
            # Try to auto-detect FastFoto
            fastfoto = find_fastfoto()
            if fastfoto:
                self.device_name = fastfoto.name

    def is_available(self) -> bool:
        """Check if the scanner is available."""
        if not self.device_name:
            return False

        if self._use_python_sane:
            try:
                import sane
                sane.init()
                devices = [d[0] for d in sane.get_devices()]
                sane.exit()
                return self.device_name in devices
            except Exception:
                return False
        else:
            devices = list_scanners()
            return any(d.name == self.device_name for d in devices)

    def get_device_info(self) -> Optional[ScannerDevice]:
        """Get information about the configured scanner."""
        devices = list_scanners()
        for device in devices:
            if device.name == self.device_name:
                return device
        return None

    def scan_batch(
        self,
        output_folder: Path,
        name_prefix: str,
        start_sequence: int = 1,
        resolution: int = 600,
        duplex: bool = True,
        mode: str = "color",
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> list[ScanResult]:
        """
        Scan a batch of photos using the ADF (Automatic Document Feeder).

        Args:
            output_folder: Folder to save scanned images
            name_prefix: Prefix for output filenames
            start_sequence: Starting sequence number
            resolution: Scan resolution in DPI
            duplex: Enable duplex (front and back) scanning
            mode: Color mode (color, gray, lineart)
            progress_callback: Optional callback(count, total) for progress updates

        Returns:
            List of ScanResult objects for each scanned photo
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        if self._use_python_sane:
            return self._scan_batch_python_sane(
                output_folder, name_prefix, start_sequence,
                resolution, duplex, mode, progress_callback
            )
        else:
            return self._scan_batch_scanimage(
                output_folder, name_prefix, start_sequence,
                resolution, duplex, mode, progress_callback
            )

    def _scan_batch_python_sane(
        self,
        output_folder: Path,
        name_prefix: str,
        start_sequence: int,
        resolution: int,
        duplex: bool,
        mode: str,
        progress_callback: Optional[Callable[[int, Optional[int]], None]],
    ) -> list[ScanResult]:
        """Scan using python-sane library."""
        import sane

        results = []
        sequence = start_sequence

        try:
            sane.init()
            device = sane.open(self.device_name)

            # Set scanner options
            try:
                device.resolution = resolution
            except Exception:
                pass

            try:
                device.mode = mode
            except Exception:
                pass

            try:
                device.source = "ADF" if duplex else "ADF Front"
            except Exception:
                pass

            # For duplex scanning on FastFoto
            if duplex:
                try:
                    device.adf_mode = "Duplex"
                except Exception:
                    pass

            # Scan until ADF is empty
            while True:
                try:
                    if progress_callback:
                        progress_callback(sequence - start_sequence + 1, None)

                    # Scan front
                    front_image = device.snap()
                    front_path = output_folder / f"{name_prefix}_{sequence:04d}.jpg"
                    front_image.save(str(front_path), "JPEG", quality=95)

                    back_path = None
                    if duplex:
                        try:
                            # Scan back (if duplex enabled)
                            back_image = device.snap()
                            back_path = output_folder / f"{name_prefix}_{sequence:04d}_b.jpg"
                            back_image.save(str(back_path), "JPEG", quality=95)
                        except Exception:
                            # No back side or duplex not supported
                            pass

                    results.append(ScanResult(
                        front_path=front_path,
                        back_path=back_path,
                        sequence_num=sequence,
                        timestamp=datetime.now(),
                    ))
                    sequence += 1

                except sane._sane.error as e:
                    if "no more documents" in str(e).lower() or "out of documents" in str(e).lower():
                        break
                    raise

            device.close()
            sane.exit()

        except Exception as e:
            try:
                sane.exit()
            except Exception:
                pass
            raise RuntimeError(f"Scanning failed: {e}")

        return results

    def _scan_batch_scanimage(
        self,
        output_folder: Path,
        name_prefix: str,
        start_sequence: int,
        resolution: int,
        duplex: bool,
        mode: str,
        progress_callback: Optional[Callable[[int, Optional[int]], None]],
    ) -> list[ScanResult]:
        """Scan using scanimage command-line tool."""
        results = []
        sequence = start_sequence

        # Build base scanimage command
        base_cmd = [
            "scanimage",
            "-d", self.device_name,
            "--resolution", str(resolution),
            "--mode", mode,
            "--format", "jpeg",
            "--batch",
            "--source", "ADF Duplex" if duplex else "ADF Front",
        ]

        # For batch scanning, scanimage outputs files with pattern
        # Use a temp directory and rename files afterwards
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Batch scan command
            batch_pattern = str(temp_path / "scan_%04d.jpg")
            cmd = base_cmd + [f"--batch={batch_pattern}"]

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Wait for completion
                stdout, stderr = process.communicate(timeout=600)

                if process.returncode != 0 and "out of documents" not in stderr.lower():
                    raise RuntimeError(f"scanimage failed: {stderr}")

            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError("Scanning timed out")

            # Process scanned files
            scanned_files = sorted(temp_path.glob("scan_*.jpg"))

            if duplex:
                # In duplex mode, files alternate: front, back, front, back...
                for i in range(0, len(scanned_files), 2):
                    front_temp = scanned_files[i]
                    back_temp = scanned_files[i + 1] if i + 1 < len(scanned_files) else None

                    front_path = output_folder / f"{name_prefix}_{sequence:04d}.jpg"
                    shutil.copy2(front_temp, front_path)

                    back_path = None
                    if back_temp:
                        back_path = output_folder / f"{name_prefix}_{sequence:04d}_b.jpg"
                        shutil.copy2(back_temp, back_path)

                    results.append(ScanResult(
                        front_path=front_path,
                        back_path=back_path,
                        sequence_num=sequence,
                        timestamp=datetime.now(),
                    ))

                    if progress_callback:
                        progress_callback(sequence - start_sequence + 1, None)

                    sequence += 1
            else:
                # Single-sided scanning
                for scan_file in scanned_files:
                    front_path = output_folder / f"{name_prefix}_{sequence:04d}.jpg"
                    shutil.copy2(scan_file, front_path)

                    results.append(ScanResult(
                        front_path=front_path,
                        back_path=None,
                        sequence_num=sequence,
                        timestamp=datetime.now(),
                    ))

                    if progress_callback:
                        progress_callback(sequence - start_sequence + 1, None)

                    sequence += 1

        return results

    def scan_single(
        self,
        output_path: Path,
        resolution: int = 600,
        duplex: bool = True,
        mode: str = "color",
    ) -> ScanResult:
        """
        Scan a single photo.

        Args:
            output_path: Path for the front image (back will be _b suffix)
            resolution: Scan resolution in DPI
            duplex: Enable duplex scanning
            mode: Color mode

        Returns:
            ScanResult for the scanned photo
        """
        output_folder = output_path.parent
        stem = output_path.stem
        sequence = 1

        results = self.scan_batch(
            output_folder=output_folder,
            name_prefix=stem,
            start_sequence=sequence,
            resolution=resolution,
            duplex=duplex,
            mode=mode,
        )

        if results:
            return results[0]
        raise RuntimeError("No photo was scanned")


def get_scanner() -> Scanner:
    """
    Get a Scanner instance using configuration settings.

    Returns:
        Configured Scanner instance
    """
    config = get_config()
    return Scanner(device_name=config.scanner.device)


def scan_photos_to_batch(
    batch_name: str,
    output_folder: Path,
    start_sequence: int = 1,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> list[ScanResult]:
    """
    Convenience function to scan photos for a batch.

    Uses configuration settings for scanner options.

    Args:
        batch_name: Name to use for output files
        output_folder: Folder to save scanned images
        start_sequence: Starting sequence number
        progress_callback: Optional progress callback

    Returns:
        List of ScanResult objects
    """
    config = get_config()
    scanner = get_scanner()

    return scanner.scan_batch(
        output_folder=output_folder,
        name_prefix=batch_name,
        start_sequence=start_sequence,
        resolution=config.scanner.resolution,
        duplex=config.scanner.duplex,
        mode=config.scanner.mode,
        progress_callback=progress_callback,
    )
