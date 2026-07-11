"""Scanner error classification: raw SANE stderr -> helper-friendly message."""
from photopipe.scanner import classify_scan_error


class TestClassifyScanError:
    def test_device_busy(self):
        msg = classify_scan_error("scanimage: open of device epsonds:net:... failed: Device busy")
        assert "busy" in msg.lower()
        assert "another computer or phone" in msg.lower()

    def test_device_io_error_is_busy(self):
        # epsonds emits this when another host holds the connection
        msg = classify_scan_error("scanimage: sane_start: Error during device I/O")
        assert "another computer or phone" in msg.lower()

    def test_unreachable(self):
        msg = classify_scan_error("scanimage: open of device epsonds:net:192.168.1.62 failed: Invalid argument")
        assert "reach the scanner" in msg.lower()

    def test_timeout_is_unreachable(self):
        msg = classify_scan_error("network read timed out")
        assert "reach the scanner" in msg.lower()

    def test_jam(self):
        msg = classify_scan_error("scanimage: sane_start: Document feeder jammed")
        assert "jam" in msg.lower()

    def test_unknown_passthrough(self):
        msg = classify_scan_error("something totally unexpected happened")
        assert "something totally unexpected happened" in msg

    def test_empty_gives_generic(self):
        msg = classify_scan_error("")
        assert msg  # non-empty, actionable
        assert "again" in msg.lower()
