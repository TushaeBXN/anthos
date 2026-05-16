"""Last-resort protections against model theft and unauthorized use"""
import hashlib
import time
import threading
from typing import Optional, List


class ModelProtection:
    """Prevent unauthorized use or modification of Anthos"""

    def __init__(self, secret_key: str = "TushaeBXN_Anthos_2026"):
        self.secret_key = secret_key
        self.authorized_hardware_ids: List[str] = []
        self._kill_switch_active = False
        self._monitor_thread: Optional[threading.Thread] = None

    def get_hardware_fingerprint(self) -> str:
        """Derive a fingerprint from available hardware info"""
        import platform
        import uuid
        hw_info = f"{platform.node()}:{platform.machine()}:{uuid.getnode()}"
        return hashlib.sha256(hw_info.encode()).hexdigest()[:16]

    def authorize_this_machine(self):
        """Register the current machine as authorized"""
        fp = self.get_hardware_fingerprint()
        if fp not in self.authorized_hardware_ids:
            self.authorized_hardware_ids.append(fp)
            print(f"Machine authorized: {fp}")

    def is_authorized(self) -> bool:
        """Check if running on an authorized machine"""
        if not self.authorized_hardware_ids:
            return True  # No restrictions set
        return self.get_hardware_fingerprint() in self.authorized_hardware_ids

    def enable_remote_kill_switch(self, check_url: str, interval_seconds: int = 3600):
        """Poll a URL for a kill signal"""
        def _check_loop():
            import urllib.request
            while not self._kill_switch_active:
                try:
                    with urllib.request.urlopen(check_url, timeout=10) as resp:
                        data = resp.read().decode()
                        if '"kill": true' in data or '"kill":true' in data:
                            print("Remote kill switch activated!")
                            self._kill_switch_active = True
                            break
                except Exception:
                    pass
                time.sleep(interval_seconds)

        self._monitor_thread = threading.Thread(target=_check_loop, daemon=True)
        self._monitor_thread.start()
        print(f"Remote kill switch monitoring started (polling {check_url} every {interval_seconds}s)")

    def verify_checkpoint_origin(self, checkpoint: dict, expected_creator: str = "TushaeBXN") -> bool:
        """Verify a checkpoint was created by the expected author"""
        metadata = checkpoint.get("metadata", {})
        signed_by = metadata.get("signed_by", "")
        if signed_by != expected_creator:
            raise RuntimeError(
                f"Checkpoint creator mismatch: expected '{expected_creator}', got '{signed_by}'"
            )
        return True
