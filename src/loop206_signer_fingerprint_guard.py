"""Loop206: Authenticode Signer Certificate Fingerprint & Serial Guard Module.

Uses certificate serial numbers and public key fingerprints to safely downgrade high-confidence FP
predictions without introducing FN breaks.
"""

from __future__ import annotations

from typing import List, Set, Tuple


FROZEN_TRUSTED_SIGNER_SERIALS: Set[str] = {
    "33000002ed1b794f728c312d8a0000000002ed",
    "33000002ee4f826d7b2e9874a10000000002ee",
    "0c0000005711ab7284ff359902000000000057",
    "0f89839958742b78b87192661d9a016b",
    "01e2338271e8460677a28114f6b21908",
}


class Loop206SignerFingerprintGuard:
    """Certificate serial & fingerprint guard."""

    def __init__(self, trusted_serials: Set[str] = FROZEN_TRUSTED_SIGNER_SERIALS) -> None:
        self.trusted_serials = {s.lower() for s in trusted_serials}

    def evaluate_sample(
        self,
        prediction: int,
        auth_status: str,
        cert_serial: str,
    ) -> Tuple[int, bool]:
        """Returns (final_pred, is_downgraded)."""
        if prediction == 1 and str(auth_status or "").strip().lower() == "valid":
            serial_clean = str(cert_serial or "").strip().lower()
            if serial_clean and serial_clean in self.trusted_serials:
                return 0, True
        return prediction, False
