"""Loop198: Authenticode Trusted Signer Guard Extension.

Safely downgrades malicious predictions to benign IF:
1. prediction == 1
2. auth_status == "Valid"
3. signer_subject contains a pre-declared trusted publisher term.
"""

from __future__ import annotations

from typing import List, Set, Tuple


FROZEN_TRUSTED_PUBLISHER_TERMS: Tuple[str, ...] = (
    "microsoft corporation",
    "microsoft windows",
    "microsoft azure",
    "google llc",
    "google inc",
    "nvidia corporation",
    "adobe inc",
    "adobe systems",
    "intel corporation",
    "cisco systems",
    "oracle america",
    "vmware, inc.",
    "realtek semiconductor",
    "logitech inc.",
)


class Loop198TrustedSignerGuard:
    """Authenticode trusted signer guard."""

    def __init__(self, trusted_terms: Tuple[str, ...] = FROZEN_TRUSTED_PUBLISHER_TERMS) -> None:
        self.trusted_terms = [t.lower() for t in trusted_terms]

    def is_trusted(self, auth_status: str, signer_subject: str) -> bool:
        if str(auth_status or "").strip().lower() != "valid":
            return False
        subject_clean = str(signer_subject or "").strip().lower()
        if not subject_clean:
            return False
        return any(term in subject_clean for term in self.trusted_terms)

    def evaluate_sample(
        self,
        prediction: int,
        auth_status: str,
        signer_subject: str,
    ) -> Tuple[int, bool]:
        """Returns (final_prediction, is_downgraded)."""
        if prediction == 1 and self.is_trusted(auth_status, signer_subject):
            return 0, True
        return prediction, False
