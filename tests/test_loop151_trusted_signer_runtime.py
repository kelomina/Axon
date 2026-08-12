from __future__ import annotations

from src.loop151_runtime.trusted_signer import TRUSTED_SIGNER_TERMS, apply_trusted_signer_guard


def test_frozen_loop151_guard_downgrades_only_a_valid_trusted_loop136_alert() -> None:
    decision = apply_trusted_signer_guard(
        loop136_prediction=1,
        authenticode_status="Valid",
        signer_subject="CN=Microsoft Corporation, O=Microsoft Corporation",
    )

    assert decision.prediction == 0
    assert decision.downgraded is True
    assert decision.matched_terms == ("Microsoft Corporation",)


def test_guard_never_downgrades_benign_invalid_or_untrusted_inputs() -> None:
    assert apply_trusted_signer_guard(loop136_prediction=0, authenticode_status="Valid", signer_subject="Microsoft Windows").prediction == 0
    assert apply_trusted_signer_guard(loop136_prediction=1, authenticode_status="NotSigned", signer_subject="Microsoft Windows").prediction == 1
    assert apply_trusted_signer_guard(loop136_prediction=1, authenticode_status="Valid", signer_subject="Unknown Publisher").prediction == 1


def test_frozen_term_list_matches_the_loop151_policy() -> None:
    assert TRUSTED_SIGNER_TERMS == (
        "Microsoft Corporation",
        "Microsoft Windows",
        "Seagate Technology",
        "FinalWire",
        "NetEase",
        "Beijing Sogou",
        "Beijing Kingsoft",
        "Beijing Qihu",
        "Wondershare",
        "IObit",
        "Yozosoft",
        "Huya",
    )
