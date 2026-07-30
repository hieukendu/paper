from vipragsent.data.clean_pii import clean_pii, detect_pii


def test_clean_pii_replaces_email_phone_handle_url_and_preserves_sarcasm_marker():
    text = "Lien he @user qua test@example.com hoac 0912 345 678 =)) xem www.example.com"
    cleaned = clean_pii(text)
    assert "[USER]" in cleaned
    assert "[EMAIL]" in cleaned
    assert "[PHONE]" in cleaned
    assert "[URL]" in cleaned
    assert "=))" in cleaned
    assert detect_pii(cleaned) == []
