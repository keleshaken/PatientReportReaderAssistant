import httpx
import pytest

from src.online_translation import GoogleWebTranslator, OnlineTranslationError, split_translation_text


def test_long_text_split_loses_no_characters():
    text = ("Diagnosis and medicines.\n" * 500) + "x" * 5000
    parts = split_translation_text(text)
    assert "".join(parts) == text
    assert all(len(part) <= 2400 for part in parts)


def test_translation_targets_malayalam_and_keeps_source():
    def respond(request):
        assert request.url.host == "translate.googleapis.com"
        assert request.url.params["sl"] == "en"
        assert request.url.params["tl"] == "ml"
        assert request.url.params["q"] == "Dose 500 mg."
        return httpx.Response(200, json=[[["ഡോസ് 500 mg.", "Dose 500 mg."]]])

    translator = GoogleWebTranslator(httpx.MockTransport(respond))
    try:
        result = translator.translate_page(2, "Dose 500 mg.")
        assert result.page == 2 and result.original == "Dose 500 mg."
        assert "ഡോസ്" in result.translated and result.numbers_match
    finally:
        translator.close()


def test_changed_dose_is_flagged():
    translator = GoogleWebTranslator(httpx.MockTransport(
        lambda _: httpx.Response(200, json=[[["ഡോസ് 650 mg.", "Dose 500 mg."]]])))
    assert not translator.translate_page(1, "Dose 500 mg.").numbers_match
    translator.close()


@pytest.mark.parametrize("response", [httpx.Response(429), httpx.Response(200, json=[]),
                                      httpx.Response(200, text="not json")])
def test_service_errors_are_not_reported_as_missing_information(response):
    translator = GoogleWebTranslator(httpx.MockTransport(lambda _: response))
    with pytest.raises(OnlineTranslationError, match="unavailable or rate-limited"):
        translator.translate_page(1, "Diagnosis: viral fever.")
    translator.close()
