import pytest

from quest1.inputs import InvalidInputError, build_job


def test_build_job_normalises_dialogue():
    job = build_job("https://ok.ru/video/248244667877", "  My mind\n rebels  at stagnation ")
    assert job.dialogue == "My mind rebels at stagnation"
    assert job.host == "ok.ru"


@pytest.mark.parametrize("bad", ["", "   ", "ok.ru/video/1", "ftp://ok.ru/v/1"])
def test_bad_urls_rejected(bad):
    with pytest.raises(InvalidInputError):
        build_job(bad, "hello")
