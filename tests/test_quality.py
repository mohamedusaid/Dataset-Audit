from audit.quality import heuristic_quality, text_metrics


def test_whitespace_runs_and_comparisons_are_not_flagged():
    text = "def f():\n" + " " * 12 + "return a < b and c > d  # " + "word " * 10

    flags = heuristic_quality(text_metrics(text), token_count=50)

    assert "repeated_characters" not in flags
    assert "html_like" not in flags


def test_real_tags_and_character_runs_are_flagged():
    flags = heuristic_quality(text_metrics("<div class='x'>hello</div> " + "-" * 12), 50)

    assert {"html_like", "repeated_characters"} <= set(flags)
