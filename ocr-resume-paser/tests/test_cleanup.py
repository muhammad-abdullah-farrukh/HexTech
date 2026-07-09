from resume_parser.cleanup import (
    clean_extraction,
    dedupe_near_identical_blocks,
    merge_split_sentences,
    normalize_whitespace,
    strip_repeated_boilerplate,
)


def test_strip_boilerplate_removes_repeated_header_across_pages():
    pages = [
        "ACME RESUME\nPage 1\nJohn Doe\nEngineer",
        "ACME RESUME\nPage 2\nExperience\nAcme Corp",
    ]
    out = strip_repeated_boilerplate(pages)
    assert all("ACME RESUME" not in p for p in out)
    # Page numbers normalize to the same token and get stripped too.
    assert all("Page" not in p for p in out)
    # Unique content survives.
    assert "John Doe" in out[0]
    assert "Acme Corp" in out[1]


def test_strip_boilerplate_noop_for_single_page():
    pages = ["only one page\nwith content"]
    assert strip_repeated_boilerplate(pages) == pages


def test_dedupe_near_identical_blocks():
    text = "Led the migration project.\n\nLed the migration project!\n\nDistinct block."
    out = dedupe_near_identical_blocks(text)
    assert out.count("Led the migration project") == 1
    assert "Distinct block." in out


def test_merge_split_sentences_rejoins_lowercase_continuation():
    text = "Built a service that handled\nmillions of requests per day"
    out = merge_split_sentences(text)
    assert "handled millions" in out


def test_merge_split_sentences_keeps_list_items_separate():
    text = "- first bullet\n- second bullet"
    out = merge_split_sentences(text)
    assert out == text


def test_normalize_whitespace_bullets_and_blank_lines():
    text = "• item\n\n\n\nnext"
    out = normalize_whitespace(text)
    assert "- item" in out
    assert "\n\n\n" not in out


def test_clean_extraction_end_to_end():
    pages = [
        "HEADER\nJohn Doe\n• Python developer with experience in\nbuilding APIs",
        "HEADER\nMore unique content here",
    ]
    out = clean_extraction(pages)
    assert "HEADER" not in out
    assert "- Python developer with experience in building APIs" in out
