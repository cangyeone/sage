"""Validates literature_corpus_entry.schema.json self-consistency and
round-trips a known-good example entry against it."""
from pathlib import Path
import json
import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "shared/contracts/passport/literature_corpus_entry.schema.json"


def _load_schema():
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validator(schema):
    """Validator with format_checker so format-keyword constraints
    (e.g. date-time on obtained_at) are actually enforced."""
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def test_schema_file_exists():
    assert SCHEMA_PATH.exists(), f"schema missing at {SCHEMA_PATH}"


def test_schema_is_valid_draft_2020_12():
    schema = _load_schema()
    Draft202012Validator.check_schema(schema)


def test_required_set_matches_spec():
    schema = _load_schema()
    assert schema["required"] == [
        "citation_key",
        "title",
        "authors",
        "year",
        "source_pointer",
    ]


def test_additional_properties_is_false():
    schema = _load_schema()
    assert schema["additionalProperties"] is False


def test_valid_personal_author_entry_passes():
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024ai",
        "title": "AI assessment",
        "authors": [{"family": "Chen", "given": "Cindy"}],
        "year": 2024,
        "source_pointer": "https://doi.org/10.1234/xyz",
    }
    _validator(schema).validate(entry)


def test_valid_institution_author_entry_passes():
    schema = _load_schema()
    entry = {
        "citation_key": "who2024report",
        "title": "World report",
        "authors": [{"literal": "World Health Organization"}],
        "year": 2024,
        "source_pointer": "https://www.who.int/report",
    }
    _validator(schema).validate(entry)


def test_missing_required_field_fails():
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024ai",
        "title": "AI assessment",
        # missing authors
        "year": 2024,
        "source_pointer": "file:///x.pdf",
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_author_must_be_either_personal_or_literal():
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "bad2024",
        "title": "Bad author",
        "authors": [{}],  # neither family nor literal
        "year": 2024,
        "source_pointer": "file:///x.pdf",
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_year_out_of_range_fails():
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "old1",
        "title": "Ancient",
        "authors": [{"family": "X"}],
        "year": 999,
        "source_pointer": "file:///x.pdf",
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_citation_key_pattern_rejects_leading_digit():
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "2024chen",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_additional_property_fails():
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
        "custom_field": "should_not_be_allowed",
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_obtained_via_enum_constrained():
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
        "obtained_via": "rubber-duck",
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_obtained_via_other_without_adapter_name_fails():
    """Spec §obtained_via: 'Required when obtained_via=other'.
    Schema MUST enforce this conditionally, not via prose only."""
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
        "obtained_via": "other",
        # adapter_name missing — must fail
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_obtained_via_other_with_adapter_name_passes():
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
        "obtained_via": "other",
        "adapter_name": "my-custom-adapter",
    }
    _validator(schema).validate(entry)


def test_obtained_via_known_value_does_not_require_adapter_name():
    """Conditional only fires for 'other'. Reference adapters
    (zotero-bbt-export, obsidian-vault, folder-scan) must validate
    without adapter_name."""
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
        "obtained_via": "folder-scan",
    }
    _validator(schema).validate(entry)


def test_invalid_obtained_at_format_fails():
    """obtained_at: format=date-time must be enforced. Without
    format_checker the keyword is silently ignored."""
    from jsonschema.exceptions import ValidationError
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
        "obtained_at": "not-a-date",
    }
    with pytest.raises(ValidationError):
        _validator(schema).validate(entry)


def test_valid_obtained_at_format_passes():
    schema = _load_schema()
    entry = {
        "citation_key": "chen2024",
        "title": "T",
        "authors": [{"family": "C"}],
        "year": 2024,
        "source_pointer": "file:///x.pdf",
        "obtained_at": "2026-04-25T10:30:00Z",
    }
    _validator(schema).validate(entry)
