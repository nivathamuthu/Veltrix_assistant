from app.ingestion.parsers.tables import rows_to_text, split_stacked_row


def test_header_names_are_repeated_in_every_row():
    text = rows_to_text(
        [["Tenure", "Accrual"], ["0-2 years", "1.5 days/month"], ["3-5 years", "2 days/month"]],
        has_header=True,
    )
    assert text.splitlines() == [
        "Row: Tenure 0-2 years | Accrual 1.5 days/month",
        "Row: Tenure 3-5 years | Accrual 2 days/month",
    ]


def test_header_less_table_is_plain_cells():
    text = rows_to_text([["Document ID", "POL-X"], ["Owner", "People Team"]], has_header=False)
    assert text.splitlines() == ["Row: Document ID | POL-X", "Row: Owner | People Team"]


def test_wrapped_cell_text_is_collapsed_to_one_line():
    text = rows_to_text([["Topic", "Rule"], ["Parental leave,\nsecondary\n  caregiver", "2   weeks"]], True)
    assert text == "Row: Topic Parental leave, secondary caregiver | Rule 2 weeks"


def test_empty_cells_and_empty_rows_are_dropped():
    text = rows_to_text([["A", "B", "C"], ["x", "", "z"], ["", None, "  "], ["p", "q", ""]], True)
    assert text.splitlines() == ["Row: A x | C z", "Row: A p | B q"]


def test_none_cells_from_merged_ranges_do_not_crash():
    assert rows_to_text([["H1", "H2"], ["only", None]], True) == "Row: H1 only"


def test_header_only_table_is_not_treated_as_having_a_header():
    assert rows_to_text([["Tenure", "Accrual"]], has_header=True) == "Row: Tenure | Accrual"


def test_empty_table_gives_empty_text():
    assert rows_to_text([], True) == ""
    assert rows_to_text([["", ""], [None, None]], False) == ""


def test_row_longer_than_header_keeps_extra_cells():
    assert rows_to_text([["A"], ["x", "extra"]], True) == "Row: A x | extra"


def test_split_stacked_row_restores_one_row_per_line():
    stacked = [["Document ID\nVersion\nOwner", "POL-X\n2.0\nPeople Team"]]
    assert split_stacked_row(stacked) == [["Document ID", "POL-X"], ["Version", "2.0"], ["Owner", "People Team"]]


def test_split_stacked_row_leaves_normal_tables_alone():
    normal = [["Topic", "Rule"], ["Sick", "10 days"]]
    assert split_stacked_row(normal) is normal
    one_row = [["single", "row"]]
    assert split_stacked_row(one_row) is one_row
    uneven = [["a\nb", "c"]]  # cells disagree on line count: not a stacked box
    assert split_stacked_row(uneven) is uneven
