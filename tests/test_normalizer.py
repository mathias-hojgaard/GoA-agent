"""Tests for the value normalizer."""

from src.resolution.normalizer import normalize_value


class TestDateNormalization:
    def test_eu_date(self):
        assert normalize_value("15.03.2024", "order_date", "DD.MM.YYYY", ",") == "2024-03-15"

    def test_us_date(self):
        assert normalize_value("03/15/2024", "order_date", "MM/DD/YYYY", ".") == "2024-03-15"

    def test_date_field_detection(self):
        """Any field containing 'date' should be date-normalized."""
        assert normalize_value("15.03.2024", "delivery_date", "DD.MM.YYYY", ",") == "2024-03-15"

    def test_non_date_field_not_normalized(self):
        """Fields without 'date' in the name should not be date-parsed."""
        assert normalize_value("15.03.2024", "order_number", "DD.MM.YYYY", ",") == "15.03.2024"

    def test_unparseable_date_passthrough(self):
        assert normalize_value("TBD", "order_date", "DD.MM.YYYY", ",") == "TBD"

    def test_suffix_date_field(self):
        """Fields ending with '_date' should be date-normalized."""
        assert normalize_value("15.03.2024", "expected_ship_date", "DD.MM.YYYY", ",") == "2024-03-15"

    def test_non_date_substring_not_normalized(self):
        """Fields containing 'date' as a substring (not suffix) should NOT be date-normalized."""
        # "mandate_reference" contains "date" but should not trigger date logic.
        assert normalize_value("MR-2024-001", "mandate_reference", "DD.MM.YYYY", ",") == "MR-2024-001"


class TestDecimalNormalization:
    def test_eu_decimal(self):
        assert normalize_value("12,50", "unit_price", "DD.MM.YYYY", ",") == "12.50"

    def test_eu_thousands(self):
        assert normalize_value("1.250,00", "unit_price", "DD.MM.YYYY", ",") == "1250.00"

    def test_us_decimal(self):
        assert normalize_value("12.50", "unit_price", "DD.MM.YYYY", ".") == "12.50"

    def test_us_thousands(self):
        assert normalize_value("1,250.00", "unit_price", "DD.MM.YYYY", ".") == "1250.00"

    def test_numeric_fields(self):
        """All known numeric fields should get decimal normalization."""
        for field in ("unit_price", "total_price", "quantity", "amount"):
            assert normalize_value("12,50", field, "DD.MM.YYYY", ",") == "12.50"


class TestQuantityNormalization:
    def test_strip_pcs(self):
        assert normalize_value("500 pcs", "quantity", "DD.MM.YYYY", ",") == "500"

    def test_strip_meters(self):
        assert normalize_value("10 m", "quantity", "DD.MM.YYYY", ",") == "10"

    def test_no_unit(self):
        assert normalize_value("500", "quantity", "DD.MM.YYYY", ",") == "500"

    def test_space_separated_thousands(self):
        """EU documents sometimes use space as thousands separator."""
        assert normalize_value("1 250 pcs", "quantity", "DD.MM.YYYY", ",") == "1250"

    def test_space_separated_thousands_no_unit(self):
        assert normalize_value("1 250", "quantity", "DD.MM.YYYY", ",") == "1250"


class TestWhitespace:
    def test_strip_and_collapse(self):
        assert normalize_value("  hello   world  ", "description", "DD.MM.YYYY", ",") == "hello world"

    def test_empty_string(self):
        assert normalize_value("", "description", "DD.MM.YYYY", ",") == ""


class TestPassthrough:
    def test_unknown_field(self):
        assert normalize_value("WG-4420-BLK", "product_number", "DD.MM.YYYY", ",") == "WG-4420-BLK"

    def test_text_field(self):
        assert normalize_value("Acme Corporation", "company_name", "DD.MM.YYYY", ",") == "Acme Corporation"
