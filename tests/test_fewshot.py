"""Tests for the few-shot store."""

from __future__ import annotations

import pytest

from src.knowledge.fewshot_store import FewShotStore


class TestFewShotStoreEmpty:
    def test_empty_store_returns_zero_similarity(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        assert store.find_similar("anything", "product_number") == 0.0

    def test_empty_store_returns_no_examples(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        assert store.get_few_shot_examples("product_number") == []


class TestFewShotStoreAddAndQuery:
    def test_add_confirmed_and_find_similar(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        store.add_confirmed("product_number", "WG-4420-BLK", "Acme tender page 5")
        score = store.find_similar("WG-4420-BLK", "product_number")
        assert score > 0.8

    def test_find_similar_different_field_returns_zero(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        store.add_confirmed("product_number", "WG-4420-BLK", "Acme tender page 5")
        score = store.find_similar("WG-4420-BLK", "quantity")
        assert score == 0.0

    def test_find_similar_partial_match(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        store.add_confirmed("product_number", "WG-4420-BLK", "context")
        score = store.find_similar("WG-4420", "product_number")
        assert 0.0 < score < 1.0

    def test_exact_match_score_is_one(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        store.add_confirmed("product_number", "SKU-123", "context")
        score = store.find_similar("SKU-123", "product_number")
        assert score == 1.0


class TestFewShotStorePersistence:
    def test_persist_and_reload(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        store.add_confirmed("product_number", "WG-4420-BLK", "Acme tender")
        store.add_confirmed("quantity", "100", "Acme tender line 1")

        # Reload from disk
        store2 = FewShotStore("test_tenant", store_dir=str(tmp_path))
        assert len(store2.entries) == 2
        assert store2.entries[0]["field_name"] == "product_number"
        assert store2.entries[1]["value"] == "100"

    def test_file_created_on_add(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        assert not store.path.exists()
        store.add_confirmed("field", "value", "context")
        assert store.path.exists()


class TestFewShotExamples:
    def test_get_few_shot_examples_returns_correct_count(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        for i in range(5):
            store.add_confirmed("product_number", f"SKU-{i}", f"context {i}")

        examples = store.get_few_shot_examples("product_number", top_k=3)
        assert len(examples) == 3

    def test_get_few_shot_examples_returns_most_recent(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        for i in range(5):
            store.add_confirmed("product_number", f"SKU-{i}", f"context {i}")

        examples = store.get_few_shot_examples("product_number", top_k=2)
        assert examples[0]["value"] == "SKU-3"
        assert examples[1]["value"] == "SKU-4"

    def test_get_few_shot_examples_filters_by_field(self, tmp_path):
        store = FewShotStore("test_tenant", store_dir=str(tmp_path))
        store.add_confirmed("product_number", "SKU-1", "ctx")
        store.add_confirmed("quantity", "100", "ctx")
        store.add_confirmed("product_number", "SKU-2", "ctx")

        examples = store.get_few_shot_examples("product_number")
        assert len(examples) == 2
        assert all(e["field_name"] == "product_number" for e in examples)
