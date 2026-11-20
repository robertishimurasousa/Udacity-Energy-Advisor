import os
import sqlite3
import tempfile
import unittest
from contextlib import chdir, closing
from pathlib import Path
from unittest.mock import patch

from tools import search_energy_tips


class GuideEmbeddings:
    """Use deterministic vectors in place of the Vocareum network call."""

    def __init__(self):
        self.indexed_texts = []

    def embed_documents(self, texts):
        self.indexed_texts.extend(texts)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "battery" in text.lower() else [0.0, 1.0]


class EnergyTipsTests(unittest.TestCase):
    def test_search_builds_and_reuses_sqlite_vectors_with_sources(self):
        embeddings = GuideEmbeddings()
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            guides = Path("data/documents")
            guides.mkdir(parents=True)
            (guides / "battery.txt").write_text("A battery holds backup energy.", encoding="utf-8")
            (guides / "hvac.txt").write_text("Clean the air filter regularly.", encoding="utf-8")
            legacy_index = Path("data/vectorstore")
            legacy_index.mkdir()
            legacy_file = legacy_index / "chroma.sqlite3"
            legacy_file.write_text("An existing local index", encoding="utf-8")

            with patch.dict(os.environ, {"VOCAREUM_API_KEY": "test-key"}):
                with patch("tools.OpenAIEmbeddings", return_value=embeddings):
                    result = search_energy_tips.invoke({"query": "battery reserve", "max_results": 1})
                    again = search_energy_tips.invoke({"query": "air filter", "max_results": 1})

            self.assertNotIn("error", result, result)
            self.assertNotIn("error", again, again)
            self.assertEqual(result["total_results"], 1)
            self.assertEqual(result["tips"][0]["source"], (guides / "battery.txt").as_posix())
            self.assertEqual(again["tips"][0]["source"], (guides / "hvac.txt").as_posix())
            self.assertEqual(len(embeddings.indexed_texts), 2)
            self.assertEqual(legacy_file.read_text(encoding="utf-8"), "An existing local index")

            index_path = Path("data/energy_tips.sqlite3")
            self.assertTrue(index_path.is_file(), "Search must persist in the new SQLite index")
            with closing(sqlite3.connect(index_path)) as connection:
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone(), ("ok",))
