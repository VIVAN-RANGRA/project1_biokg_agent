"""Tests for biokg_agent.checkpoints"""
import pytest
from biokg_agent.checkpoints import (
    CheckpointStore,
    checkpoint_exists,
    load_json,
    load_pickle,
    save_json,
    save_pickle,
)


class TestSaveLoadJson:
    def test_round_trip_dict(self, tmp_path):
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_round_trip_list(self, tmp_path):
        data = [1, 2, 3, "four"]
        path = tmp_path / "test_list.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.json"
        save_json({"a": 1}, path)
        assert path.exists()


class TestSaveLoadPickle:
    def test_round_trip_dict(self, tmp_path):
        data = {"key": "value", "number": 42}
        path = tmp_path / "test.pkl"
        save_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded == data

    def test_round_trip_list(self, tmp_path):
        data = [1, 2.0, "three", None]
        path = tmp_path / "test_list.pkl"
        save_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded == data

    def test_round_trip_complex_object(self, tmp_path):
        data = {"nested": {"deep": [1, 2, 3]}, "tuple": (1, 2)}
        path = tmp_path / "complex.pkl"
        save_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded["nested"]["deep"] == [1, 2, 3]

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.pkl"
        save_pickle({"a": 1}, path)
        assert path.exists()


class TestCheckpointExists:
    def test_existing_file(self, tmp_path):
        path = tmp_path / "exists.json"
        save_json({}, path)
        assert checkpoint_exists(path) is True

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        assert checkpoint_exists(path) is False


class TestCheckpointStore:
    def test_creates_base_dir(self, tmp_path):
        store_dir = tmp_path / "store"
        store = CheckpointStore(store_dir)
        assert store_dir.exists()

    def test_path_constructs_correct_paths(self, tmp_path):
        store = CheckpointStore(tmp_path)
        p = store.path("subdir", "file.json")
        assert str(p) == str(tmp_path / "subdir" / "file.json")

    def test_save_json(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.save_json({"x": 1}, "test.json")
        loaded = store.load_json("test.json")
        assert loaded == {"x": 1}

    def test_save_pickle(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.save_pickle({"y": 2}, "test.pkl")
        loaded = store.load_pickle("test.pkl")
        assert loaded == {"y": 2}

    def test_save_text(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.save_text("hello world", "test.txt")
        content = (tmp_path / "test.txt").read_text(encoding="utf-8")
        assert content == "hello world"

    def test_saves_to_correct_directory(self, tmp_path):
        store = CheckpointStore(tmp_path)
        result_path = store.save_json({"z": 3}, "out.json")
        assert result_path.parent == tmp_path
