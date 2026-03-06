"""Tests for the parameter version logger."""

import json
import os
from datetime import datetime, timedelta

import pytest

from data.db import init_db, get_session, Base, get_engine
from review.param_log import (
    compute_config_hash,
    log_param_version,
    get_param_history,
    get_params_at_date,
)


@pytest.fixture(autouse=True)
def clean_db():
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    engine = init_db("sqlite:///:memory:")
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(clean_db):
    return get_session(clean_db)


class TestConfigHash:
    def test_deterministic(self):
        config = {"a": 1, "b": 2}
        h1 = compute_config_hash(config)
        h2 = compute_config_hash(config)
        assert h1 == h2

    def test_order_independent(self):
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_configs_different_hash(self):
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        assert h1 != h2


class TestLogParamVersion:
    def test_logs_new_config(self, session):
        config = {"name": "test", "version": "1.0.0", "lookback": 252}
        result = log_param_version("test", config, "Initial", session)
        assert result is not None
        assert result.version == "1.0.0"

    def test_skips_unchanged_config(self, session):
        config = {"name": "test", "version": "1.0.0", "lookback": 252}
        log_param_version("test", config, "Initial", session)
        result = log_param_version("test", config, "Same config", session)
        assert result is None

    def test_logs_changed_config(self, session):
        config1 = {"name": "test", "version": "1.0.0", "lookback": 252}
        config2 = {"name": "test", "version": "1.0.1", "lookback": 200}
        log_param_version("test", config1, "Initial", session)
        result = log_param_version("test", config2, "Changed lookback", session)
        assert result is not None
        assert result.version == "1.0.1"


class TestGetParamHistory:
    def test_returns_ordered_history(self, session):
        for i in range(3):
            config = {"name": "test", "version": f"1.0.{i}", "val": i}
            log_param_version("test", config, f"Version {i}", session)

        history = get_param_history("test", session=session)
        assert len(history) == 3
        # Most recent first
        assert history[0]["version"] == "1.0.2"
        assert history[2]["version"] == "1.0.0"

    def test_empty_history(self, session):
        history = get_param_history("nonexistent", session=session)
        assert history == []


class TestGetParamsAtDate:
    def test_returns_correct_version(self, session):
        config = {"name": "test", "version": "1.0.0", "lookback": 252}
        log_param_version("test", config, "Initial", session)
        result = get_params_at_date("test", datetime.utcnow() + timedelta(days=1), session)
        assert result is not None
        assert result["version"] == "1.0.0"

    def test_returns_none_before_any_version(self, session):
        config = {"name": "test", "version": "1.0.0"}
        log_param_version("test", config, "Initial", session)
        result = get_params_at_date("test", datetime(2020, 1, 1), session)
        assert result is None
