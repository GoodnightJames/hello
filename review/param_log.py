"""
Parameter Version Logger — tracks every strategy config change.

Every time a strategy's YAML config is loaded or modified, a snapshot
is saved to the param_versions table with:
- Full parameter dump (JSON)
- Version string
- Change reason (manual annotation)
- Effective date

This enables auditing: "what parameters were active on date X?"
"""

import json
import hashlib
from datetime import datetime

import yaml

from core.logging import get_logger
from data.db import ParamVersion, get_session, init_db

logger = get_logger("review.param_log")


def compute_config_hash(config_dict):
    """Compute a deterministic hash of a config dict for change detection."""
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def get_latest_param_version(session, strategy_name):
    """Get the most recently logged parameter version for a strategy."""
    return (
        session.query(ParamVersion)
        .filter(ParamVersion.strategy == strategy_name)
        .order_by(ParamVersion.effective_date.desc())
        .first()
    )


def log_param_version(strategy_name, config_dict, change_reason=None, session=None):
    """
    Log a strategy parameter version to the database.

    Only logs if the config has actually changed (hash comparison).

    Args:
        strategy_name: Strategy identifier.
        config_dict: Full strategy configuration dict.
        change_reason: Human-readable reason for the change.
        session: DB session (created if not provided).

    Returns:
        ParamVersion record if logged, None if unchanged.
    """
    close_session = False
    if session is None:
        init_db()
        session = get_session()
        close_session = True

    try:
        config_hash = compute_config_hash(config_dict)
        version = config_dict.get("version", "unknown")

        # Check if config has changed
        latest = get_latest_param_version(session, strategy_name)
        if latest:
            existing_config = json.loads(latest.params_json)
            existing_hash = compute_config_hash(existing_config)
            if existing_hash == config_hash:
                logger.info(
                    f"Config unchanged for {strategy_name}",
                    extra={"extra_data": {"hash": config_hash, "version": version}},
                )
                return None

        # Log the new version
        record = ParamVersion(
            strategy=strategy_name,
            version=version,
            params_json=json.dumps(config_dict, indent=2),
            change_reason=change_reason or "Config loaded",
            effective_date=datetime.utcnow(),
        )
        session.add(record)
        session.commit()

        logger.info(
            f"Parameter version logged for {strategy_name}",
            extra={
                "extra_data": {
                    "strategy": strategy_name,
                    "version": version,
                    "hash": config_hash,
                    "reason": change_reason,
                }
            },
        )
        return record

    finally:
        if close_session:
            session.close()


def log_config_from_file(config_path, change_reason=None, session=None):
    """
    Load a YAML config file and log its version.

    Args:
        config_path: Path to strategy YAML config.
        change_reason: Reason for the change.
        session: DB session.

    Returns:
        ParamVersion record if logged, None if unchanged.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    strategy_name = config.get("name", config_path)
    return log_param_version(strategy_name, config, change_reason, session)


def get_param_history(strategy_name, limit=20, session=None):
    """
    Get the parameter change history for a strategy.

    Args:
        strategy_name: Strategy identifier.
        limit: Max records to return.
        session: DB session.

    Returns:
        List of dicts with version info.
    """
    close_session = False
    if session is None:
        init_db()
        session = get_session()
        close_session = True

    try:
        records = (
            session.query(ParamVersion)
            .filter(ParamVersion.strategy == strategy_name)
            .order_by(ParamVersion.effective_date.desc())
            .limit(limit)
            .all()
        )

        history = []
        for r in records:
            history.append({
                "id": r.id,
                "version": r.version,
                "change_reason": r.change_reason,
                "effective_date": r.effective_date.isoformat(),
                "config_hash": compute_config_hash(json.loads(r.params_json)),
            })

        return history

    finally:
        if close_session:
            session.close()


def get_params_at_date(strategy_name, target_date, session=None):
    """
    Get the strategy parameters that were active on a specific date.

    Args:
        strategy_name: Strategy identifier.
        target_date: datetime to query.
        session: DB session.

    Returns:
        Config dict that was active on that date, or None.
    """
    close_session = False
    if session is None:
        init_db()
        session = get_session()
        close_session = True

    try:
        record = (
            session.query(ParamVersion)
            .filter(
                ParamVersion.strategy == strategy_name,
                ParamVersion.effective_date <= target_date,
            )
            .order_by(ParamVersion.effective_date.desc())
            .first()
        )

        if record is None:
            return None

        return json.loads(record.params_json)

    finally:
        if close_session:
            session.close()
