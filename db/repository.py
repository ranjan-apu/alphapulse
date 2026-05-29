"""
Postgres Repository classes for AlphaPulse.
Provides typed CRUD methods and helpers.
"""
import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional

class BaseRepository:
    def __init__(self, cursor):
        self.cursor = cursor

    def _insert(self, table: str, data: Dict[str, Any]) -> None:
        """Helper to insert a dictionary into a table, converting dicts/lists to JSON strings where appropriate."""
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                processed_data[k] = json.dumps(v)
            else:
                processed_data[k] = v

        columns = list(processed_data.keys())
        values = [processed_data[col] for col in columns]
        placeholders = ", ".join(["%s"] * len(columns))
        col_str = ", ".join(columns)
        
        query = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"
        self.cursor.execute(query, values)

    def _update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> None:
        """Helper to update a table."""
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                processed_data[k] = json.dumps(v)
            else:
                processed_data[k] = v

        set_parts = []
        values = []
        for k, v in processed_data.items():
            set_parts.append(f"{k} = %s")
            values.append(v)
        
        where_parts = []
        for k, v in where.items():
            where_parts.append(f"{k} = %s")
            values.append(v)
            
        set_str = ", ".join(set_parts)
        where_str = " AND ".join(where_parts)
        
        query = f"UPDATE {table} SET {set_str} WHERE {where_str}"
        self.cursor.execute(query, values)


class RunRepository(BaseRepository):
    def save_run(self, data: Dict[str, Any]) -> None:
        self._insert("runs", data)
        
    def save_experiment_run(self, data: Dict[str, Any]) -> None:
        self._insert("experiment_runs", data)

    def update_experiment_run(self, run_id: str, data: Dict[str, Any]) -> None:
        self._update("experiment_runs", data, {"run_id": run_id})

    def get_experiment_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM experiment_runs WHERE run_id = %s", (run_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        # Convert row tuple to dict
        columns = [desc[0] for desc in self.cursor.description]
        res = dict(zip(columns, row))
        # Parse JSON
        for k in ("metrics", "config_snapshot"):
            if res.get(k) and isinstance(res[k], str):
                res[k] = json.loads(res[k])
        return res


class SnapshotRepository(BaseRepository):
    def save_snapshot_set(self, data: Dict[str, Any]) -> None:
        self._insert("data_snapshot_sets", data)

    def save_snapshot(self, data: Dict[str, Any]) -> None:
        self._insert("data_snapshots", data)

    def get_snapshot_set(self, set_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM data_snapshot_sets WHERE set_id = %s", (set_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def get_snapshots_for_set(self, set_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM data_snapshots WHERE set_id = %s", (set_id,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]


class PortfolioRepository(BaseRepository):
    def save_portfolio_snapshot(self, data: Dict[str, Any]) -> None:
        self._insert("portfolio_snapshots", data)

    def get_latest_snapshot(self, run_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM portfolio_snapshots WHERE run_id = %s ORDER BY timestamp DESC, created_at DESC LIMIT 1",
            (run_id,)
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))


class PositionRepository(BaseRepository):
    def save_position(self, data: Dict[str, Any]) -> None:
        self._insert("positions", data)

    def update_position(self, position_id: str, data: Dict[str, Any]) -> None:
        self._update("positions", data, {"position_id": position_id})

    def get_active_position(self, run_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM positions WHERE run_id = %s AND symbol = %s AND active = TRUE LIMIT 1",
            (run_id, symbol)
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM positions WHERE position_id = %s", (position_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))


class OrderRepository(BaseRepository):
    def save_order(self, data: Dict[str, Any]) -> None:
        self._insert("orders_simulated", data)

    def update_order(self, order_id: str, data: Dict[str, Any]) -> None:
        self._update("orders_simulated", data, {"order_id": order_id})

    def get_orders_for_position(self, position_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM orders_simulated WHERE position_id = %s", (position_id,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]


class DecisionRepository(BaseRepository):
    def save_decision(self, data: Dict[str, Any]) -> None:
        self._insert("decisions", data)

    def update_decision_outcome(self, decision_id: str, data: Dict[str, Any]) -> None:
        self._update("decisions", data, {"decision_id": decision_id})

    def get_decision_history(self, run_id: str, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM decisions WHERE run_id = %s AND symbol = %s ORDER BY decision_time DESC LIMIT %s",
            (run_id, symbol, limit)
        )
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            for k in ("checklist_json", "tool_calls_json", "memory_references", "reflection_ids", "raw_llm_responses"):
                if d.get(k) and isinstance(d[k], str):
                    d[k] = json.loads(d[k])
            res.append(d)
        return res


class SessionRepository(BaseRepository):
    # Session map
    def save_session_map(self, data: Dict[str, Any]) -> None:
        self._insert("session_maps", data)

    def update_session_map(self, session_id: str, data: Dict[str, Any]) -> None:
        self._update("session_maps", data, {"session_id": session_id})

    def get_session_map(self, session_id: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM session_maps WHERE session_id = %s", (session_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    def get_latest_session_map(self, run_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM session_maps WHERE run_id = %s AND symbol = %s ORDER BY session_date DESC, created_at DESC LIMIT 1",
            (run_id, symbol)
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))

    # Session levels
    def save_level(self, data: Dict[str, Any]) -> None:
        self._insert("session_levels", data)

    def update_level(self, level_id: str, data: Dict[str, Any]) -> None:
        self._update("session_levels", data, {"level_id": level_id})

    def get_active_levels(self, session_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM session_levels WHERE session_id = %s AND state IN ('ACTIVE', 'TESTED')",
            (session_id,)
        )
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_all_levels(self, session_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM session_levels WHERE session_id = %s", (session_id,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    # Session events
    def save_event(self, data: Dict[str, Any]) -> None:
        self._insert("session_events", data)

    def get_events_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM session_events WHERE session_id = %s ORDER BY event_time ASC", (session_id,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            if d.get("event_data") and isinstance(d["event_data"], str):
                d["event_data"] = json.loads(d["event_data"])
            res.append(d)
        return res


class MemoryRepository(BaseRepository):
    def save_episode(self, data: Dict[str, Any]) -> None:
        self._insert("memory_episodes", data)

    def get_episodes(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM memory_episodes WHERE symbol = %s ORDER BY created_at DESC LIMIT %s", (symbol, limit))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            for k in ("setup_tags", "thesis_json", "mistakes"):
                if d.get(k) and isinstance(d[k], str):
                    d[k] = json.loads(d[k])
            res.append(d)
        return res

    def save_reflection(self, data: Dict[str, Any]) -> None:
        self._insert("memory_reflections", data)

    def update_reflection(self, reflection_id: str, data: Dict[str, Any]) -> None:
        self._update("memory_reflections", data, {"reflection_id": reflection_id})

    def get_reflections(self, symbol: str, limit: int = 30) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM memory_reflections WHERE symbol = %s ORDER BY created_at DESC LIMIT %s", (symbol, limit))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            for k in ("tags", "source_episode_ids"):
                if d.get(k) and isinstance(d[k], str):
                    d[k] = json.loads(d[k])
            res.append(d)
        return res


class CalibrationRepository(BaseRepository):
    def save_calibration_stats(self, data: Dict[str, Any]) -> None:
        self._insert("calibration_stats", data)

    def update_calibration_stats(self, stat_id: str, data: Dict[str, Any]) -> None:
        self._update("calibration_stats", data, {"stat_id": stat_id})

    def get_stats_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute("SELECT * FROM calibration_stats WHERE run_id = %s", (run_id,))
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]


class TradeLockRepository(BaseRepository):
    def save_lock(self, data: Dict[str, Any]) -> None:
        self._insert("trade_locks", data)

    def get_active_locks(self, run_id: str, symbol: str, current_time: datetime) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM trade_locks WHERE run_id = %s AND symbol = %s AND expires_at > %s",
            (run_id, symbol, current_time)
        )
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in rows]


class AgentTurnRepository(BaseRepository):
    def save_turn(self, data: Dict[str, Any]) -> None:
        self._insert("agent_turn_records", data)

    def get_turns_for_decision(self, decision_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM agent_turn_records WHERE decision_id = %s ORDER BY turn_number ASC",
            (decision_id,)
        )
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            for k in ("schema_errors",):
                if d.get(k) and isinstance(d[k], str):
                    d[k] = json.loads(d[k])
            res.append(d)
        return res


class ToolTraceRepository(BaseRepository):
    def save_trace(self, data: Dict[str, Any]) -> None:
        self._insert("tool_call_traces", data)

    def get_traces_for_decision(self, decision_id: str) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM tool_call_traces WHERE decision_id = %s ORDER BY round_num ASC",
            (decision_id,)
        )
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            for k in ("arguments", "result"):
                if d.get(k) and isinstance(d[k], str):
                    d[k] = json.loads(d[k])
            res.append(d)
        return res


class AuditEventRepository(BaseRepository):
    def save_event(self, data: Dict[str, Any]) -> None:
        self._insert("audit_events", data)

    def get_events_for_run(self, run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM audit_events WHERE run_id = %s ORDER BY created_at DESC LIMIT %s",
            (run_id, limit)
        )
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            if d.get("details") and isinstance(d["details"], str):
                d["details"] = json.loads(d["details"])
            res.append(d)
        return res


class TradeEventRepository(BaseRepository):
    def save_event(self, data: Dict[str, Any]) -> None:
        self._insert("trade_events", data)

    def get_events_for_run(self, run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.cursor.execute(
            "SELECT * FROM trade_events WHERE run_id = %s ORDER BY created_at ASC LIMIT %s",
            (run_id, limit)
        )
        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]
        res = []
        for row in rows:
            d = dict(zip(columns, row))
            if d.get("details") and isinstance(d["details"], str):
                d["details"] = json.loads(d["details"])
            res.append(d)
        return res
