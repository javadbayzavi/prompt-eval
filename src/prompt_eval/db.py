from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import event, inspect, text
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from prompt_eval import models  # noqa: F401  (registers tables on SQLModel.metadata)


def make_engine(database_url: str) -> Engine:
    connect_args = {}
    is_sqlite = database_url.startswith("sqlite:///")
    if is_sqlite:
        # FastAPI may touch a session from different threads.
        connect_args["check_same_thread"] = False
        db_path = database_url.removeprefix("sqlite:///")
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, connect_args=connect_args)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
    add_missing_columns(engine)


def _default_literal(column) -> str | None:
    """SQL literal for a new column's DEFAULT, so existing rows get a value."""
    default = column.default
    value = default.arg if default is not None and default.is_scalar else None
    if value is None:
        if column.nullable:
            return None
        python_type = getattr(column.type, "python_type", str)
        value = {str: "", int: 0, float: 0.0, bool: False}.get(python_type, "")
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def add_missing_columns(engine: Engine) -> list[str]:
    """Additive-only migration: add columns that the models have and the tables lack.

    `create_all` creates missing tables but never alters existing ones, so without this an
    older database would break whenever a model gains a field. New fields must be nullable or
    have a scalar default. Renames and removals are not handled.
    """
    added = []
    existing_tables = set(inspect(engine).get_table_names())
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {col["name"] for col in inspect(conn).get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" '
                ddl += column.type.compile(dialect=engine.dialect)
                if (literal := _default_literal(column)) is not None:
                    ddl += (
                        f" NOT NULL DEFAULT {literal}"
                        if not column.nullable
                        else f" DEFAULT {literal}"
                    )
                conn.execute(text(ddl))
                added.append(f"{table.name}.{column.name}")
    return added


def session_scope(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session
