"""Validation of member and generated SQL before it reaches the sandbox.

A query must be a single read-only SELECT over tables the member is permitted to read. Personal columns
are refused, dangerous functions are refused, and row-level policies are applied by replacing each table
with a filtered subquery. The SQL that runs is regenerated from the validated syntax tree, so there is no
gap between what was checked and what executes.
"""

from dataclasses import dataclass, field
from typing import cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError

MAX_SQL_LENGTH = 40_000
DEFAULT_SEARCH_PATH = ("gold", "silver")

_WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Set,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Into,
    exp.Lock,
    exp.LoadData,
    exp.Grant,
    exp.TruncateTable,
    exp.Use,
    exp.Pragma,
)
# Typed functions sqlglot knows that are still unsafe in a shared sandbox.
_DENIED_FUNCS: tuple[type[exp.Expression], ...] = (exp.GenerateSeries, exp.ExplodingGenerateSeries)
# Functions sqlglot does not model (they parse as Anonymous) that are safe and useful for analysis.
_ALLOWED_ANONYMOUS = frozenset(
    """to_char to_date to_number date_part age justify_days justify_hours width_bucket regr_r2 regr_count
    regr_avgx regr_avgy regr_sxx regr_syy regr_sxy covar_pop covar_samp var_pop var_samp stddev_pop stddev_samp
    percentile_disc mode split_part btrim initcap lpad rpad translate strpos format_number bool_and bool_or
    every cume_dist percent_rank ntile nth_value first_value last_value lag lead row_number rank dense_rank
    make_date make_timestamp date_bin isfinite scale trunc sign cbrt degrees radians div gcd lcm
    string_agg array_agg jsonb_build_object json_build_object""".split()
)
_DENIED_PREFIXES = (
    "pg_",
    "lo_",
    "dblink",
    "current_setting",
    "set_config",
    "query_to_",
    "txid",
    "file",
    "inet_",
)


class QueryRefused(Exception):
    def __init__(self, reason: str, suggestion: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.suggestion = suggestion


@dataclass(frozen=True)
class TableGrant:
    table: str  # schema.name
    blocked_columns: frozenset[str] = field(default_factory=frozenset)
    row_policy: str | None = None  # an already-validated boolean expression over the table's columns


@dataclass(frozen=True)
class GuardedSql:
    sql: str
    tables: list[str]
    policies_applied: list[str]


def _parse_single(sql: str) -> exp.Expression:
    if not sql or not sql.strip():
        raise QueryRefused("The query is empty.", "Write a SELECT statement.")
    if len(sql) > MAX_SQL_LENGTH:
        raise QueryRefused("The query is too long.", f"Keep queries under {MAX_SQL_LENGTH} characters.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except ParseError as error:
        first = error.errors[0] if error.errors else {}
        where = f" near line {first.get('line')}" if first.get("line") else ""
        raise QueryRefused(f"The query could not be read{where}.", "Check the SQL syntax.") from error
    except SqlglotError as error:
        raise QueryRefused("The query could not be read.", "Check the SQL syntax.") from error
    if len(statements) != 1:
        raise QueryRefused("Run one statement at a time.", "Split the statements and run them one by one.")
    return cast(exp.Expression, statements[0])


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _check_functions(tree: exp.Expression) -> None:
    for node in tree.find_all(exp.Func):
        if isinstance(node, _DENIED_FUNCS):
            raise QueryRefused(
                f"{_function_name(node)} is not available in the sandbox.",
                "Use a table of permitted data instead.",
            )
        name = _function_name(node)
        if name.startswith(_DENIED_PREFIXES):
            raise QueryRefused(f"{name} is a system function and is not available in the sandbox.")
        if isinstance(node, exp.Anonymous) and name not in _ALLOWED_ANONYMOUS:
            raise QueryRefused(
                f"The function {name} is not on the sandbox's list of permitted functions.",
                "Use standard aggregate, date, text and window functions.",
            )


def guard_select(
    sql: str, grants: dict[str, TableGrant], search_path: tuple[str, ...] = DEFAULT_SEARCH_PATH
) -> GuardedSql:
    """Validate a query against the member's permitted tables. Returns the SQL to run, or raises QueryRefused."""
    tree = _parse_single(sql)
    if not isinstance(tree, exp.Select | exp.SetOperation):
        raise QueryRefused(
            "This identity can only read permitted objects. Write and administrative statements are blocked.",
            "Use SELECT to read data.",
        )
    for node in tree.walk():
        if isinstance(node, _WRITE_NODES):
            raise QueryRefused(
                "This identity can only read permitted objects. Write and administrative statements are blocked.",
                "Use SELECT to read data.",
            )
    _check_functions(tree)

    cte_names = {str(cte.alias_or_name).lower() for cte in tree.find_all(exp.CTE)}
    by_name: dict[str, list[str]] = {}
    for key in grants:
        schema, _, name = key.partition(".")
        by_name.setdefault(name.lower(), []).append(key)

    referenced: dict[int, str] = {}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise QueryRefused(
                "Table functions are not available in the sandbox.", "Read from a permitted table."
            )
        if table.args.get("catalog"):
            raise QueryRefused("Cross-database references are not allowed.")
        schema, name = str(table.db or "").lower(), str(table.name).lower()
        if not schema and name in cte_names:
            continue
        if schema:
            key = f"{schema}.{name}"
            if key not in grants:
                raise QueryRefused(
                    f"{key} is not one of the tables you are permitted to read.",
                    "Open the catalogue to see the approved tables you can query.",
                )
        else:
            candidates = [k for s in search_path for k in by_name.get(name, []) if k.startswith(f"{s}.")]
            if not candidates:
                raise QueryRefused(
                    f"{name} is not one of the tables you are permitted to read.",
                    "Qualify the table with its layer, for example gold.revenue_by_region.",
                )
            key = candidates[0]
        referenced[id(table)] = key

    blocked: dict[str, str] = {}
    for key in set(referenced.values()):
        for column in grants[key].blocked_columns:
            blocked[column.lower()] = key
    if blocked:
        for node in tree.find_all(exp.Column):
            if str(node.name).lower() in blocked:
                raise QueryRefused(
                    f"{node.name} is masked personal information and is not available to this role.",
                    "Remove the personal column, or count rows instead of listing them.",
                )
        for star in tree.find_all(exp.Star):
            parent = star.parent
            in_count = isinstance(parent, exp.Count) or (
                isinstance(parent, exp.Column) and isinstance(parent.parent, exp.Count)
            )
            if not in_count:
                raise QueryRefused(
                    f"SELECT * would include masked personal columns ({', '.join(sorted(blocked))}).",
                    "List the columns you need.",
                )

    applied: list[str] = []

    def rewrite(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Table) or id(node) not in referenced:
            return node
        key = referenced[id(node)]
        schema, _, name = key.partition(".")
        grant = grants[key]
        alias = node.alias or name
        if grant.row_policy:
            applied.append(key)
            inner = sqlglot.parse_one(
                f"SELECT * FROM {exp.to_identifier(schema).sql('postgres')}.{exp.to_identifier(name).sql('postgres')} "
                f"WHERE {grant.row_policy}",
                read="postgres",
            )
            return exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias)))
        qualified = exp.table_(name, db=schema, quoted=None)
        if node.alias:
            qualified.set("alias", exp.TableAlias(this=exp.to_identifier(node.alias)))
        return qualified

    # In place: the recorded node ids refer to this tree, and a copy would have new ones.
    rewritten = tree.transform(rewrite, copy=False)
    return GuardedSql(
        rewritten.sql(dialect="postgres"), sorted(set(referenced.values())), sorted(set(applied))
    )


_PREDICATE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Column,
    exp.Identifier,
    exp.Literal,
    exp.Boolean,
    exp.Null,
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.In,
    exp.Between,
    exp.Is,
    exp.And,
    exp.Or,
    exp.Not,
    exp.Paren,
    exp.Neg,
    exp.Cast,
    exp.DataType,
    exp.Like,
    exp.ILike,
    exp.Tuple,
    exp.Where,
    exp.Lower,
    exp.Upper,
)


def validate_predicate(predicate: str, columns: set[str]) -> str:
    """Check a row-policy or metric filter: comparisons over the table's own columns and literal values only."""
    if not predicate.strip():
        raise QueryRefused("The condition is empty.")
    try:
        tree = sqlglot.parse_one(f"SELECT 1 FROM t WHERE {predicate}", read="postgres")
    except SqlglotError as error:
        raise QueryRefused("The condition could not be read.", "Write it like: region = 'Gauteng'") from error
    where = tree.args.get("where")
    if not isinstance(tree, exp.Select) or where is None:
        raise QueryRefused("The condition could not be read.")
    lowered = {c.lower() for c in columns}
    for node in where.walk():
        if not isinstance(node, _PREDICATE_NODES):
            raise QueryRefused(
                f"Conditions may only compare columns with values ({type(node).__name__} is not allowed).",
                "Write it like: region IN ('Gauteng', 'Limpopo')",
            )
        if isinstance(node, exp.Column):
            if node.table or str(node.name).lower() not in lowered:
                raise QueryRefused(f"{node.sql()} is not a column of this table.")
    return str(where.this.sql(dialect="postgres"))
