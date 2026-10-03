from dataforge_backend.infrastructure.superset import warehouse_uri


def test_the_bi_login_keeps_a_poolers_project_suffix_and_tls() -> None:
    uri = warehouse_uri(
        "postgresql://postgres.abcproject:secret@aws-1-eu-central-1.pooler.supabase.com:5432/postgres?ssl=require",
        "dataforge_bi",
        "p@ss/word",
    )
    assert uri == (
        "postgresql+psycopg2://dataforge_bi.abcproject:p%40ss%2Fword"
        "@aws-1-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require"
    )


def test_a_plain_local_database_is_unchanged() -> None:
    assert (
        warehouse_uri("postgresql://postgres:root@localhost:5432/dataforge", "dataforge_bi", "x")
        == "postgresql+psycopg2://dataforge_bi:x@localhost:5432/dataforge"
    )
