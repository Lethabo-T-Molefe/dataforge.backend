"""The pandas code recorded for each cleaning step, written from the step's actual parameters."""

import json
from typing import Any


def _list(values: Any) -> str:
    return json.dumps(list(values or []))


def code_for(operation: str, params: dict[str, Any]) -> str:
    p = params
    if operation == "accept_profile":
        return "# Profiling only measures. Nothing is changed.\nprofile = df.describe(include='all').T"
    if operation == "fill_missing":
        columns = _list(p.get("columns"))
        strategy = p.get("strategy", "constant")
        lines = [f"columns = {columns}"]
        if p.get("add_indicator"):
            lines.append("for c in columns:\n    df[f'{c}_was_missing'] = df[c].isna()")
        if strategy in ("constant", "unknown"):
            lines.append(f"df[columns] = df[columns].fillna({json.dumps(p.get('value', 'Unknown'))})")
        elif strategy in ("mean", "median"):
            lines.append(
                f"for c in columns:\n    x = pd.to_numeric(df[c], errors='coerce')\n    df[c] = x.fillna(x.{strategy}())"
            )
        elif strategy == "mode":
            lines.append("for c in columns:\n    df[c] = df[c].fillna(df[c].mode().iloc[0])")
        elif strategy == "group_median":
            lines.append(
                f"for c in columns:\n    x = pd.to_numeric(df[c], errors='coerce')\n"
                f"    df[c] = x.fillna(x.groupby(df[{json.dumps(p.get('group_by'))}]).transform('median'))"
            )
        elif strategy in ("ffill", "bfill"):
            order = f".sort_values({json.dumps(p.get('order_by'))})" if p.get("order_by") else ""
            group = f".groupby({json.dumps(p.get('group_by'))})" if p.get("group_by") else ""
            lines.append(f"df = df{order}\ndf[columns] = df{group}[columns].{strategy}()")
        elif strategy == "knn":
            lines.append(
                "from sklearn.impute import KNNImputer\nfrom sklearn.preprocessing import StandardScaler\n"
                "scaler = StandardScaler()\nx = scaler.fit_transform(df[columns])\n"
                f"df[columns] = scaler.inverse_transform(KNNImputer(n_neighbors={p.get('n_neighbors', 5)}).fit_transform(x))"
            )
        elif strategy == "iterative":
            lines.append(
                "from sklearn.experimental import enable_iterative_imputer  # noqa: F401\n"
                "from sklearn.impute import IterativeImputer\n"
                f"df[columns] = IterativeImputer(max_iter={p.get('max_iter', 10)}, random_state=0).fit_transform(df[columns])"
            )
        elif strategy == "missforest":
            lines.append(
                "from sklearn.ensemble import RandomForestRegressor\n"
                "from sklearn.experimental import enable_iterative_imputer  # noqa: F401\n"
                "from sklearn.impute import IterativeImputer\n"
                "imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=30, max_depth=12, random_state=0), max_iter=3, random_state=0)\n"
                "df[numeric_columns] = imputer.fit_transform(df[numeric_columns])\n"
                "# Categories are predicted with RandomForestClassifier from the other columns."
            )
        return "\n".join(lines)
    if operation == "drop_missing_rows":
        return f"df = df.dropna(subset={_list(p.get('columns'))}, how={json.dumps(p.get('how', 'any'))})"
    if operation == "drop_sparse_columns":
        return f"df = df.loc[:, df.isna().mean() <= {p.get('threshold', 0.5)}]"
    if operation == "deduplicate":
        subset = p.get("subset") or None
        order = f".sort_values({json.dumps(p.get('order_by'))})" if p.get("order_by") else ""
        return f"df = df{order}.drop_duplicates(subset={json.dumps(subset)}, keep={json.dumps(p.get('keep', 'first'))})"
    if operation == "link_records":
        return (
            f"# Block, then compare {_list(p.get('columns'))} with Jaro-Winkler (threshold {p.get('threshold', 0.92)}).\n"
            "# Pairs above the threshold are joined into clusters (connected components).\n"
            "df['record_cluster'] = cluster_ids\ndf['possible_duplicate'] = df['record_cluster'].duplicated(keep=False)"
        )
    if operation == "handle_outliers":
        action = p.get("action", "flag")
        body = [f"for c in {_list(p.get('columns'))}:", "    x = pd.to_numeric(df[c], errors='coerce')"]
        if action == "cap_percentile":
            body.append(
                f"    df[c] = x.clip(x.quantile({p.get('lower', 0.01)}), x.quantile({p.get('upper', 0.99)}))"
            )
        else:
            body += [
                "    q1, q3 = x.quantile(0.25), x.quantile(0.75)",
                "    mad = (x - x.median()).abs().median() or 1",
                "    extreme = (x < q1 - 3 * (q3 - q1)) | (x > q3 + 3 * (q3 - q1)) | (0.6745 * (x - x.median()).abs() / mad > 3.5)",
            ]
            body.append(
                {
                    "flag": "    df[f'{c}_outlier'] = extreme",
                    "null": "    df.loc[extreme, c] = None",
                    "drop": "    df = df[~extreme]",
                }.get(action, "    df[f'{c}_outlier'] = extreme")
            )
        return "\n".join(body)
    if operation == "isolation_forest":
        return (
            "from sklearn.ensemble import IsolationForest\n"
            f"x = df[{_list(p.get('columns'))}].apply(pd.to_numeric, errors='coerce')\n"
            f"model = IsolationForest(contamination={json.dumps(p.get('contamination', 'auto'))}, random_state=0)\n"
            "df['multivariate_outlier'] = model.fit_predict(x.fillna(x.median())) == -1"
        )
    if operation == "smooth_series":
        order = f".sort_values({json.dumps(p.get('order_by'))})" if p.get("order_by") else ""
        return f"df = df{order}\ndf[{json.dumps(p.get('column'))}] = df[{json.dumps(p.get('column'))}].rolling({p.get('window', 5)}, center=True, min_periods=1).median()"
    if operation == "standardise_formats":
        lines = [
            f"for c in {_list(p.get('dates'))}:",
            f"    df[c] = pd.to_datetime(df[c], dayfirst={p.get('day_first', True)}, errors='coerce').dt.date",
            f"for c in {_list(p.get('numbers'))}:",
            "    text = df[c].astype(str).str.replace(r'[R$\\s]', '', regex=True)",
            "    df[c] = pd.to_numeric(text.str.replace(',', '.', regex=False), errors='coerce')",
        ]
        return "\n".join(lines)
    if operation == "parse_dates":
        return f"for c in {_list(p.get('columns'))}:\n    df[c] = pd.to_datetime(df[c], dayfirst={p.get('day_first', True)}, errors='coerce').dt.date"
    if operation == "parse_numbers":
        return f"for c in {_list(p.get('columns'))}:\n    df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[R$\\s]', '', regex=True), errors='coerce')"
    if operation in ("normalize_text", "normalise_text_and_phones", "normalize_phones"):
        text = p.get("text_columns", p.get("columns") if operation == "normalize_text" else [])
        phones = p.get("phone_columns", p.get("columns") if operation == "normalize_phones" else [])
        return (
            "import unicodedata\n"
            f"for c in {_list(text)}:\n"
            "    df[c] = df[c].str.replace(r'\\s+', ' ', regex=True).str.strip().map(lambda s: unicodedata.normalize('NFC', s) if isinstance(s, str) else s)\n"
            f"for c in {_list(phones)}:\n"
            "    df[c] = df[c].str.replace(r'\\D', '', regex=True).str.replace(r'^0', '+27', regex=True)"
        )
    if operation == "coerce_types":
        converters = {
            "integer": "pd.to_numeric(df[c], errors='coerce').round().astype('Int64')",
            "decimal": "pd.to_numeric(df[c], errors='coerce')",
            "date": "pd.to_datetime(df[c], dayfirst=True, errors='coerce').dt.date",
            "boolean": "df[c].str.lower().map({'true': True, 'yes': True, '1': True, 'false': False, 'no': False, '0': False})",
            "string": "df[c].astype('string')",
        }
        lines = [
            "sentinels = {'', 'n/a', 'na', 'null', 'none', '-', '--', '?', 'nan'}",
            "df = df.mask(df.apply(lambda c: c.astype(str).str.strip().str.lower().isin(sentinels)))",
        ]
        for column, kind in dict(p.get("types") or {}).items():
            lines.append(f"c = {json.dumps(column)}\ndf[c] = {converters.get(kind, converters['string'])}")
        return "\n".join(lines)
    if operation == "flatten_json":
        column = json.dumps(p.get("column"))
        return f"extra = pd.json_normalize(df[{column}].map(json.loads)).add_prefix({column} + '_')\ndf = pd.concat([df.drop(columns=[{column}]), extra], axis=1)"
    if operation == "map_values":
        return f"df[{json.dumps(p.get('column'))}] = df[{json.dumps(p.get('column'))}].replace({json.dumps(p.get('mapping') or {})})"
    if operation == "cluster_labels":
        column = json.dumps(p.get("column"))
        return (
            "def fingerprint(s):\n"
            "    return ' '.join(sorted({t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in str(s)).split()}))\n"
            f"keys = df[{column}].map(fingerprint)\n"
            f"canonical = df.groupby(keys)[{column}].agg(lambda s: s.value_counts().index[0])\n"
            f"df[{column}] = keys.map(canonical)"
        )
    if operation == "drop_columns":
        return f"df = df.drop(columns={_list(p.get('columns'))}, errors='ignore')"
    if operation == "rename_columns":
        return f"df = df.rename(columns={json.dumps(p.get('mapping') or {})})"
    if operation == "tidy_reshape":
        return (
            f"df = df.melt(id_vars={_list(p.get('id_columns'))}, value_vars={_list(p.get('value_columns'))}, "
            f"var_name={json.dumps(p.get('var_name', 'variable'))}, value_name={json.dumps(p.get('value_name', 'value'))})"
        )
    if operation == "mask_columns":
        method = p.get("method", "redact")
        if method == "drop":
            return f"df = df.drop(columns={_list(p.get('columns'))})"
        fn = "pseudonymise" if method == "hash" else "mask"
        return f"for c in {_list(p.get('columns'))}:\n    df[c] = df[c].map({fn})  # applied server-side; raw values are never logged"
    if operation == "k_anonymity":
        quasi = _list(p.get("quasi_identifiers"))
        return f"sizes = df.groupby({quasi})[{quasi}[0]].transform('size')\ndf = df[sizes >= {p.get('k', 5)}]"
    if operation == "validate_rules":
        return (
            "# Rule pack (flag, do not delete):\n"
            + "\n".join(f"#   {json.dumps(rule)}" for rule in p.get("rules") or [])
            + "\ndf['row_failed_rules'] = ~passes_all_rules(df)"
        )
    if operation == "validate_sa_id":
        return f"df[{json.dumps(str(p.get('column')) + '_valid')}] = df[{json.dumps(p.get('column'))}].map(sa_id_checksum)  # computed server-side"
    if operation == "flag_orphans":
        return f"master = permitted_frame({json.dumps(p.get('reference'))})\ndf[{json.dumps(str(p.get('column')) + '_orphan')}] = ~df[{json.dumps(p.get('column'))}].isin(master)"
    if operation == "repair_constraints":
        det, dep = json.dumps(p.get("determinant")), json.dumps(p.get("dependent"))
        return (
            f"majority = df.groupby({det})[{dep}].agg(lambda s: s.value_counts().index[0])\n"
            f"share = df.groupby({det})[{dep}].agg(lambda s: s.value_counts(normalize=True).iloc[0])\n"
            f"fix = (df[{dep}] != df[{det}].map(majority)) & (df[{det}].map(share) >= {p.get('min_share', 0.8)})\n"
            f"df.loc[fix, {dep}] = df.loc[fix, {det}].map(majority)"
        )
    if operation == "reconcile_master":
        return f"master = permitted_frame({json.dumps(p.get('reference'))})\ndf[{json.dumps(p.get('column'))}] = match_to_master(df[{json.dumps(p.get('column'))}], master, threshold={p.get('threshold', 0.9)})"
    return f"# {operation}({json.dumps(p)})"
