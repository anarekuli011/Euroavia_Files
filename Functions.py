# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:28:35 2026

@author: aprash
"""

import matplotlib.pyplot as plt
import cv2
import pandas as pd

def show_img(path, x=20, y=12):
    image = cv2.imread(path)
    
    if image is None:
        print(f"Error: Could not load image at {path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(x, y)) 
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def parse_ids(val):
    if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
        return []
    out = []
    for token in str(val).strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = a.strip(), b.strip()
            if a.isdigit() and b.isdigit():
                out.extend([str(i) for i in range(int(a), int(b) + 1)])
            else:
                out.append(token)
        else:
            out.append(token)
    return out   

def filter_by_ids(df, ids, id_col="ID"):
    ids = {str(x) for x in ids if str(x).strip() != ""}
    if not ids:
        return df.iloc[0:0].copy()
    return df[df[id_col].astype(str).isin(ids)].copy()

def build_airport_chain(df_airports, df_runways, df_lds, df_functions, df_segments, df_ccrcircuits):
    """
    Build a chain:
      Airport -> Runway -> (LD -> Function OR Runway-direct Function) -> Segment -> Circuit

    Runway-direct functions are included only if they are NOT already referenced by any LD on that runway.

    Returns: pd.DataFrame
    """

    # fast lookups by ID as string
    df_runways_i     = df_runways.assign(ID=df_runways["ID"].astype(str)).set_index("ID", drop=False)
    df_lds_i         = df_lds.assign(ID=df_lds["ID"].astype(str)).set_index("ID", drop=False)
    df_functions_i   = df_functions.assign(ID=df_functions["ID"].astype(str)).set_index("ID", drop=False)
    df_segments_i    = df_segments.assign(ID=df_segments["ID"].astype(str)).set_index("ID", drop=False)
    df_ccrcircuits_i = df_ccrcircuits.assign(ID=df_ccrcircuits["ID"].astype(str)).set_index("ID", drop=False)

    chain = []

    for _, airport in df_airports.iterrows():
        airport_id = airport.get("ID")
        airport_name = airport.get("Name")

        # 1) Airport -> Runways
        runway_ids = parse_ids(airport.get("RunwayIDs", ""))

        for rwy_id in runway_ids:
            rwy_id = str(rwy_id)
            if rwy_id not in df_runways_i.index:
                continue

            rwy = df_runways_i.loc[rwy_id]
            rwy_name = rwy.get("Name")

            # 2) Runway -> LDs
            lds_ids = []
            for ld_key in ("LD1ID", "LD2ID"):
                ld_id = rwy.get(ld_key)
                if pd.notna(ld_id) and str(ld_id).strip():
                    lds_ids.append(str(ld_id).strip())

            # collect functions referenced by LDs
            ld_function_ids = set()
            ld_rows = []  # (ld_id, ld_name, ld_row)
            for ld_id in lds_ids:
                if ld_id not in df_lds_i.index:
                    continue
                ld = df_lds_i.loc[ld_id]
                ld_name = ld.get("Name")
                ld_rows.append((ld_id, ld_name, ld))

                for func_key in ("LandFunctionIDs", "TakeOffFunctionIDs"):
                    ld_function_ids.update(parse_ids(ld.get(func_key, "")))

            # Runway -> Functions (direct)
            runway_function_ids = set(parse_ids(rwy.get("FunctionIDs", "")))

            # Only runway-direct functions NOT already tied to any LD
            runway_direct_only_function_ids = runway_function_ids - ld_function_ids

            def add_function_chain(func_id, ld_id=None, ld_name=None):
                func_id = str(func_id)
                if func_id not in df_functions_i.index:
                    return

                func = df_functions_i.loc[func_id]
                func_name = func.get("Name")

                # Function -> Segments
                for seg_id in parse_ids(func.get("SegmentIDs", "")):
                    seg_id = str(seg_id)
                    if seg_id not in df_segments_i.index:
                        continue

                    seg = df_segments_i.loc[seg_id]
                    seg_name = seg.get("Name")

                    # Segment -> Circuits
                    for circ_id in parse_ids(seg.get("CCRCircuitIDs", "")):
                        circ_id = str(circ_id)
                        if circ_id not in df_ccrcircuits_i.index:
                            continue

                        circ = df_ccrcircuits_i.loc[circ_id]

                        chain.append({
                            "AirportID": airport_id,
                            "Airport": airport_name,
                            "RunwayID": rwy_id,
                            "Runway": rwy_name,
                            "LDID": ld_id,       # None for runway-direct path
                            "LD": ld_name,       # None for runway-direct path
                            "FunctionID": func_id,
                            "Function": func_name,
                            "SegmentID": seg_id,
                            "Segment": seg_name,
                            "CircuitID": circ_id,
                            "Circuit": circ.get("Name", None),
                            "CCR_ID": circ.get("CCRID", None),
                            "CCR_Name": circ.get("CCRCircuitIndex", None),
                        })

            # 3a) LD -> Functions
            for ld_id, ld_name, ld in ld_rows:
                for func_key in ("LandFunctionIDs", "TakeOffFunctionIDs"):
                    for func_id in parse_ids(ld.get(func_key, "")):
                        add_function_chain(func_id, ld_id=ld_id, ld_name=ld_name)

            # 3b) Runway -> Functions (direct-only)
            for func_id in sorted(runway_direct_only_function_ids,
                                  key=lambda x: int(x) if str(x).isdigit() else str(x)):
                add_function_chain(func_id, ld_id=None, ld_name=None)

    return pd.DataFrame(chain)



from __future__ import annotations


# -----------------------------
# Reference implementations
# -----------------------------

def _ref_get_lds_for_runways(df_lds, df_runways_for_airport):
    ld_ids = set()
    for _, rwy in df_runways_for_airport.iterrows():
        for key in ("LD1ID", "LD2ID"):
            ld_ids.update(parse_ids(rwy.get(key)))

    df_lds_for_airport = filter_by_ids(df_lds, ld_ids)
    return ld_ids, df_lds_for_airport


def _ref_get_function_ids_from_lds(df_lds_for_airport):
    ld_function_ids = set()
    for _, ld in df_lds_for_airport.iterrows():
        for key in ("LandFunctionIDs", "TakeOffFunctionIDs"):
            ld_function_ids.update(parse_ids(ld.get(key)))
    return ld_function_ids


def _ref_get_function_ids_from_runways(df_runways_for_airport):
    runway_function_ids = set()
    for _, rwy in df_runways_for_airport.iterrows():
        runway_function_ids.update(parse_ids(rwy.get("FunctionIDs")))
    return runway_function_ids


def _ref_get_functions_for_airport(df_functions, ld_function_ids, runway_function_ids):
    runway_direct_only = set(runway_function_ids) - set(ld_function_ids)
    function_ids = set(ld_function_ids) | runway_direct_only
    df_functions_for_airport = filter_by_ids(df_functions, function_ids)
    return function_ids, runway_direct_only, df_functions_for_airport


def _ref_get_segments_for_functions(df_segments, df_functions_for_airport):
    segment_ids = set()
    for _, fn in df_functions_for_airport.iterrows():
        segment_ids.update(parse_ids(fn.get("SegmentIDs")))

    df_segments_for_airport = filter_by_ids(df_segments, segment_ids)
    return segment_ids, df_segments_for_airport


def _ref_get_circuit_ids_for_segments(df_segments_for_airport):
    circuit_ids = set()
    for _, seg in df_segments_for_airport.iterrows():
        circuit_ids.update(parse_ids(seg.get("CCRCircuitIDs")))
    return circuit_ids


# -----------------------------
# Comparison helpers
# -----------------------------

def _sorted_ids(x):
    return sorted(map(str, set(x)))


def _assert_set_equal(got, exp, label: str):
    got, exp = set(got), set(exp)
    if got != exp:
        missing = exp - got
        extra = got - exp
        raise AssertionError(
            f"{label}: set mismatch\n"
            f"  missing: {_sorted_ids(missing)}\n"
            f"  extra:   {_sorted_ids(extra)}"
        )


def _normalize_df(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    if id_col in out.columns:
        out[id_col] = out[id_col].astype(str)
        out = out.sort_values(id_col)
    return out.reset_index(drop=True)


def _assert_df_equal(got: pd.DataFrame, exp: pd.DataFrame, label: str, id_col: str = "ID"):
    got_n = _normalize_df(got, id_col=id_col)
    exp_n = _normalize_df(exp, id_col=id_col)
    try:
        pd.testing.assert_frame_equal(
            got_n, exp_n,
            check_dtype=False,   # tolerate dtype differences (common with CSVs)
            check_like=True      # tolerate column order differences
        )
    except AssertionError as e:
        raise AssertionError(f"{label}: dataframe mismatch\n{e}") from None


# -----------------------------
# Main checker (pipeline)
# -----------------------------

def check_airport_pipeline(
    student_ns: dict,
    df_airports,
    df_runways,
    df_lds,
    df_functions,
    df_segments,
    airport_id="1",
    collect_all_errors: bool = False,
):
    """
    student_ns: pass globals() from the notebook so we can access student functions by name.

    Raises AssertionError with a helpful message, or (if collect_all_errors=True)
    returns a list of error strings.
    """
    errors = []

    def _run_check(fn_label, check_callable):
        nonlocal errors
        try:
            check_callable()
        except Exception as e:
            if collect_all_errors:
                errors.append(f"{fn_label}: {e}")
            else:
                raise

    # Use your provided function to build the common starting point
    airport, runway_ids, df_runways_for_airport = get_airport_and_runways(
        df_airports, df_runways, airport_id=airport_id
    )

    # --- get_lds_for_runways ---
    def _check_lds():
        exp_ld_ids, exp_df_lds = _ref_get_lds_for_runways(df_lds, df_runways_for_airport)
        got_ld_ids, got_df_lds = student_ns["get_lds_for_runways"](df_lds, df_runways_for_airport)
        _assert_set_equal(got_ld_ids, exp_ld_ids, "get_lds_for_runways -> ld_ids")
        _assert_df_equal(got_df_lds, exp_df_lds, "get_lds_for_runways -> df_lds_for_airport")

    _run_check("get_lds_for_runways", _check_lds)

    # If previous step failed and we're collecting errors, we may not have valid got_* objects.
    # So compute expected intermediates for the remaining checks from reference:
    exp_ld_ids, exp_df_lds = _ref_get_lds_for_runways(df_lds, df_runways_for_airport)

    # --- get_function_ids_from_lds ---
    def _check_fn_ids_from_lds():
        exp = _ref_get_function_ids_from_lds(exp_df_lds)
        got = student_ns["get_function_ids_from_lds"](exp_df_lds)
        _assert_set_equal(got, exp, "get_function_ids_from_lds -> ld_function_ids")

    _run_check("get_function_ids_from_lds", _check_fn_ids_from_lds)

    # --- get_function_ids_from_runways ---
    def _check_fn_ids_from_runways():
        exp = _ref_get_function_ids_from_runways(df_runways_for_airport)
        got = student_ns["get_function_ids_from_runways"](df_runways_for_airport)
        _assert_set_equal(got, exp, "get_function_ids_from_runways -> runway_function_ids")

    _run_check("get_function_ids_from_runways", _check_fn_ids_from_runways)

    exp_ld_function_ids = _ref_get_function_ids_from_lds(exp_df_lds)
    exp_runway_function_ids = _ref_get_function_ids_from_runways(df_runways_for_airport)

    # --- get_functions_for_airport ---
    def _check_functions_for_airport():
        exp_function_ids, exp_runway_direct_only, exp_df_functions = _ref_get_functions_for_airport(
            df_functions, exp_ld_function_ids, exp_runway_function_ids
        )
        got_function_ids, got_runway_direct_only, got_df_functions = student_ns["get_functions_for_airport"](
            df_functions, exp_ld_function_ids, exp_runway_function_ids
        )
        _assert_set_equal(got_function_ids, exp_function_ids, "get_functions_for_airport -> function_ids")
        _assert_set_equal(got_runway_direct_only, exp_runway_direct_only, "get_functions_for_airport -> runway_direct_only")
        _assert_df_equal(got_df_functions, exp_df_functions, "get_functions_for_airport -> df_functions_for_airport")

    _run_check("get_functions_for_airport", _check_functions_for_airport)

    exp_function_ids, exp_runway_direct_only, exp_df_functions = _ref_get_functions_for_airport(
        df_functions, exp_ld_function_ids, exp_runway_function_ids
    )

    # --- get_segments_for_functions ---
    def _check_segments_for_functions():
        exp_segment_ids, exp_df_segments = _ref_get_segments_for_functions(df_segments, exp_df_functions)
        got_segment_ids, got_df_segments = student_ns["get_segments_for_functions"](df_segments, exp_df_functions)
        _assert_set_equal(got_segment_ids, exp_segment_ids, "get_segments_for_functions -> segment_ids")
        _assert_df_equal(got_df_segments, exp_df_segments, "get_segments_for_functions -> df_segments_for_airport")

    _run_check("get_segments_for_functions", _check_segments_for_functions)

    exp_segment_ids, exp_df_segments = _ref_get_segments_for_functions(df_segments, exp_df_functions)

    # --- get_circuit_ids_for_segments ---
    def _check_circuit_ids():
        exp_circuit_ids = _ref_get_circuit_ids_for_segments(exp_df_segments)
        got_circuit_ids = student_ns["get_circuit_ids_for_segments"](exp_df_segments)
        _assert_set_equal(got_circuit_ids, exp_circuit_ids, "get_circuit_ids_for_segments -> circuit_ids")

    _run_check("get_circuit_ids_for_segments", _check_circuit_ids)

    return errors if collect_all_errors else True

