# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:28:35 2026

@author: aprash
"""
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pathlib import Path

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

def _module_dir() -> Path:
    # folder that contains this function.py file
    return Path(__file__).resolve().parent


def _norm_id_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)  # Excel sometimes makes IDs "123.0"
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s

def normalize_chain_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()

    # Normalize ID-like cols
    id_cols = [c for c in out.columns if c.endswith("ID") or c in ("CCR_ID",)]
    for c in id_cols:
        if c in out.columns:
            out[c] = _norm_id_series(out[c])

    # Normalize some name-like cols
    name_cols = [c for c in out.columns if c.lower() in {
        "airport","runway","ld","function","segment","circuit","ccr_name"
    }]
    for c in name_cols:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip()

    return out

def read_expected_chain_excel(
    filename: str = "Queenstown.xlsx",
    sheet_name=0
) -> pd.DataFrame:
    path = _module_dir() / filename
    return pd.read_excel(path, sheet_name=sheet_name)

def compare_chain_to_excel(
    df_chain: pd.DataFrame,
    filename: str = "Queenstown.xlsx",
    sheet_name=0,
    raise_on_mismatch: bool = True
):
    """
    Compares df_chain to an Excel file located in the SAME folder as function.py.

    Returns dict with:
      matches (bool),
      missing_from_excel (DataFrame),
      missing_from_chain (DataFrame),
      column_report (dict)
    """
    df_xlsx = read_expected_chain_excel(filename=filename, sheet_name=sheet_name)

    c = normalize_chain_df(df_chain)
    x = normalize_chain_df(df_xlsx)

    missing_in_xlsx = sorted(set(c.columns) - set(x.columns))
    missing_in_chain = sorted(set(x.columns) - set(c.columns))
    common_cols = [col for col in c.columns if col in x.columns]

    c2 = c[common_cols].copy()
    x2 = x[common_cols].copy()

    # Compare as unordered sets of rows (by sorting on all common columns)
    c2s = c2.sort_values(common_cols, na_position="first").reset_index(drop=True)
    x2s = x2.sort_values(common_cols, na_position="first").reset_index(drop=True)

    # Find row-level differences
    m = c2.merge(x2, how="outer", indicator=True)
    missing_from_excel = m[m["_merge"] == "left_only"].drop(columns="_merge")
    missing_from_chain = m[m["_merge"] == "right_only"].drop(columns="_merge")

    matches = (len(missing_from_excel) == 0 and len(missing_from_chain) == 0
               and not missing_in_xlsx and not missing_in_chain)

    result = {
        "matches": matches,
        "missing_from_excel": missing_from_excel,
        "missing_from_chain": missing_from_chain,
        "column_report": {
            "missing_in_excel": missing_in_xlsx,
            "missing_in_chain": missing_in_chain,
            "common_cols_used_for_compare": common_cols,
        }
    }

    if raise_on_mismatch and not matches:
        msg = (
            f"df_chain does NOT match {filename}\n"
            f"Missing columns in Excel: {missing_in_xlsx}\n"
            f"Missing columns in chain: {missing_in_chain}\n"
            f"Rows in chain but not Excel: {len(missing_from_excel)}\n"
            f"Rows in Excel but not chain: {len(missing_from_chain)}\n"
        )
        raise AssertionError(msg)

    return result