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


