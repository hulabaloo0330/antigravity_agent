"""Multi-layer signed connectome priors, CeNGEN receptor polarity, 3D soma coordinates, and functional tiers.

Implements Interface Contract 3 (`PROJECT.md`) and `MODEL_REQUIREMENTS.md` §7:
- Preserves distinct left/right and ON/OFF identities (`ASEL` vs `ASER`, `AWCON` vs `AWCOFF`, `AWCL` vs `AWCR`).
- Never assumes all `EAT-4` (glutamate) or `UNC-17` (acetylcholine) edges are uniformly `+1`; resolves
  edge sign from presynaptic vesicular transporter expression paired with postsynaptic ionotropic
  excitatory vs Cl-channel inhibitory receptor expression in CeNGEN (`cengen_021821_conservative_threshold3.csv`).
- Exports cached artifact bundle under `/home/antigravity/data/connectomes/` and Randi 2023 perturbation
  atlas under `/home/antigravity/data/randi2023_atlas/`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

CONNECTOME_DIR_DEFAULT = Path("/home/antigravity/data/connectomes")
RANDI_DIR_DEFAULT = Path("/home/antigravity/data/randi2023_atlas")

PREMOTOR_COMMAND_CLASSES = {"AVA", "AVB", "AVD", "AVE", "PVC"}

# Canonical ionotropic receptor gene polarity panels (WormBook / Hobert / Yemini 2021 / Fenyves 2020)
GLU_EXCITATORY_GENES = [f"glr-{i}" for i in range(1, 9)] + ["nmr-1", "nmr-2"]
GLU_INHIBITORY_GENES = ["avr-14", "avr-15", "glc-1", "glc-2", "glc-3", "glc-4"]

ACH_EXCITATORY_GENES = [
    "acr-2", "acr-3", "acr-5", "acr-6", "acr-7", "acr-8", "acr-9", "acr-10",
    "acr-11", "acr-12", "acr-14", "acr-15", "acr-16", "acr-18", "acr-19",
    "acr-20", "acr-21", "acr-23", "unc-29", "unc-38", "unc-63", "lev-1", "lev-8", "deg-3", "des-2",
]
ACH_INHIBITORY_GENES = ["acc-1", "acc-2", "acc-3", "acc-4", "lgc-46", "lgc-47", "lgc-48", "lgc-49"]

GABA_INHIBITORY_GENES = ["unc-49", "lgc-35", "lgc-36", "lgc-37", "lgc-38", "gab-1"]
GABA_EXCITATORY_GENES = ["exp-1"]


def canonicalize_neuron_name(name: str) -> str:
    """Normalize neuron name formatting while strictly preserving L/R and ON/OFF identities."""
    s = str(name).strip()
    if not s:
        return s
    # Normalize AWC^ON / AWC_ON / AWCon -> AWCON, AWC^OFF / AWCOff -> AWCOFF
    up = s.upper().replace("^", "").replace("-", "").replace("_", "")
    if up == "AWCON":
        return "AWCON"
    if up == "AWCOFF":
        return "AWCOFF"
    # Normalize zero-padded ventral cord motor neuron names: VA01 -> VA1, DB02 -> DB2, AS01 -> AS1, VD01 -> VD1, DD01 -> DD1, DA01 -> DA1, VB01 -> VB1, VC01 -> VC1
    m = re.match(r"^(VA|VB|VC|VD|DA|DB|DD|AS)0+([1-9]\d*)$", s.strip())
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s.strip()


def neuron_to_cengen_lookup_keys(neuron: str) -> List[str]:
    """Return ordered lookup keys into CeNGEN columns (exact cell first, then class fallback)."""
    n = canonicalize_neuron_name(neuron)
    keys = [n]
    if n in ("AWCON", "AWCOFF", "AWCL", "AWCR"):
        keys.extend(["AWCON" if n == "AWCON" else ("AWCOFF" if n == "AWCOFF" else n), "AWCON", "AWCOFF", "AWC"])
    if n in ("ASEL", "ASER"):
        keys.append("ASE")
    # Strip L/R suffix for symmetric pairs (avoid stripping AQR, RIR, PQR)
    if n not in ("AQR", "RIR", "PQR") and len(n) >= 4 and n[-1] in ("L", "R"):
        base = n[:-1]
        keys.append(base)
        # Strip D/V if 4-fold (e.g., CEPDL -> CEPD -> CEP, IL1VL -> IL1V -> IL1)
        if len(base) >= 4 and base[-1] in ("D", "V"):
            keys.append(base[:-1])
    # Ventral cord motor classes (e.g., VA1 -> VA, DB2 -> DB)
    m = re.match(r"^(VA|VB|VC|VD|DA|DB|DD|AS)\d+$", n)
    if m:
        keys.append(m.group(1))
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def infer_functional_tier(neuron: str, cell_info_map: Dict[str, Tuple[str, str]]) -> str:
    """Assign canonical functional tier: sensory, interneuron, premotor_command, motor, pharyngeal, or unknown."""
    n = canonicalize_neuron_name(neuron)
    # Check premotor command first
    for cls in PREMOTOR_COMMAND_CLASSES:
        if n == cls or (n.startswith(cls) and n[len(cls):] in ("L", "R")):
            return "premotor_command"
    # Male-specific sensory/interneurons
    if n.startswith("CEM"):
        return "sensory"
    if n.startswith("MCM"):
        return "interneuron"
    # Pharyngeal neurons
    if re.match(r"^(I[1-6]|M[1-5]|MC|MI|NSM|NSML|NSMR|MCL|MCR|I1L|I1R|I2L|I2R|M2L|M2R|M3L|M3R)$", n):
        return "pharyngeal"
    # Lookup in cect all_cell_info.csv
    lookup_candidates = [n]
    if n in ("AWCON", "AWCOFF"):
        lookup_candidates.extend(["AWCL", "AWCR"])
    for cand in lookup_candidates:
        if cand in cell_info_map:
            ctype, cdesc = cell_info_map[cand]
            blob = f"{ctype} {cdesc}".lower()
            if "pharyn" in blob:
                return "pharyngeal"
            if "sensory" in blob or "mechanosensory" in blob or "amphid" in blob or "phasmid" in blob:
                return "sensory"
            if "motor" in blob and "interneuron" not in blob:
                return "motor"
            if "interneuron" in blob:
                return "interneuron"
            if "motor" in blob:
                return "motor"
    # Ventral cord motor fallback
    if re.match(r"^(VA|VB|VC|VD|DA|DB|DD|AS|RMD|RME|SMB|SMD|SIA|SIB|URA)", n):
        return "motor"
    return "interneuron"


def _matrix_from_edgelist_df(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    val_col: Optional[str],
    idx_map: Dict[str, int],
) -> np.ndarray:
    n = len(idx_map)
    mat = np.zeros((n, n), dtype=np.float32)
    for _, row in df.iterrows():
        u = canonicalize_neuron_name(str(row[src_col]))
        v = canonicalize_neuron_name(str(row[dst_col]))
        w = float(row[val_col]) if val_col is not None else 1.0
        # Map AWCL/AWCR also to AWCON/AWCOFF only via explicit AWC handling later
        if u in idx_map and v in idx_map:
            mat[idx_map[u], idx_map[v]] += w
    return mat


def _propagate_awc_aliases(mat: np.ndarray, idx_map: Dict[str, int]) -> None:
    """Populate AWCON/AWCOFF rows & cols from canonical AWCR/AWCL anatomical/molecular templates when unset.

    Follows WormNeuroAtlas / Cook convention (`AWCR -> AWCON`, `AWCL -> AWCOFF` for symmetric anatomical priors)
    while keeping all four entries (`AWCL`, `AWCR`, `AWCON`, `AWCOFF`) distinctly addressable.
    """
    pairs = [("AWCR", "AWCON"), ("AWCL", "AWCOFF")]
    for anat, func in pairs:
        if anat in idx_map and func in idx_map:
            ia, jf = idx_map[anat], idx_map[func]
            if np.all(mat[jf, :] == 0):
                mat[jf, :] = mat[ia, :]
            if np.all(mat[:, jf] == 0):
                mat[:, jf] = mat[:, ia]
    for anat1, func1 in pairs:
        for anat2, func2 in pairs:
            if anat1 in idx_map and func1 in idx_map and anat2 in idx_map and func2 in idx_map:
                if mat[idx_map[func1], idx_map[func2]] == 0:
                    mat[idx_map[func1], idx_map[func2]] = mat[idx_map[anat1], idx_map[anat2]]


def build_and_save_connectome_bundle(
    connectome_dir: Path = CONNECTOME_DIR_DEFAULT,
    randi_dir: Path = RANDI_DIR_DEFAULT,
) -> Dict[str, object]:
    """Build multi-layer connectome priors, CeNGEN receptor polarity, 3D coordinates, and Randi 2023 atlas."""
    import cect
    import wormneuroatlas as wna
    from cect.readers import (
        Cook2019HermReader,
        Cook2019MaleReader,
        WitvlietDataReader7,
        WitvlietDataReader8,
    )

    connectome_dir.mkdir(parents=True, exist_ok=True)
    randi_dir.mkdir(parents=True, exist_ok=True)

    cect_data = Path(cect.__file__).resolve().parent / "data"
    wna_data = Path(wna.__file__).resolve().parent / "data"

    # 1. Load cell info & 3D soma positions (300 neurons + CANL/CANR + AWCON/AWCOFF + male-specific CEM/MCM)
    cell_info_map: Dict[str, Tuple[str, str]] = {}
    with open(cect_data / "all_cell_info.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = canonicalize_neuron_name(row["Cell name"])
            cell_info_map[cname] = (row.get("Type", ""), row.get("Classification", ""))

    pos_lines = (wna_data / "anatlas_neuron_positions.txt").read_text(encoding="utf-8").splitlines()
    pos_names_raw = pos_lines[0].lstrip("#").strip().split()
    pos_coords_raw = []
    for line in pos_lines[1:]:
        if line.strip():
            parts = [float(x) for x in line.strip().split()]
            if len(parts) == 3:
                pos_coords_raw.append(parts)
    pos_map: Dict[str, Tuple[float, float, float]] = {}
    for nm, xyz in zip(pos_names_raw, pos_coords_raw):
        pos_map[canonicalize_neuron_name(nm)] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    # Build canonical neuron list: 302 hermaphrodite neurons + AWCON/AWCOFF + male head neurons CEMDL/DR/VL/VR, MCML/R
    herm_neurons_set = set(pos_map.keys()) | {"CANL", "CANR", "AWCON", "AWCOFF"}
    male_head_specific = {"CEMDL", "CEMDR", "CEMVL", "CEMVR", "MCML", "MCMR"}
    canonical_neurons = sorted(herm_neurons_set | male_head_specific)
    idx_map = {n: i for i, n in enumerate(canonical_neurons)}
    n_total = len(canonical_neurons)

    # 3D soma coordinates [N, 3] and pairwise Euclidean distance [N, N]
    soma_xyz = np.full((n_total, 3), np.nan, dtype=np.float32)
    for n, i in idx_map.items():
        if n in pos_map:
            soma_xyz[i] = pos_map[n]
        elif n == "AWCON" and "AWCR" in pos_map:
            soma_xyz[i] = pos_map["AWCR"]
        elif n == "AWCOFF" and "AWCL" in pos_map:
            soma_xyz[i] = pos_map["AWCL"]

    diff_xyz = soma_xyz[:, None, :] - soma_xyz[None, :, :]
    soma_dist = np.sqrt(np.sum(diff_xyz ** 2, axis=-1)).astype(np.float32)

    functional_tiers = [infer_functional_tier(n, cell_info_map) for n in canonical_neurons]

    # 2. Extract Cook 2019 Hermaphrodite & Male Chemical + Gap Junction matrices
    def extract_cect_reader(reader_cls) -> Tuple[np.ndarray, np.ndarray]:
        inst = reader_cls.get_instance()
        nodes = [canonicalize_neuron_name(x) for x in inst.nodes]
        cs_raw = inst.connections["Generic_CS"]
        gj_raw = inst.connections["Generic_GJ"]
        chem = np.zeros((n_total, n_total), dtype=np.float32)
        gap = np.zeros((n_total, n_total), dtype=np.float32)
        for r_i, u in enumerate(nodes):
            if u not in idx_map:
                continue
            iu = idx_map[u]
            nz_c = np.nonzero(cs_raw[r_i])[0]
            for c_j in nz_c:
                v = nodes[c_j]
                if v in idx_map:
                    chem[iu, idx_map[v]] += float(cs_raw[r_i, c_j])
            nz_g = np.nonzero(gj_raw[r_i])[0]
            for c_j in nz_g:
                v = nodes[c_j]
                if v in idx_map:
                    gap[iu, idx_map[v]] += float(gj_raw[r_i, c_j])
        # Make gap junction explicitly symmetric
        gap = 0.5 * (gap + gap.T)
        _propagate_awc_aliases(chem, idx_map)
        _propagate_awc_aliases(gap, idx_map)
        return chem, gap

    cook_h_chem, cook_h_gap = extract_cect_reader(Cook2019HermReader)
    cook_m_chem, cook_m_gap = extract_cect_reader(Cook2019MaleReader)

    # 3. Extract Witvliet 2021 Conserved Adult Core (intersection of Adult 7 and Adult 8, mean weight where both > 0)
    wit7_chem, wit7_gap = extract_cect_reader(WitvlietDataReader7)
    wit8_chem, wit8_gap = extract_cect_reader(WitvlietDataReader8)
    wit_conserved_chem = np.where((wit7_chem > 0) & (wit8_chem > 0), 0.5 * (wit7_chem + wit8_chem), 0.0).astype(np.float32)
    wit_conserved_gap = np.where((wit7_gap > 0) & (wit8_gap > 0), 0.5 * (wit7_gap + wit8_gap), 0.0).astype(np.float32)

    # 4. Extract Ripoll-Sanchez 2023 Neuropeptide (short, mid, long range) & Monoamine matrices
    def load_square_csv(csv_path: Path) -> np.ndarray:
        df = pd.read_csv(csv_path, index_col=0)
        mat = np.zeros((n_total, n_total), dtype=np.float32)
        rows = [canonicalize_neuron_name(str(x)) for x in df.index]
        cols = [canonicalize_neuron_name(str(x)) for x in df.columns]
        vals = df.to_numpy(dtype=np.float32)
        # Note: verify orientation — in Ripoll-Sanchez / cect CSVs, rows=receiver or sender?
        # Check against cect RipollSanchezShortRangeReader convention:
        for r_i, u in enumerate(rows):
            if u not in idx_map:
                continue
            iu = idx_map[u]
            nz = np.nonzero(vals[r_i])[0]
            for c_j in nz:
                v = cols[c_j]
                if v in idx_map:
                    mat[iu, idx_map[v]] = float(vals[r_i, c_j])
        _propagate_awc_aliases(mat, idx_map)
        return mat

    # Check cect RipollSanchezShortRangeReader orientation directly so sender->receiver matches chemical [pre, post]
    from cect.readers import RipollSanchezShortRangeReader
    rs_inst = RipollSanchezShortRangeReader.get_instance()
    rs_key = list(rs_inst.connections.keys())[0]
    rs_nodes = [canonicalize_neuron_name(x) for x in rs_inst.nodes]
    rs_raw = rs_inst.connections[rs_key]
    np_short = np.zeros((n_total, n_total), dtype=np.float32)
    for r_i, u in enumerate(rs_nodes):
        if u in idx_map:
            iu = idx_map[u]
            for c_j in np.nonzero(rs_raw[r_i])[0]:
                v = rs_nodes[c_j]
                if v in idx_map:
                    np_short[iu, idx_map[v]] = float(rs_raw[r_i, c_j])
    _propagate_awc_aliases(np_short, idx_map)

    # Determine whether CSV is transposed relative to cect [pre, post]
    csv_short_raw = load_square_csv(cect_data / "01022024_neuropeptide_connectome_short_range_model.csv")
    same_orient = np.sum(np.abs(csv_short_raw - np_short)) <= np.sum(np.abs(csv_short_raw.T - np_short))
    np_mid = load_square_csv(cect_data / "01022024_neuropeptide_connectome_mid_range_model.csv")
    np_long = load_square_csv(cect_data / "01022024_neuropeptide_connectome_long_range_model.csv")
    monoamine = load_square_csv(cect_data / "08062023_monoamine_connectome.csv")
    if not same_orient:
        np_mid = np_mid.T.copy()
        np_long = np_long.T.copy()
        monoamine = monoamine.T.copy()

    # 5. Genuine Biophysical Receptor-Mediated Edge Polarity from CeNGEN (`cengen_021821_conservative_threshold3.csv`)
    cengen_df = pd.read_csv(wna_data / "cengen_021821_conservative_threshold3.csv")
    cengen_df["gene_lower"] = cengen_df["gene_name"].astype(str).str.strip().str.lower()
    cengen_df = cengen_df.set_index("gene_lower")
    cengen_cols = set(cengen_df.columns)

    def get_expr(neuron: str, genes: Sequence[str]) -> float:
        keys = [k for k in neuron_to_cengen_lookup_keys(neuron) if k in cengen_cols]
        if not keys:
            return np.nan
        col = keys[0]
        present = [g.lower() for g in genes if g.lower() in cengen_df.index]
        if not present:
            return 0.0
        return float(cengen_df.loc[present, col].astype(float).sum())

    pre_glu = np.array([get_expr(n, ["eat-4"]) for n in canonical_neurons], dtype=np.float32)
    pre_ach = np.array([get_expr(n, ["unc-17", "cha-1"]) for n in canonical_neurons], dtype=np.float32)
    pre_gaba = np.array([get_expr(n, ["unc-25", "unc-47"]) for n in canonical_neurons], dtype=np.float32)

    post_glu_exc = np.array([get_expr(n, GLU_EXCITATORY_GENES) for n in canonical_neurons], dtype=np.float32)
    post_glu_inh = np.array([get_expr(n, GLU_INHIBITORY_GENES) for n in canonical_neurons], dtype=np.float32)
    post_ach_exc = np.array([get_expr(n, ACH_EXCITATORY_GENES) for n in canonical_neurons], dtype=np.float32)
    post_ach_inh = np.array([get_expr(n, ACH_INHIBITORY_GENES) for n in canonical_neurons], dtype=np.float32)
    post_gaba_inh = np.array([get_expr(n, GABA_INHIBITORY_GENES) for n in canonical_neurons], dtype=np.float32)
    post_gaba_exc = np.array([get_expr(n, GABA_EXCITATORY_GENES) for n in canonical_neurons], dtype=np.float32)

    # Edge polarity matrix P_ij in {-1.0, 0.0 (mixed/ambiguous), +1.0, np.nan (unknown/no receptor)}
    # Evaluated across all anatomical chemical edges (union of Cook H, Cook M, Witvliet)
    any_chem = (cook_h_chem > 0) | (cook_m_chem > 0) | (wit_conserved_chem > 0)
    polarity = np.full((n_total, n_total), np.nan, dtype=np.float32)
    for i in range(n_total):
        if np.isnan(pre_glu[i]) and np.isnan(pre_ach[i]) and np.isnan(pre_gaba[i]):
            continue
        has_glu = (not np.isnan(pre_glu[i])) and pre_glu[i] > 0
        has_ach = (not np.isnan(pre_ach[i])) and pre_ach[i] > 0
        has_gaba = (not np.isnan(pre_gaba[i])) and pre_gaba[i] > 0
        if not (has_glu or has_ach or has_gaba):
            continue
        js = np.nonzero(any_chem[i])[0]
        for j in js:
            if np.isnan(post_glu_exc[j]):
                continue
            pos_drive = 0.0
            neg_drive = 0.0
            if has_glu:
                if post_glu_exc[j] > 0:
                    pos_drive += 1.0
                if post_glu_inh[j] > 0:
                    neg_drive += 1.0
            if has_ach:
                if post_ach_exc[j] > 0:
                    pos_drive += 1.0
                if post_ach_inh[j] > 0:
                    neg_drive += 1.0
            if has_gaba:
                if post_gaba_exc[j] > 0:
                    pos_drive += 1.0
                if post_gaba_inh[j] > 0:
                    neg_drive += 1.0
            if pos_drive > 0 and neg_drive == 0:
                polarity[i, j] = 1.0
            elif neg_drive > 0 and pos_drive == 0:
                polarity[i, j] = -1.0
            elif pos_drive > 0 and neg_drive > 0:
                # Dual/mixed excitatory + Cl-channel inhibitory receptors (e.g. GLR + GLC/AVR): explicitly 0.0 (ambiguous/biphasic)
                polarity[i, j] = 0.0
            else:
                polarity[i, j] = np.nan

    bundle_path = connectome_dir / "connectome_priors_bundle.npz"
    np.savez_compressed(
        bundle_path,
        canonical_neurons=np.array(canonical_neurons, dtype="<U16"),
        functional_tiers=np.array(functional_tiers, dtype="<U24"),
        soma_xyz=soma_xyz,
        soma_distance=soma_dist,
        cook2019_herm_chem=cook_h_chem,
        cook2019_herm_gap=cook_h_gap,
        cook2019_male_chem=cook_m_chem,
        cook2019_male_gap=cook_m_gap,
        witvliet2021_conserved_chem=wit_conserved_chem,
        witvliet2021_conserved_gap=wit_conserved_gap,
        ripollsanchez2023_short=np_short,
        ripollsanchez2023_mid=np_mid,
        ripollsanchez2023_long=np_long,
        monoamine=monoamine,
        edge_polarity=polarity,
        pre_eat4_glu=pre_glu,
        pre_unc17_ach=pre_ach,
        pre_unc25_gaba=pre_gaba,
    )

    meta_summary = {
        "n_canonical_neurons": n_total,
        "canonical_neurons": canonical_neurons,
        "tier_counts": {t: int(functional_tiers.count(t)) for t in sorted(set(functional_tiers))},
        "soma_xyz_coverage": int(np.sum(~np.isnan(soma_xyz[:, 0]))),
        "layers": {
            "cook2019_herm_chem_nnz": int(np.sum(cook_h_chem > 0)),
            "cook2019_herm_gap_nnz": int(np.sum(cook_h_gap > 0)),
            "cook2019_male_chem_nnz": int(np.sum(cook_m_chem > 0)),
            "cook2019_male_gap_nnz": int(np.sum(cook_m_gap > 0)),
            "witvliet2021_conserved_chem_nnz": int(np.sum(wit_conserved_chem > 0)),
            "witvliet2021_conserved_gap_nnz": int(np.sum(wit_conserved_gap > 0)),
            "ripollsanchez2023_short_nnz": int(np.sum(np_short > 0)),
            "ripollsanchez2023_mid_nnz": int(np.sum(np_mid > 0)),
            "ripollsanchez2023_long_nnz": int(np.sum(np_long > 0)),
            "monoamine_nnz": int(np.sum(monoamine > 0)),
        },
        "polarity_counts_on_chem_edges": {
            "excitatory_plus1": int(np.sum(polarity == 1.0)),
            "inhibitory_minus1": int(np.sum(polarity == -1.0)),
            "mixed_biphasic_zero": int(np.sum(polarity == 0.0)),
            "unknown_nan": int(np.sum(any_chem & np.isnan(polarity))),
        },
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    }
    (connectome_dir / "connectome_metadata.json").write_text(json.dumps(meta_summary, indent=2), encoding="utf-8")

    # 6. Export Randi et al. (Nature 2023) WT vs unc-31 Optogenetic Perturbation Atlas (`funatlas.h5`)
    with h5py.File(wna_data / "funatlas.h5", "r") as fh:
        r_ids = [canonicalize_neuron_name(x.decode("utf-8")) for x in fh["neuron_ids"][:]]
        wt_dff = fh["wt/dFF"][:].astype(np.float32)
        wt_q = fh["wt/q"][:].astype(np.float32)
        wt_occ = fh["wt/occ1"][:].astype(np.int32)
        u31_dff = fh["unc31/dFF"][:].astype(np.float32)
        u31_q = fh["unc31/q"][:].astype(np.float32)
        u31_occ = fh["unc31/occ1"][:].astype(np.int32)

    randi_npz = randi_dir / "randi2023_wt_unc31_atlas.npz"
    np.savez_compressed(
        randi_npz,
        neuron_ids=np.array(r_ids, dtype="<U16"),
        wt_dFF=wt_dff,
        wt_q=wt_q,
        wt_occ1=wt_occ,
        unc31_dFF=u31_dff,
        unc31_q=u31_q,
        unc31_occ1=u31_occ,
    )
    # Identify unc-31-dependent wireless edges: significant in WT (q < 0.05) and abolished/reduced in unc-31 (q >= 0.05 and |unc31_dFF| < 0.5 * |wt_dFF|, tested >= 1 in unc-31)
    wt_sig = (~np.isnan(wt_q)) & (wt_q < 0.05)
    u31_tested = u31_occ > 0
    u31_dep = wt_sig & u31_tested & ((np.isnan(u31_q) | (u31_q >= 0.05)) & (np.abs(np.nan_to_num(u31_dff)) < 0.5 * np.abs(np.nan_to_num(wt_dff))))
    randi_meta = {
        "dataset_id": "randi2023_atlas",
        "n_neurons": len(r_ids),
        "wt_animals": 113,
        "unc31_animals": 18,
        "wt_tested_directed_pairs": int(np.sum(wt_occ > 0)),
        "wt_significant_pairs_q005": int(np.sum(wt_sig)),
        "unc31_tested_directed_pairs": int(np.sum(u31_tested)),
        "unc31_dependent_candidate_pairs": int(np.sum(u31_dep)),
        "governance_boundary": "EXTERNAL_MECHANISTIC_PERTURBATION_VALIDATION_ONLY_NEVER_USED_FOR_EDGE_PRIOR_SELECTION",
        "sha256": hashlib.sha256(randi_npz.read_bytes()).hexdigest(),
    }
    (randi_dir / "randi2023_metadata.json").write_text(json.dumps(randi_meta, indent=2), encoding="utf-8")
    return meta_summary


def load_connectome_priors(
    canonical_neurons: Optional[Sequence[str]] = None,
    sex: str = "hermaphrodite",
    connectome_dir: Path = CONNECTOME_DIR_DEFAULT,
) -> Dict[str, object]:
    """Load multi-layer connectome priors aligned to `canonical_neurons` (Interface Contract 3).

    Parameters
    ----------
    canonical_neurons : sequence of str, optional
        Ordered neuron names to align `[N, N]` and `[N, 3]` arrays to. If `None`, returns the full
        canonical neuron set stored in `connectome_priors_bundle.npz`.
    sex : {'hermaphrodite', 'male'}
        Selects `cook2019` sex-matched wired chemical and gap-junction matrices alongside all layers.
    connectome_dir : Path
        Directory containing `connectome_priors_bundle.npz`.
    """
    bundle_path = Path(connectome_dir) / "connectome_priors_bundle.npz"
    if not bundle_path.exists():
        build_and_save_connectome_bundle(connectome_dir=Path(connectome_dir))

    z = np.load(bundle_path, allow_pickle=False)
    stored_neurons = [str(x) for x in z["canonical_neurons"].tolist()]
    stored_idx = {n: i for i, n in enumerate(stored_neurons)}

    if canonical_neurons is None:
        target_neurons = stored_neurons
    else:
        target_neurons = [canonicalize_neuron_name(n) for n in canonical_neurons]

    m = len(target_neurons)
    map_idx = [stored_idx.get(n, -1) for n in target_neurons]
    valid_mask = np.array([idx >= 0 for idx in map_idx], dtype=bool)

    def align_2d(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
        out = np.full((m, m), fill, dtype=np.float32)
        valid_pos = np.where(valid_mask)[0]
        src_pos = [map_idx[p] for p in valid_pos]
        if src_pos:
            out[np.ix_(valid_pos, valid_pos)] = arr[np.ix_(src_pos, src_pos)]
        return out

    def align_1d_str(arr: np.ndarray, default: str = "unknown") -> List[str]:
        return [str(arr[idx]) if idx >= 0 else default for idx in map_idx]

    soma_xyz = np.full((m, 3), np.nan, dtype=np.float32)
    for p, idx in enumerate(map_idx):
        if idx >= 0:
            soma_xyz[p] = z["soma_xyz"][idx]

    sex_norm = sex.strip().lower()
    if sex_norm not in ("hermaphrodite", "male"):
        raise ValueError(f"Unsupported sex={sex!r}; expected 'hermaphrodite' or 'male'.")

    active_chem = align_2d(z["cook2019_male_chem"] if sex_norm == "male" else z["cook2019_herm_chem"], 0.0)
    active_gap = align_2d(z["cook2019_male_gap"] if sex_norm == "male" else z["cook2019_herm_gap"], 0.0)
    polarity = align_2d(z["edge_polarity"], np.nan)

    # Signed chemical adjacency variants: excitatory-only, inhibitory-only, ambiguous-only
    chem_excitatory = np.where(polarity == 1.0, active_chem, 0.0).astype(np.float32)
    chem_inhibitory = np.where(polarity == -1.0, active_chem, 0.0).astype(np.float32)
    chem_ambiguous = np.where((active_chem > 0) & ((polarity == 0.0) | np.isnan(polarity)), active_chem, 0.0).astype(np.float32)

    return {
        "canonical_neurons": target_neurons,
        "sex": sex_norm,
        "functional_tiers": align_1d_str(z["functional_tiers"]),
        "soma_xyz": soma_xyz,
        "soma_distance": align_2d(z["soma_distance"], np.nan),
        "chem": active_chem,
        "gap": active_gap,
        "chem_excitatory": chem_excitatory,
        "chem_inhibitory": chem_inhibitory,
        "chem_ambiguous": chem_ambiguous,
        "edge_polarity": polarity,
        "cook2019_herm_chem": align_2d(z["cook2019_herm_chem"], 0.0),
        "cook2019_herm_gap": align_2d(z["cook2019_herm_gap"], 0.0),
        "cook2019_male_chem": align_2d(z["cook2019_male_chem"], 0.0),
        "cook2019_male_gap": align_2d(z["cook2019_male_gap"], 0.0),
        "witvliet2021_conserved_chem": align_2d(z["witvliet2021_conserved_chem"], 0.0),
        "witvliet2021_conserved_gap": align_2d(z["witvliet2021_conserved_gap"], 0.0),
        "neuropeptide_short": align_2d(z["ripollsanchez2023_short"], 0.0),
        "neuropeptide_mid": align_2d(z["ripollsanchez2023_mid"], 0.0),
        "neuropeptide_long": align_2d(z["ripollsanchez2023_long"], 0.0),
        "monoamine": align_2d(z["monoamine"], 0.0),
    }


if __name__ == "__main__":
    summary = build_and_save_connectome_bundle()
    print(json.dumps({k: v for k, v in summary.items() if k != "canonical_neurons"}, indent=2))
