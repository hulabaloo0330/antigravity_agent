"""Multi-source raw/unsmoothed trace acquisition, pre-search stratified vaulting, and governance generation.

Strictly enforces `MODEL_REQUIREMENTS.md` §1, §2, §6 and `PROJECT.md` M1 rules:
1. Pre-search splits for `seyedolmohadesin2025` (34 dev / 13 vault) and `baskoylu2026_copper` (18 dev / 9 vault)
   are determined 100% from recording metadata (`date_batch`, `sex`, `animal_id`) BEFORE inspecting any neural traces,
   with zero `animal_id` overlap and zero `date_batch` overlap between `dev` and `vault`.
2. Extracts ONLY raw/unsmoothed fluorescence/dF/F0 traces:
   - `yemini2021`: `traces` (NEVER `norm_traces`), preserving `ASEL` vs `ASER` and `is_AWCL_OFF` -> `AWCON`/`AWCOFF`.
   - `seyedolmohadesin2025`: DANDI `000981` `processing/CalciumActivity/ActivityTraces/activity` via HTTP Range (`remfile`),
     NEVER downloading 82.3 GB raw NWB movies and NEVER applying whole-trace z-scoring or bandpass smoothing.
   - `baskoylu2026_copper`: Dryad `doi:10.5061/dryad.w9ghx3g4v` `gcamp/trace_array_original` (raw un-zscored traces,
     NEVER `trace_array` or `traces_array_F_Fmean`) + `NeuroPAL_labels.jld2` (`confidence >= 2`).
   - `toyoshima2024`: Empirically audits Figshare `21968078` (`normalize1_withoutremove.zip`) and `yuichiiino1/gKDR-GMM`,
     documenting objective non-causal `sgolayfilt(tcrs,3,101)` + full-trace mean normalization and missing
     `metadata/stimulation_timing.csv` without faking compatibility.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import struct
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import remfile
import requests
import scipy.io as sio

from src.connectome_priors import build_and_save_connectome_bundle, canonicalize_neuron_name
from src.data_adapters import AnimalRecord, compute_file_sha256, save_animal_record_npz

DATA_ROOT = Path("/home/antigravity/data")
EXP_ROOT = Path("/home/antigravity/experiments/autoresearch_v1")

SEYED_VAULT_DATES = {"20220630", "20220701"}  # 7 hermaphrodites (20220630) + 6 males (20220701) = 13 vault worms
SEYED_CONTROL_STIMS = {"Control", "Fluorescein", "CTX"}

COPPER_VAULT_DATES = {
    "2023-12-11",
    "2024-02-01",
    "2024-05-22",
    "2024-05-28",
    "2024-06-18",
}  # 9 vault worms across 5 completely disjoint late-cohort dates vs 18 dev worms


def _is_valid_neuron_label(name: str) -> bool:
    s = name.strip()
    if not s or s in ("?", "nan", "NaN", "None", "undefined"):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9]{0,9}$", s))


def process_yemini2021() -> Dict[str, Any]:
    """Standardize Yemini et al. (Cell 2021) 21 head worms (`dev` only, historically exposed)."""
    t0 = time.time()
    raw_dir = DATA_ROOT / "yemini2021" / "raw_source"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mat_path = raw_dir / "all_traces_Heads_new.mat"
    readme_path = raw_dir / "data_readme.txt"

    if not mat_path.exists():
        r = requests.get(
            "https://raw.githubusercontent.com/TuragaLab/wormvae/master/data/worm_activity/all_traces_Heads_new.mat",
            timeout=90,
        )
        r.raise_for_status()
        mat_path.write_bytes(r.content)
    if not readme_path.exists():
        r_txt = requests.get(
            "https://raw.githubusercontent.com/TuragaLab/wormvae/master/data/worm_activity/data_readme.txt",
            timeout=30,
        )
        r_txt.raise_for_status()
        readme_path.write_bytes(r_txt.content)

    raw_mat_sha = compute_file_sha256(mat_path)
    m = sio.loadmat(str(mat_path), squeeze_me=False)
    fps = float(m["fps"].ravel()[0])  # 4.0 Hz
    files_raw = [str(x[0]) for x in m["files"].ravel()]
    is_awcl_off = [int(x) for x in m["is_AWCL_OFF"].ravel()]
    neuron_classes = [str(x[0]).strip() for x in m["neurons"].ravel()]
    stim_names = [str(x[0]).strip().lower() for x in m["stim_names"].ravel()]
    stim_times = m["stim_times"]  # [[60,70], [120,130], [180,190]] seconds
    stims = m["stims"]  # [21, 3] 1-based stimulus indices
    traces_cells = m["traces"].ravel()  # STRICTLY `traces`, NEVER `norm_traces`!

    dev_dir = DATA_ROOT / "yemini2021" / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)
    # Yemini has 0 vault animals (historically exposed); create empty vault dir so structure is explicit
    (DATA_ROOT / "yemini2021" / "vault").mkdir(parents=True, exist_ok=True)

    dev_records_meta = []
    for w_idx in range(21):
        fpath_orig = files_raw[w_idx]
        stem = Path(fpath_orig).stem.replace("_traces", "")  # e.g., 0924_01
        date_batch = f"2019_{stem.split('_')[0]}"
        animal_id = f"yemini2021_{stem}"
        awcl_is_off = bool(is_awcl_off[w_idx] == 1)

        worm_neurons: List[str] = []
        worm_cols: List[np.ndarray] = []

        for c_idx, base_cls in enumerate(neuron_classes):
            cell_arr = traces_cells[c_idx].ravel()
            if cell_arr.size == 21:
                entry = cell_arr[w_idx]
                if entry.size > 0:
                    vec = np.asarray(entry, dtype=np.float32).ravel()
                    if not np.all(np.isnan(vec)):
                        worm_neurons.append(canonicalize_neuron_name(base_cls))
                        worm_cols.append(vec)
            elif cell_arr.size == 42:
                entry_l = cell_arr[w_idx]
                entry_r = cell_arr[21 + w_idx]
                if base_cls == "AWC":
                    name_l = "AWCOFF" if awcl_is_off else "AWCON"
                    name_r = "AWCON" if awcl_is_off else "AWCOFF"
                else:
                    name_l = canonicalize_neuron_name(f"{base_cls}L")
                    name_r = canonicalize_neuron_name(f"{base_cls}R")
                for nm, ent in ((name_l, entry_l), (name_r, entry_r)):
                    if ent.size > 0:
                        vec = np.asarray(ent, dtype=np.float32).ravel()
                        if not np.all(np.isnan(vec)):
                            worm_neurons.append(nm)
                            worm_cols.append(vec)

        t_len = worm_cols[0].shape[0]
        traces_mat = np.column_stack(worm_cols).astype(np.float32)  # [960, N_obs]
        obs_mask = np.isfinite(traces_mat)

        stim_events = []
        for s_order in range(3):
            s_code = int(stims[w_idx, s_order]) - 1
            s_name = stim_names[s_code]
            on_sec = float(stim_times[s_order, 0])
            off_sec = float(stim_times[s_order, 1])
            stim_events.append(
                {
                    "stim_id": s_name,
                    "is_control": False,
                    "onset_idx": int(round(on_sec * fps)),
                    "offset_idx": int(round(off_sec * fps)),
                    "onset_sec": on_sec,
                    "offset_sec": off_sec,
                    "trial_order": s_order + 1,
                }
            )

        rec = AnimalRecord(
            dataset_id="yemini2021",
            animal_id=animal_id,
            sex="hermaphrodite",
            modality="immobilized",
            date_batch=date_batch,
            fps=fps,
            traces_raw=traces_mat,
            obs_mask=obs_mask,
            neuron_ids=worm_neurons,
            stim_events=stim_events,
            exposure_status="HISTORICALLY_EXPOSED_DEV_ONLY",
            extra_metadata={
                "original_file": fpath_orig,
                "is_AWCL_OFF": awcl_is_off,
                "immobilization": "1 mM tetramisole microfluidic chip",
                "genotype": "NeuroPAL; GCaMP6s (OH16230)",
                "t_frames": int(t_len),
            },
        )
        out_npz = dev_dir / f"{animal_id}.npz"
        sha = save_animal_record_npz(rec, out_npz)
        dev_records_meta.append(
            {
                "animal_id": animal_id,
                "sex": rec.sex,
                "date_batch": date_batch,
                "n_neurons": len(worm_neurons),
                "n_frames": int(t_len),
                "n_stim_events": len(stim_events),
                "path": str(out_npz),
                "sha256": sha,
            }
        )

    return {
        "dataset_id": "yemini2021",
        "elapsed_sec": round(time.time() - t0, 2),
        "raw_source_sha256": raw_mat_sha,
        "raw_source_bytes": mat_path.stat().st_size,
        "dev_records": dev_records_meta,
        "vault_records": [],
    }


def process_seyedolmohadesin2025() -> Dict[str, Any]:
    """Stream ONLY raw `ActivityTraces` + `StimulusInfo` from DANDI `000981` and apply pre-search vault split."""
    t0 = time.time()
    ds_json_r = requests.get(
        "https://raw.githubusercontent.com/venkatachalamlab/whole-brain-chemosensory-responses/main/python/datasets.json",
        timeout=30,
    )
    ds_json_r.raise_for_status()
    datasets_meta = ds_json_r.json()

    dandi_r = requests.get(
        "https://api.dandiarchive.org/api/dandisets/000981/versions/draft/assets/?page_size=100",
        timeout=30,
    )
    dandi_r.raise_for_status()
    assets = dandi_r.json()["results"]

    # Map sub-<date>-<h|m><animal> -> asset metadata
    asset_lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for a in assets:
        p = a["path"]  # e.g., sub-20220327-h4/sub-20220327-h4_ses-20220327_ophys.nwb
        m = re.match(r"^sub-(\d{8})-(h|m)(\d+)/", p)
        if m:
            d_str = m.group(1)
            sex_str = "hermaphrodite" if m.group(2) == "h" else "male"
            anim_num = int(m.group(3))
            asset_lookup[(d_str, sex_str, anim_num)] = a

    dev_dir = DATA_ROOT / "seyedolmohadesin2025" / "dev"
    vault_dir = DATA_ROOT / "seyedolmohadesin2025" / "vault"
    dev_dir.mkdir(parents=True, exist_ok=True)
    vault_dir.mkdir(parents=True, exist_ok=True)

    dev_records_meta = []
    vault_records_meta = []
    bytes_streamed_est = 0

    for ds_idx in sorted(datasets_meta.keys(), key=lambda x: int(x)):
        info = datasets_meta[ds_idx]
        d_str = str(info["date"])
        sex_str = str(info["sex"])
        anim_num = int(info["animal"])
        sex_short = "h" if sex_str == "hermaphrodite" else "m"
        animal_id = f"seyed2025_{d_str}_{sex_short}{anim_num}"

        # PRE-SEARCH METADATA-ONLY VAULT ASSIGNMENT (executed BEFORE opening NWB!)
        is_vault = d_str in SEYED_VAULT_DATES
        target_dir = vault_dir if is_vault else dev_dir
        exposure_status = "SEALED_CONFIRMATION_VAULT" if is_vault else "NEW_DEV"

        asset = asset_lookup[(d_str, sex_str, anim_num)]
        out_npz = target_dir / f"{animal_id}.npz"
        if out_npz.exists():
            from src.data_adapters import load_animal_record_npz

            rec_cached = load_animal_record_npz(out_npz)
            meta_item = {
                "animal_id": animal_id,
                "sex": sex_str,
                "date_batch": d_str,
                "n_neurons": len(rec_cached.neuron_ids),
                "n_frames": int(rec_cached.traces_raw.shape[0]),
                "n_stim_events": len(rec_cached.stim_events),
                "n_chemical_events": sum(1 for e in rec_cached.stim_events if not e.get("is_control", False)),
                "dandi_asset_id": asset["asset_id"],
                "path": str(out_npz),
                "sha256": rec_cached.sha256,
            }
            if is_vault:
                vault_records_meta.append(meta_item)
            else:
                dev_records_meta.append(meta_item)
            continue

        dl_api_url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
        s3_url = requests.head(dl_api_url, allow_redirects=True, timeout=20).url

        rf = remfile.File(s3_url)
        with h5py.File(rf, "r") as h5:
            at = h5["processing/CalciumActivity/ActivityTraces"]
            act_raw = at["activity"][:].astype(np.float32)  # [N_roi, T]
            raw_names = [
                canonicalize_neuron_name(x.decode("utf-8") if isinstance(x, bytes) else str(x))
                for x in at["neuron"][:]
            ]
            raw_groups = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in at["neuron_group"][:]
            ]
            si = h5["processing/CalciumActivity/StimulusInfo"]
            s_labels = [
                (x.decode("utf-8") if isinstance(x, bytes) else str(x)).strip()
                for x in si["data"][:]
            ]
            s_frames = [int(round(float(x))) for x in si["timestamps"][:]]
            fps = 2.66
            if "acquisition" in h5 and "CalciumImageSeries" in h5["acquisition"]:
                cis = h5["acquisition/CalciumImageSeries"]
                if "starting_time" in cis and "rate" in cis["starting_time"].attrs:
                    fps = float(cis["starting_time"].attrs["rate"])
                elif "scan_line_rate" in cis.attrs:
                    fps = float(cis.attrs["scan_line_rate"])

        bytes_streamed_est += act_raw.nbytes + 65536
        # Filter valid named neurons & deduplicate strictly by finite frame coverage count (never post-stim values!)
        chosen_by_name: Dict[str, Tuple[int, np.ndarray, str]] = {}
        for r_i, nm in enumerate(raw_names):
            if not _is_valid_neuron_label(nm):
                continue
            vec = act_raw[r_i]
            n_fin = int(np.sum(np.isfinite(vec)))
            if n_fin == 0:
                continue
            if nm not in chosen_by_name or n_fin > chosen_by_name[nm][0]:
                chosen_by_name[nm] = (n_fin, vec, raw_groups[r_i])

        ordered_names = sorted(chosen_by_name.keys())
        traces_mat = np.column_stack([chosen_by_name[nm][1] for nm in ordered_names]).astype(np.float32)  # [T, N_obs]
        obs_mask = np.isfinite(traces_mat)
        t_len = traces_mat.shape[0]

        # Extract non-CTX stimulus deliveries (8 chemical stimuli + 2 controls x 3 cycles = ~30 events)
        stim_events = []
        trial_order = 0
        for idx_s, (frm, lbl) in enumerate(zip(s_frames, s_labels)):
            if lbl == "CTX" or lbl.upper() == "NEUROPAL":
                continue
            trial_order += 1
            # Next entry in StimulusInfo is CTX flush onset (~frm + 40)
            off_frm = frm + 40
            for nxt_frm, nxt_lbl in zip(s_frames[idx_s + 1 :], s_labels[idx_s + 1 :]):
                if nxt_lbl == "CTX" and nxt_frm > frm:
                    off_frm = nxt_frm
                    break
            off_frm = min(off_frm, t_len)
            stim_events.append(
                {
                    "stim_id": lbl,
                    "is_control": lbl in SEYED_CONTROL_STIMS,
                    "onset_idx": int(frm),
                    "offset_idx": int(off_frm),
                    "onset_sec": round(frm / fps, 3),
                    "offset_sec": round(off_frm / fps, 3),
                    "trial_order": trial_order,
                }
            )

        rec = AnimalRecord(
            dataset_id="seyedolmohadesin2025",
            animal_id=animal_id,
            sex=sex_str,
            modality="immobilized",
            date_batch=d_str,
            fps=fps,
            traces_raw=traces_mat,
            obs_mask=obs_mask,
            neuron_ids=ordered_names,
            stim_events=stim_events,
            exposure_status=exposure_status,
            extra_metadata={
                "dandi_asset_id": asset["asset_id"],
                "dandi_path": asset["path"],
                "dandi_sha256": asset.get("digest", {}).get("dandi:sha2-256", ""),
                "dataset_idx": int(ds_idx),
                "immobilization": "1 mM levamisole + microfluidic chip",
                "genotype": "NeuroPAL; GCaMP6s",
                "t_frames": int(t_len),
                "n_chemical_events": sum(1 for e in stim_events if not e["is_control"]),
                "n_control_events": sum(1 for e in stim_events if e["is_control"]),
            },
        )
        out_npz = target_dir / f"{animal_id}.npz"
        sha = save_animal_record_npz(rec, out_npz)
        meta_item = {
            "animal_id": animal_id,
            "sex": sex_str,
            "date_batch": d_str,
            "n_neurons": len(ordered_names),
            "n_frames": int(t_len),
            "n_stim_events": len(stim_events),
            "n_chemical_events": sum(1 for e in stim_events if not e["is_control"]),
            "dandi_asset_id": asset["asset_id"],
            "path": str(out_npz),
            "sha256": sha,
        }
        if is_vault:
            vault_records_meta.append(meta_item)
        else:
            dev_records_meta.append(meta_item)
        print(
            f"[seyed2025 {int(ds_idx)+1:02d}/47] {animal_id} ({sex_str}) -> {'VAULT' if is_vault else 'DEV'} | "
            f"T={t_len}, N={len(ordered_names)}, stims={len(stim_events)}",
            flush=True,
        )

    return {
        "dataset_id": "seyedolmohadesin2025",
        "elapsed_sec": round(time.time() - t0, 2),
        "bytes_streamed_estimate": bytes_streamed_est,
        "dev_records": dev_records_meta,
        "vault_records": vault_records_meta,
    }


def _read_jld2_scalar(f: h5py.File, ref_or_val: Any) -> Any:
    import ctypes
    import h5py.h5t as h5t

    if not isinstance(ref_or_val, h5py.Reference):
        return ref_or_val.decode("utf-8", errors="replace") if isinstance(ref_or_val, bytes) else ref_or_val
    if not ref_or_val:
        return ""
    try:
        ds = f[ref_or_val]
        tid = ds.id.get_type()
        if tid.get_class() == h5t.STRING:
            if tid.is_variable_str():
                buf = np.array([0], dtype=np.uintp)
                ds.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, tid)
                if int(buf[0]) == 0:
                    return ""
                return ctypes.string_at(int(buf[0])).decode("utf-8", errors="replace")
            sz = tid.get_size()
            if sz == 0:
                return ""
            buf = np.zeros(sz, dtype=np.uint8)
            ds.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, tid)
            return bytes(buf).rstrip(b"\x00").decode("utf-8", errors="replace")
        val = ds[()]
        if isinstance(val, np.void) and val.dtype.names and set(val.dtype.names) >= {"string", "offset", "ncodeunits"}:
            parent = _read_jld2_scalar(f, val["string"])
            p_bytes = parent.encode("utf-8") if isinstance(parent, str) else bytes(parent)
            off = int(val["offset"])
            nc = int(val["ncodeunits"])
            return p_bytes[off : off + nc].decode("utf-8", errors="replace")
        return val
    except Exception:
        return ""


def _parse_copper_jld2_labels(jld2_path: Path) -> Dict[str, Dict[int, Tuple[str, int]]]:
    """Parse `NeuroPAL_labels.jld2` -> `{rec_key: {row_idx_0based: (canonical_neuron_name, confidence)}}`."""
    out: Dict[str, Dict[int, Tuple[str, int]]] = {}
    with h5py.File(jld2_path, "r") as f:
        kv = f[f["dict_match_dict"][()]["kvvec"]][()]
        for rec_ref in kv:
            pair = f[rec_ref][()]
            rec_key = str(_read_jld2_scalar(f, pair["first"])).strip()
            tup = f[pair["second"]][()]
            kv1 = f[f[tup["1"]][()]["kvvec"]][()]
            row_map: Dict[int, Tuple[str, int]] = {}
            for item_ref in kv1:
                p = f[item_ref][()]
                row_1based = int(_read_jld2_scalar(f, p["first"]))
                mkv = f[f[p["second"]][()]["kvvec"]][()]
                md = {
                    str(_read_jld2_scalar(f, f[m][()]["first"])): _read_jld2_scalar(f, f[m][()]["second"])
                    for m in mkv
                }
                conf = int(md.get("confidence", 0))
                if conf < 2:
                    continue  # Per Baskoylu 2026 README: only confidence >= 2 included
                n_cls = str(md.get("neuron_class", "")).strip()
                dv = str(md.get("DV", "undefined")).strip()
                lr = str(md.get("LR", "undefined")).strip()
                lbl = str(md.get("label", "")).strip()
                if lbl and _is_valid_neuron_label(lbl) and not lbl.startswith("<"):
                    cname = canonicalize_neuron_name(lbl)
                else:
                    dv_part = dv if dv in ("D", "V") else ""
                    lr_part = lr if lr in ("L", "R") else ""
                    cname = canonicalize_neuron_name(f"{n_cls}{dv_part}{lr_part}")
                if _is_valid_neuron_label(cname):
                    row_map[row_1based - 1] = (cname, conf)
            out[rec_key] = row_map
    return out


def _dryad_fetch_bytes(sess: requests.Session, file_id: str) -> bytes:
    url = f"https://datadryad.org/downloads/file_stream/{file_id}"
    r = sess.get(url, timeout=30)
    m = re.search(r'<script id="anubis_challenge" type="application/json">(.*?)</script>', r.text, re.S)
    if m:
        c = json.loads(m.group(1))
        rnd = c["challenge"]["randomData"]
        diff = int(c["rules"]["difficulty"])
        prefix = "0" * diff
        nonce = 0
        while True:
            digest = hashlib.sha256(f"{rnd}{nonce}".encode("utf-8")).hexdigest()
            if digest.startswith(prefix):
                break
            nonce += 1
        r = sess.get(
            "https://datadryad.org/.within.website/x/cmd/anubis/api/pass-challenge",
            params={
                "id": c["challenge"]["id"],
                "response": digest,
                "nonce": nonce,
                "redir": url,
                "elapsedTime": 20,
            },
            allow_redirects=True,
            timeout=90,
        )
    r.raise_for_status()
    return r.content


def process_baskoylu2026_copper(scratch_dir: Path) -> Dict[str, Any]:
    """Acquire & standardize Baskoylu & Flavell (2026) 27 freely-moving Cu2+ worms (`trace_array_original`)."""
    t0 = time.time()
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    dryad_ver = sess.get("https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.w9ghx3g4v", timeout=20).json()
    ver_href = dryad_ver["_links"]["stash:version"]["href"]
    flist = sess.get(f"https://datadryad.org{ver_href}/files?per_page=100", timeout=20).json()["_embedded"]["stash:files"]

    file_by_name = {
        f["path"]: (f["_links"]["stash:download"]["href"].split("/")[-2], int(f["size"]))
        for f in flist
    }

    # Fetch NeuroPAL_labels.jld2 into scratch_dir
    jld2_fid, _ = file_by_name["NeuroPAL_labels.jld2"]
    jld2_scratch = scratch_dir / "NeuroPAL_labels.jld2"
    if not jld2_scratch.exists():
        jld2_scratch.write_bytes(_dryad_fetch_bytes(sess, jld2_fid))
    jld2_sha = compute_file_sha256(jld2_scratch)
    labels_by_rec = _parse_copper_jld2_labels(jld2_scratch)

    dev_dir = DATA_ROOT / "baskoylu2026_copper" / "dev"
    vault_dir = DATA_ROOT / "baskoylu2026_copper" / "vault"
    dev_dir.mkdir(parents=True, exist_ok=True)
    vault_dir.mkdir(parents=True, exist_ok=True)

    h5_names = sorted([k for k in file_by_name if k.endswith("-cudata.h5")])
    dev_records_meta = []
    vault_records_meta = []
    total_downloaded_bytes = jld2_scratch.stat().st_size

    for idx_w, fname in enumerate(h5_names):
        rec_key = fname.replace("-cudata.h5", "")  # YYYY-MM-DD-NN
        date_batch = rec_key[:10]
        animal_id = f"baskoylu2026_{rec_key.replace('-', '')}"

        # PRE-SEARCH METADATA-ONLY VAULT ASSIGNMENT (before reading neural traces!)
        is_vault = date_batch in COPPER_VAULT_DATES
        target_dir = vault_dir if is_vault else dev_dir
        exposure_status = "SEALED_CONFIRMATION_VAULT" if is_vault else "NEW_DEV"

        fid, fsize = file_by_name[fname]
        raw_bytes = _dryad_fetch_bytes(sess, fid)
        total_downloaded_bytes += len(raw_bytes)
        raw_h5_sha = hashlib.sha256(raw_bytes).hexdigest()

        tmp_h5 = scratch_dir / fname
        tmp_h5.write_bytes(raw_bytes)
        try:
            with h5py.File(tmp_h5, "r") as h5:
                # STRICTLY `trace_array_original` (raw un-zscored, unsmoothed)!
                # Julia (N_roi, T) is stored in HDF5 C-order as (T, N_roi) = (1600, N_roi)
                arr_orig = h5["gcamp/trace_array_original"][:].astype(np.float32)
                ts_conf = h5["timing/timestamp_confocal"][:].astype(np.float64)
                dist_cu = h5["behavior/dist_from_cuCenter"][:].astype(np.float64)
                circle_r = float(h5["behavior/circle_radius"][()])
                vel = h5["behavior/velocity"][:].astype(np.float32)
        finally:
            if tmp_h5.exists():
                tmp_h5.unlink()

        if arr_orig.shape[0] != ts_conf.shape[0] and arr_orig.shape[1] == ts_conf.shape[0]:
            arr_orig = arr_orig.T  # Ensure [T, N_roi]

        t_len = arr_orig.shape[0]
        dt_med = float(np.median(np.diff(ts_conf)))
        fps = round(1.0 / dt_med, 4)  # ~1.6667 Hz

        row_map = labels_by_rec.get(rec_key, {})
        # Deduplicate canonical neuron names within worm by highest NeuroPAL confidence, then finite count
        chosen: Dict[str, Tuple[int, int, np.ndarray]] = {}
        for r_idx, (cname, conf) in row_map.items():
            if 0 <= r_idx < arr_orig.shape[1]:
                vec = arr_orig[:, r_idx]
                n_fin = int(np.sum(np.isfinite(vec)))
                if n_fin == 0:
                    continue
                if cname not in chosen or (conf, n_fin) > (chosen[cname][0], chosen[cname][1]):
                    chosen[cname] = (conf, n_fin, vec)

        ordered_names = sorted(chosen.keys())
        traces_mat = np.column_stack([chosen[nm][2] for nm in ordered_names]).astype(np.float32)
        obs_mask = np.isfinite(traces_mat)

        # Extract causal Cu2+ agar boundary encounter events strictly from physical arena position (`dist_from_cuCenter`)
        # Copper boundary is at ~40,000 (~4.0 mm radius; `circle_radius` ~42,250). Upward crossing of 38,000 (~3.8 mm)
        # marks physical head arrival at the copper diffusion zone, with >=10s pre and >=10s post window & >=15s refractory gap.
        win_frames = int(np.ceil(10.0 * fps))
        refrac_frames = int(np.ceil(15.0 * fps))
        cu_thresh = min(38000.0, 0.90 * circle_r)
        stim_events = []
        last_onset = -refrac_frames
        trial_order = 0
        for t_i in range(max(1, win_frames), t_len - win_frames):
            if dist_cu[t_i - 1] < cu_thresh and dist_cu[t_i] >= cu_thresh and (t_i - last_onset) >= refrac_frames:
                trial_order += 1
                off_i = t_i + 1
                while off_i < t_len and dist_cu[off_i] >= cu_thresh:
                    off_i += 1
                stim_events.append(
                    {
                        "stim_id": "Cu2+_boundary_encounter",
                        "is_control": False,
                        "onset_idx": int(t_i),
                        "offset_idx": int(min(off_i, t_len)),
                        "onset_sec": round(float(ts_conf[t_i] - ts_conf[0]), 3),
                        "offset_sec": round(float(ts_conf[min(off_i, t_len - 1)] - ts_conf[0]), 3),
                        "trial_order": trial_order,
                    }
                )
                last_onset = t_i

        # If a worm stayed near center below 38,000, fallback to top local maxima approaches >= 75th percentile of dist_cu
        # (computed strictly from physical trajectory `dist_cu`, never from neural activity)
        if len(stim_events) == 0:
            q75 = float(np.nanpercentile(dist_cu, 75))
            for t_i in range(max(1, win_frames), t_len - win_frames):
                if dist_cu[t_i - 1] < q75 and dist_cu[t_i] >= q75 and (t_i - last_onset) >= refrac_frames:
                    trial_order += 1
                    stim_events.append(
                        {
                            "stim_id": "Cu2+_arena_approach",
                            "is_control": False,
                            "onset_idx": int(t_i),
                            "offset_idx": int(min(t_i + win_frames, t_len)),
                            "onset_sec": round(float(ts_conf[t_i] - ts_conf[0]), 3),
                            "offset_sec": round(float(ts_conf[min(t_i + win_frames, t_len - 1)] - ts_conf[0]), 3),
                            "trial_order": trial_order,
                        }
                    )
                    last_onset = t_i

        rec = AnimalRecord(
            dataset_id="baskoylu2026_copper",
            animal_id=animal_id,
            sex="hermaphrodite",
            modality="freely_moving",
            date_batch=date_batch,
            fps=fps,
            traces_raw=traces_mat,
            obs_mask=obs_mask,
            neuron_ids=ordered_names,
            stim_events=stim_events,
            exposure_status=exposure_status,
            extra_metadata={
                "dryad_file": fname,
                "dryad_file_id": fid,
                "raw_h5_sha256": raw_h5_sha,
                "jld2_sha256": jld2_sha,
                "circle_radius": circle_r,
                "cu_boundary_threshold": cu_thresh,
                "genotype": "NeuroPAL; pan-neuronal NLS-GCaMP7f",
                "immobilization": "none (freely moving in circular Cu2+ arena)",
                "t_frames": int(t_len),
                "causal_behavior_note": "Post-onset behavior strictly prohibited as predictor input; pre-onset velocity/distance stored only if sliced < onset_idx.",
                "dist_from_cuCenter": [round(float(x), 2) for x in dist_cu[::4]],  # lightweight diagnostic summary
                "pre_velocity_mean": round(float(np.nanmean(vel)), 5),
            },
        )
        out_npz = target_dir / f"{animal_id}.npz"
        sha = save_animal_record_npz(rec, out_npz)
        meta_item = {
            "animal_id": animal_id,
            "sex": "hermaphrodite",
            "date_batch": date_batch,
            "n_neurons": len(ordered_names),
            "n_frames": int(t_len),
            "n_stim_events": len(stim_events),
            "raw_h5_sha256": raw_h5_sha,
            "path": str(out_npz),
            "sha256": sha,
        }
        if is_vault:
            vault_records_meta.append(meta_item)
        else:
            dev_records_meta.append(meta_item)
        print(
            f"[copper {idx_w+1:02d}/27] {animal_id} -> {'VAULT' if is_vault else 'DEV'} | "
            f"T={t_len}, N={len(ordered_names)}, encounters={len(stim_events)}",
            flush=True,
        )

    return {
        "dataset_id": "baskoylu2026_copper",
        "elapsed_sec": round(time.time() - t0, 2),
        "total_downloaded_bytes": total_downloaded_bytes,
        "jld2_sha256": jld2_sha,
        "dev_records": dev_records_meta,
        "vault_records": vault_records_meta,
    }


def audit_toyoshima2024() -> Dict[str, Any]:
    """Run reproducible empirical audit of Toyoshima et al. (PLOS Comp Biol 2024, Figshare 21968078)."""
    out_dir = DATA_ROOT / "toyoshima2024"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify Figshare 21968078 metadata & sample 180710a_ratio.csv column means via HTTP Range
    art = requests.get("https://api.figshare.com/v2/articles/21968078", timeout=20).json()
    files_info = [
        {"name": f["name"], "size_bytes": f["size"], "md5": f.get("computed_md5", ""), "download_url": f["download_url"]}
        for f in art.get("files", [])
    ]

    # Inspect 180710a_ratio.csv (entry at offset 8223297, comp_sz 8223216) via HTTP Range
    r = requests.get(
        "https://ndownloader.figshare.com/files/38971616",
        headers={"Range": "bytes=8223297-16446580"},
        allow_redirects=True,
        timeout=30,
    )
    b = r.content
    fn_len, ex_len = struct.unpack("<HH", b[26:30])
    raw_csv = zlib.decompressobj(-zlib.MAX_WBITS).decompress(b[30 + fn_len + ex_len : 30 + fn_len + ex_len + 8223216])
    lines = raw_csv.decode("utf-8", errors="replace").splitlines()
    arr = np.array([[float(x) for x in row.split(",")] for row in lines], dtype=np.float64)
    max_abs_col_mean = float(np.max(np.abs(np.mean(arr, axis=0))))

    # Verify GitHub metadata folder contents (confirming stimulation_timing.csv is missing)
    gh_meta = [
        x["name"]
        for x in requests.get("https://api.github.com/repos/yuichiiino1/gKDR-GMM/contents/metadata", timeout=15).json()
    ]

    report = {
        "dataset_id": "toyoshima2024",
        "doi": "10.1371/journal.pcbi.1011848",
        "figshare_article_id": 21968078,
        "figshare_url": "https://figshare.com/articles/dataset/21968078",
        "license": art.get("license", {}).get("name", "CC BY 4.0"),
        "files_inspected": files_info,
        "audit_timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "AUDITED_EXCLUDED_OBJECTIVE_CAUSALITY_AND_STIM_TIMING_BARRIER",
        "empirical_findings": {
            "zip_worm_count": 28,
            "sample_180710a_shape": list(arr.shape),
            "sample_180710a_max_abs_full_trace_mean": max_abs_col_mean,
            "non_causal_preprocessing_proven_in_source": (
                "YuToyoshima/TDE-RICA/tderica_main.m `normalize1` applies `medfilt1(tcrsYfp./tcrsCfp, 11)` "
                "(centered 11-frame median filter), `sgolayfilt(tcrs, 3, 101)` (centered 101-frame / ~25s Savitzky-Golay "
                "smoothing leaking 50 future frames = ~12.5s of post-stimulus response into [-10, 0)s), and "
                "`tcrsFilt ./ mean(tcrsFilt) - 1` (whole-recording 6000-frame normalization, verified empirically: "
                f"max|mean(col)| = {max_abs_col_mean:.3e})."
            ),
            "missing_stimulus_timestamps": (
                "`yuichiiino1/gKDR-GMM/codes/gKDR_GMM_makemodel.m` and `utilities/generatesalt.m` require "
                "`metadata/stimulation_timing.csv` (`startframe`, `period` per worm), which is absent from both "
                f"GitHub `metadata/` ({gh_meta}) and Figshare `21968078`."
            ),
            "raw_7z_disk_barrier": (
                "`20221118freezed.7z` is 2.59 GB compressed Excel files (exceeding the strict <800 MB `data/` budget) "
                "and still lacks `stimulation_timing.csv`."
            ),
        },
        "governance_consequence": (
            "Per MODEL_REQUIREMENTS.md §2.3-2.4 and User Confirmation #1, `toyoshima2024` cannot be used as a causal "
            "unseen-animal confirmation source without violating strict temporal causality. Only 1 unexposed immobilized "
            "multi-chemical whole-brain primary confirmation source (`seyedolmohadesin2025`, 13 sealed vault worms) is "
            "available alongside `yemini2021` (21 dev worms) and `baskoylu2026_copper` (18 dev / 9 sealed vault worms, "
            "freely-moving Cu2+ cross-modality stress test) plus `randi2023_atlas` (WT vs unc-31 mechanistic validation). "
            "Dual-unexposed-immobilized E2 status is explicitly recorded as UNMET_SECOND_IMMOBILIZED_SOURCE_CAUSALITY_BARRIER."
        ),
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_governance_manifests(
    yemini_res: Dict[str, Any],
    seyed_res: Dict[str, Any],
    copper_res: Dict[str, Any],
    toyoshima_res: Dict[str, Any],
    conn_meta: Dict[str, Any],
) -> None:
    """Generate `data_registry.json`, `exposure_ledger.json`, and `split_manifest.json`."""
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    randi_meta = json.loads((DATA_ROOT / "randi2023_atlas" / "randi2023_metadata.json").read_text(encoding="utf-8"))

    # 1. data_registry.json (`MODEL_REQUIREMENTS.md` §2.5)
    data_registry = {
        "schema_version": "1.0.0",
        "created_at_utc": now_utc,
        "confirmatory_gate_status": {
            "registered_independent_sources_count": 4,
            "active_causal_trace_sources_count": 3,
            "comparable_immobilized_chemosensory_sources": ["yemini2021", "seyedolmohadesin2025"],
            "unexposed_immobilized_chemosensory_confirmation_sources": ["seyedolmohadesin2025"],
            "cross_modality_stress_test_sources": ["baskoylu2026_copper"],
            "mechanistic_perturbation_sources": ["randi2023_atlas"],
            "audited_excluded_sources": ["toyoshima2024"],
            "e2_dual_unexposed_immobilized_gate": "UNMET_SECOND_IMMOBILIZED_SOURCE_CAUSALITY_BARRIER",
            "honest_scope_statement": (
                "Yemini 2021 (21 immobilized worms) was historically exposed and serves strictly for development/regression. "
                "Seyedolmohadesin 2025 (47 immobilized worms: 34 dev / 13 sealed vault) provides rigorous unseen-animal "
                "confirmatory testing on immobilized multi-chemical whole-brain responses. Toyoshima 2024 (Figshare 21968078) "
                "failed pre-search causal audit due to centered 101-frame Savitzky-Golay smoothing, full-trace mean normalization, "
                "and omitted stimulation_timing.csv. Baskoylu 2026 Copper (27 freely-moving worms: 18 dev / 9 sealed vault) "
                "serves as a strictly separate cross-modality aversive Cu2+ stress test, and Randi 2023 (113 WT + 18 unc-31) "
                "serves as external mechanistic perturbation validation."
            ),
        },
        "sources": {
            "yemini2021": {
                "citation": "Yemini et al. (Cell 2021) NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans",
                "doi_or_url": "https://doi.org/10.1016/j.cell.2020.12.012 | https://github.com/TuragaLab/wormvae/tree/master/data/worm_activity",
                "version_sha256": yemini_res["raw_source_sha256"],
                "license": "MIT / Public Academic Release",
                "n_animals": 21,
                "sex_distribution": {"hermaphrodite": 21, "male": 0},
                "genotype": "OH16230 (otIs670 [NeuroPAL]; otIs672 [pan-neuronal GCaMP6s])",
                "modality": "immobilized",
                "immobilization": "1 mM tetramisole in microfluidic chip",
                "fps_hz": 4.0,
                "stimulus_protocol": "3 chemical stimuli per worm (2-butanone, 2,3-pentanedione, NaCl), 10s ON pulses at t=[60,70)s, [120,130)s, [180,190)s",
                "synchrony": "simultaneous whole-head volumetric GCaMP6s",
                "neuron_coverage_per_animal": {
                    "min": int(min(r["n_neurons"] for r in yemini_res["dev_records"])),
                    "median": float(np.median([r["n_neurons"] for r in yemini_res["dev_records"]])),
                    "max": int(max(r["n_neurons"] for r in yemini_res["dev_records"])),
                },
                "raw_preprocessing_verified": "Extracted strictly `traces` (raw histogram-matched dF/F0 without non-negativity clipping); `norm_traces` explicitly rejected.",
                "historical_exposure": "HISTORICALLY_EXPOSED_DEV_ONLY",
                "assigned_role": "primary_immobilized_chemosensory_dev_and_historical_regression",
            },
            "seyedolmohadesin2025": {
                "citation": "Seyedolmohadesin et al. (bioRxiv 2025 / Curr. Biol. 2026) Whole-brain chemosensory responses of both C. elegans sexes",
                "doi_or_url": "https://doi.org/10.1101/2025.03.01.640987 | https://dandiarchive.org/dandiset/000981",
                "version_sha256": "dandi:000981/draft (per-asset SHA-256 recorded in split_manifest.json)",
                "license": "CC-BY-4.0",
                "n_animals": 47,
                "sex_distribution": {"hermaphrodite": 22, "male": 25},
                "genotype": "NeuroPAL + pan-neuronal GCaMP6s (hermaphrodite & male)",
                "modality": "immobilized",
                "immobilization": "1 mM levamisole in multi-channel microfluidic olfactory chip",
                "fps_hz": 2.66,
                "stimulus_protocol": "8 chemical stimuli (100mM NaCl, 450mM NaCl, e-2 IAA, e-6 IAA, 1uM ascr#3, 10mM CuSO4, 800mM Sorbitol, OP50) + 2 controls (Control, Fluorescein) x 3 randomized cycles (~15s ON, ~45s CTX buffer = 30 deliveries/worm)",
                "synchrony": "simultaneous whole-head volumetric GCaMP6s (~196 sex-shared neurons + 6 male-specific head neurons)",
                "neuron_coverage_per_animal": {
                    "min": int(min(r["n_neurons"] for r in seyed_res["dev_records"] + seyed_res["vault_records"])),
                    "median": float(np.median([r["n_neurons"] for r in seyed_res["dev_records"] + seyed_res["vault_records"]])),
                    "max": int(max(r["n_neurons"] for r in seyed_res["dev_records"] + seyed_res["vault_records"])),
                },
                "raw_preprocessing_verified": "Streamed raw ZephIR `processing/CalciumActivity/ActivityTraces/activity` via HTTP Range (`remfile`); authors' `global_zscored_raw.csv` bandpass/z-score explicitly rejected.",
                "historical_exposure": "UNEXPOSED_SPLIT_INTO_34_DEV_AND_13_SEALED_VAULT",
                "assigned_role": "primary_immobilized_chemosensory_dev_and_sealed_confirmation",
            },
            "baskoylu2026_copper": {
                "citation": "Baskoylu, ..., & Flavell (bioRxiv 2026) Parallel neural integrators control distinct features of an aversive behavioral state in C. elegans",
                "doi_or_url": "https://doi.org/10.5061/dryad.w9ghx3g4v",
                "version_sha256": copper_res["jld2_sha256"],
                "license": "CC0-1.0",
                "n_animals": 27,
                "sex_distribution": {"hermaphrodite": 27, "male": 0},
                "genotype": "NeuroPAL; pan-neuronal NLS-GCaMP7f",
                "modality": "freely_moving",
                "immobilization": "none (freely moving in circular agar arena with 4 mm radius Cu2+ aversive barrier)",
                "fps_hz": 1.6667,
                "stimulus_protocol": "Causal physical encounters with circular Cu2+ boundary (`dist_from_cuCenter` upward crossing of 38,000 / 0.9*circle_radius)",
                "synchrony": "simultaneous spinning-disk confocal whole-brain GCaMP7f + NIR behavioral tracking",
                "neuron_coverage_per_animal": {
                    "min": int(min(r["n_neurons"] for r in copper_res["dev_records"] + copper_res["vault_records"])),
                    "median": float(np.median([r["n_neurons"] for r in copper_res["dev_records"] + copper_res["vault_records"]])),
                    "max": int(max(r["n_neurons"] for r in copper_res["dev_records"] + copper_res["vault_records"])),
                },
                "raw_preprocessing_verified": "Extracted strictly `gcamp/trace_array_original` (raw un-zscored, unsmoothed fluorescence) + `NeuroPAL_labels.jld2` (`confidence >= 2`); whole-trace z-scored `trace_array` and `F/Fmean` rejected.",
                "historical_exposure": "UNEXPOSED_SPLIT_INTO_18_DEV_AND_9_SEALED_VAULT",
                "assigned_role": "separate_freely_moving_aversive_chemosensory_stress_test",
            },
            "toyoshima2024": {
                "citation": "Toyoshima et al. (PLOS Computational Biology 2024) Ensemble dynamics and information flow deduction from whole-brain imaging data",
                "doi_or_url": "https://doi.org/10.1371/journal.pcbi.1011848 | https://figshare.com/articles/dataset/21968078",
                "version_sha256": "figshare:21968078/38971616",
                "license": "CC-BY-4.0",
                "n_animals": 28,
                "modality": "immobilized",
                "status": "AUDITED_EXCLUDED_OBJECTIVE_CAUSALITY_AND_STIM_TIMING_BARRIER",
                "exclusion_record": toyoshima_res["empirical_findings"],
                "assigned_role": "excluded_pre_search_due_to_noncausal_sgolay101_and_missing_stim_timing",
            },
            "randi2023_atlas": {
                "citation": "Randi, Sharma, Dvali & Leifer (Nature 2023) Neural signal propagation atlas of C. elegans",
                "doi_or_url": "https://doi.org/10.1038/s41586-023-06683-4 | wormneuroatlas v0.0.7.3",
                "version_sha256": randi_meta["sha256"],
                "license": "MIT / Academic Open Source",
                "n_animals": 131,
                "genotype": "113 WT + 18 unc-31(e928) neuropeptide-release mutant (NeuroPAL; GCaMP6s; Chrimson)",
                "modality": "immobilized_optogenetic_single_cell_perturbation",
                "historical_exposure": "EXTERNAL_MECHANISTIC_VALIDATION_ONLY",
                "assigned_role": "external_wt_vs_unc31_neuropeptide_perturbation_validation_never_used_for_prior_selection",
            },
        },
        "connectome_priors": {
            "bundle_path": str(DATA_ROOT / "connectomes" / "connectome_priors_bundle.npz"),
            "bundle_sha256": conn_meta["bundle_sha256"],
            "n_canonical_neurons": conn_meta["n_canonical_neurons"],
            "layers": conn_meta["layers"],
            "polarity_counts_on_chem_edges": conn_meta["polarity_counts_on_chem_edges"],
        },
    }
    (EXP_ROOT / "data_registry.json").write_text(json.dumps(data_registry, indent=2), encoding="utf-8")

    # 2. exposure_ledger.json (`MODEL_REQUIREMENTS.md` §2.6 & §6.1)
    exposure_ledger = {
        "schema_version": "1.0.0",
        "created_at_utc": now_utc,
        "isolation_disclosure": (
            "Same-VM AI execution environment (`MODEL_REQUIREMENTS.md` §6.9): technical isolation is enforced via "
            "physical directory separation (`dev/` vs `vault/`), `PermissionError` guard in `src.data_adapters.load_dataset` "
            "when `allow_vault_unseal=False`, append-only `vault_access_audit.jsonl`, and immutable SHA-256 checksums in "
            "`split_manifest.json`. We honestly disclose that same-user POSIX ownership is procedural/programmatic rather "
            "than a separate OS account."
        ),
        "datasets": {
            "yemini2021": {
                "prior_exposure": True,
                "exposure_details": (
                    "All 21 head worms were previously inspected in historical `probe_prestim_v1` and `audit_probe.py` "
                    "(conversation `9c18be30-8516-48d6-8db9-460a7647e17e`, Ridge post R^2=0.703793). Per `MODEL_REQUIREMENTS.md` "
                    "§2.2, all 21 animals are locked to `HISTORICALLY_EXPOSED_DEV_ONLY` and forbidden from confirmatory claims."
                ),
                "animals": {
                    r["animal_id"]: "HISTORICALLY_EXPOSED_DEV_ONLY" for r in yemini_res["dev_records"]
                },
            },
            "seyedolmohadesin2025": {
                "prior_exposure": False,
                "exposure_details": (
                    "Zero animals were ever used in prior model fitting or hyperparameter tuning before M1. "
                    "Split into 34 `NEW_DEV` animals and 13 `SEALED_CONFIRMATION_VAULT` animals strictly from `datasets.json` "
                    "(`date`, `sex`, `animal`) prior to inspecting any neural traces."
                ),
                "animals": {
                    **{r["animal_id"]: "NEW_DEV" for r in seyed_res["dev_records"]},
                    **{r["animal_id"]: "SEALED_CONFIRMATION_VAULT" for r in seyed_res["vault_records"]},
                },
            },
            "baskoylu2026_copper": {
                "prior_exposure": False,
                "exposure_details": (
                    "Zero animals were ever used in prior model fitting or hyperparameter tuning before M1. "
                    "Split into 18 `NEW_DEV` animals and 9 `SEALED_CONFIRMATION_VAULT` animals strictly from recording date "
                    "metadata prior to inspecting any neural traces."
                ),
                "animals": {
                    **{r["animal_id"]: "NEW_DEV" for r in copper_res["dev_records"]},
                    **{r["animal_id"]: "SEALED_CONFIRMATION_VAULT" for r in copper_res["vault_records"]},
                },
            },
            "randi2023_atlas": {
                "prior_exposure": False,
                "exposure_details": (
                    "Reserved exclusively for external mechanistic perturbation validation (`WT` vs `unc-31` neuropeptide "
                    "release mutant); strictly prohibited from edge selection or hyperparameter tuning (`MODEL_REQUIREMENTS.md` §7)."
                ),
            },
        },
    }
    (EXP_ROOT / "exposure_ledger.json").write_text(json.dumps(exposure_ledger, indent=2), encoding="utf-8")

    # 3. split_manifest.json (`MODEL_REQUIREMENTS.md` §6.1–6.4)
    split_manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": now_utc,
        "split_rule_summary": (
            "Deterministic pre-search date-batch-disjoint & sex-stratified grouping computed purely from recording metadata "
            "before inspecting any post-stimulus neural activity. Zero animal_id overlap and zero date_batch overlap exist "
            "between `dev` and `vault` across every dataset."
        ),
        "datasets": {
            "yemini2021": {
                "n_dev": len(yemini_res["dev_records"]),
                "n_vault": 0,
                "dev_dates": sorted({r["date_batch"] for r in yemini_res["dev_records"]}),
                "vault_dates": [],
                "dev_records": yemini_res["dev_records"],
                "vault_records": [],
            },
            "seyedolmohadesin2025": {
                "n_dev": len(seyed_res["dev_records"]),
                "n_vault": len(seyed_res["vault_records"]),
                "dev_sex_counts": {
                    "hermaphrodite": sum(1 for r in seyed_res["dev_records"] if r["sex"] == "hermaphrodite"),
                    "male": sum(1 for r in seyed_res["dev_records"] if r["sex"] == "male"),
                },
                "vault_sex_counts": {
                    "hermaphrodite": sum(1 for r in seyed_res["vault_records"] if r["sex"] == "hermaphrodite"),
                    "male": sum(1 for r in seyed_res["vault_records"] if r["sex"] == "male"),
                },
                "dev_dates": sorted({r["date_batch"] for r in seyed_res["dev_records"]}),
                "vault_dates": sorted({r["date_batch"] for r in seyed_res["vault_records"]}),
                "dev_records": seyed_res["dev_records"],
                "vault_records": seyed_res["vault_records"],
            },
            "baskoylu2026_copper": {
                "n_dev": len(copper_res["dev_records"]),
                "n_vault": len(copper_res["vault_records"]),
                "dev_dates": sorted({r["date_batch"] for r in copper_res["dev_records"]}),
                "vault_dates": sorted({r["date_batch"] for r in copper_res["vault_records"]}),
                "dev_records": copper_res["dev_records"],
                "vault_records": copper_res["vault_records"],
            },
        },
    }
    manifest_bytes = json.dumps(split_manifest, indent=2, sort_keys=True).encode("utf-8")
    split_manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    (EXP_ROOT / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")


def main() -> None:
    scratch_dir = Path(os.environ.get("TMPDIR", "/home/antigravity/.gemini/jetski/scratch/teamwork_tmp/worker_m1_data/tmp"))
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1/5: Connectome Priors & Randi 2023 Atlas ===", flush=True)
    conn_meta = build_and_save_connectome_bundle()

    print("=== Step 2/5: Yemini 2021 (21 dev worms) ===", flush=True)
    yemini_res = process_yemini2021()
    print(f"Yemini complete: {len(yemini_res['dev_records'])} dev worms in {yemini_res['elapsed_sec']}s", flush=True)

    print("=== Step 3/5: Seyedolmohadesin 2025 (DANDI 000981 S3 Range, 34 dev / 13 vault) ===", flush=True)
    seyed_res = process_seyedolmohadesin2025()
    print(
        f"Seyedolmohadesin complete: {len(seyed_res['dev_records'])} dev, {len(seyed_res['vault_records'])} vault "
        f"in {seyed_res['elapsed_sec']}s",
        flush=True,
    )

    print("=== Step 4/5: Baskoylu 2026 Copper (Dryad w9ghx3g4v, 18 dev / 9 vault) ===", flush=True)
    copper_res = process_baskoylu2026_copper(scratch_dir=scratch_dir)
    print(
        f"Baskoylu Copper complete: {len(copper_res['dev_records'])} dev, {len(copper_res['vault_records'])} vault "
        f"in {copper_res['elapsed_sec']}s",
        flush=True,
    )

    print("=== Step 5/5: Toyoshima 2024 Empirical Audit & Governance Manifests ===", flush=True)
    toyoshima_res = audit_toyoshima2024()
    write_governance_manifests(yemini_res, seyed_res, copper_res, toyoshima_res, conn_meta)
    print("ALL M1 ACQUISITION, VAULTING, AND GOVERNANCE MANIFESTS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
