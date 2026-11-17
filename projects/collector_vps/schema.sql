-- SQLite schema. Run PRAGMA foreign_keys/busy_timeout on EVERY connection.
-- One local database, one program-owned writer. CSV files are exported views.
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
PRAGMA journal_mode = DELETE;

CREATE TABLE papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi TEXT UNIQUE,                      -- normalized lowercase DOI, no https://doi.org/
    pmid TEXT UNIQUE,
    pmcid TEXT UNIQUE,
    title TEXT NOT NULL,
    publication_year INTEGER,
    journal TEXT,
    landing_url TEXT NOT NULL,
    screening_state TEXT NOT NULL DEFAULT 'unreviewed'
      CHECK(screening_state IN ('unreviewed','candidate','excluded','review')),
    reason_code TEXT,
    evidence_snapshot_path TEXT,
    evidence_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key TEXT NOT NULL UNIQUE,        -- repository + record/accession + concrete version
    repository TEXT NOT NULL,
    accession TEXT,
    version TEXT NOT NULL,                 -- repository version, commit, or dated observed snapshot
    landing_url TEXT NOT NULL,
    description TEXT,
    species_status TEXT NOT NULL DEFAULT 'unknown'
      CHECK(species_status IN ('yes','no','unknown')),
    behavior_series_status TEXT NOT NULL DEFAULT 'unknown'
      CHECK(behavior_series_status IN ('yes','no','unknown')),
    collection_decision TEXT NOT NULL DEFAULT 'review'
      CHECK(collection_decision IN ('download_candidate','metadata_only','exclude','review')),
    declared_record_count INTEGER,         -- author-reported, NOT locally measured count
    license_text TEXT,
    license_url TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE paper_datasets (
    paper_id INTEGER NOT NULL REFERENCES papers,
    dataset_id INTEGER NOT NULL REFERENCES datasets,
    relation TEXT NOT NULL CHECK(relation IN ('primary','reused','mentioned','unknown')),
    evidence_url TEXT NOT NULL,
    evidence_snapshot_path TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    PRIMARY KEY(paper_id,dataset_id)
);

CREATE TABLE files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key TEXT NOT NULL UNIQUE,        -- repository/version/file identifier, NOT signed URL
    landing_url TEXT NOT NULL,
    download_url TEXT,                     -- resolved by connector, can expire; never used as ID
    original_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('data','documentation','code','video','unknown')),
    file_state TEXT NOT NULL DEFAULT 'discovered'
      CHECK(file_state IN ('discovered','queued','verified','review','unavailable')),
    declared_bytes INTEGER CHECK(declared_bytes IS NULL OR declared_bytes >= 0),
    downloaded_bytes INTEGER CHECK(downloaded_bytes IS NULL OR downloaded_bytes >= 0),
    sha256 TEXT CHECK(sha256 IS NULL OR length(sha256)=64),
    object_relpath TEXT,                   -- objects/sha256/ab/<full hash>; shared by identical bytes
    published_checksum_algorithm TEXT,
    published_checksum TEXT,
    published_checksum_matches INTEGER CHECK(published_checksum_matches IN (0,1)),
    content_type_observed TEXT,
    magic_type_observed TEXT,
    etag TEXT,
    last_modified TEXT,
    downloaded_at TEXT,
    evidence_snapshot_path TEXT,
    verified_at TEXT,
    CHECK(file_state <> 'verified' OR
      (sha256 IS NOT NULL AND object_relpath IS NOT NULL AND downloaded_bytes IS NOT NULL AND verified_at IS NOT NULL))
);
CREATE INDEX files_content_hash ON files(sha256);

CREATE TABLE dataset_files (
    dataset_id INTEGER NOT NULL REFERENCES datasets,
    file_id INTEGER NOT NULL REFERENCES files,
    PRIMARY KEY(dataset_id,file_id)
);

CREATE TABLE recordings (
    recording_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets,
    source_record_key TEXT NOT NULL,
    animal_id_source TEXT,                 -- NULL if unavailable; never invented from recording_id
    animal_identity_status TEXT NOT NULL DEFAULT 'unknown'
      CHECK(animal_identity_status IN ('source_verified','inferred','unknown')),
    split_group_id TEXT,                   -- conservative grouping; never proves independent animals
    split_group_basis TEXT CHECK(split_group_basis IN ('animal','session','video','recording','unknown')),
    species TEXT,
    strain TEXT,
    sex TEXT,
    stage_source TEXT,
    stage_conflict INTEGER NOT NULL DEFAULT 0 CHECK(stage_conflict IN (0,1)),
    condition_json TEXT NOT NULL DEFAULT '{}',
    representation TEXT NOT NULL CHECK(representation IN ('kinematics','eigenworm','body_shape','position2d','mixed','unknown')),
    representation_key TEXT NOT NULL,       -- includes units, convention and basis identity when relevant
    timing_kind TEXT NOT NULL CHECK(timing_kind IN ('timestamps','documented_rate','frame_only','unaligned')),
    timing_evidence TEXT,
    sampling_hz REAL CHECK(sampling_hz IS NULL OR sampling_hz > 0),
    duration_s REAL CHECK(duration_s IS NULL OR duration_s >= 0),
    n_rows INTEGER CHECK(n_rows IS NULL OR n_rows >= 0),
    n_channels INTEGER CHECK(n_channels IS NULL OR n_channels > 0),
    n_complete_rows INTEGER,
    n_gap_boundaries INTEGER,
    numeric_sha256 TEXT,
    duplicate_of_recording_id INTEGER REFERENCES recordings,
    qc_state TEXT NOT NULL DEFAULT 'pending'
      CHECK(qc_state IN ('pending','usable','partial','review','invalid','duplicate')),
    qc_reason_codes_json TEXT NOT NULL DEFAULT '[]',
    parsed_relpath TEXT,
    parser_name TEXT,
    parser_version TEXT,
    parsed_sha256 TEXT,
    last_validated_at TEXT,
    UNIQUE(dataset_id,source_record_key),
    CHECK(n_complete_rows IS NULL OR (n_complete_rows >= 0 AND n_complete_rows <= n_rows)),
    CHECK(timing_kind <> 'frame_only' OR (sampling_hz IS NULL AND duration_s IS NULL)),
    CHECK(qc_state <> 'usable' OR
      (parsed_relpath IS NOT NULL AND n_rows > 0 AND n_channels > 0 AND parsed_sha256 IS NOT NULL))
);

CREATE TABLE recording_sources (
    recording_id INTEGER NOT NULL REFERENCES recordings,
    file_id INTEGER NOT NULL REFERENCES files,
    source_locator TEXT NOT NULL,          -- member name, HDF5 dataset key, MAT cell, table/row range
    source_role TEXT NOT NULL CHECK(source_role IN ('behavior','time','metadata','basis','neural')),
    PRIMARY KEY(recording_id,file_id,source_locator,source_role)
);

CREATE TABLE channels (
    recording_id INTEGER NOT NULL REFERENCES recordings,
    column_index INTEGER NOT NULL CHECK(column_index >= 1),
    column_name TEXT NOT NULL,             -- c0001, c0002, ...; stable within recording
    source_name TEXT NOT NULL,
    quantity TEXT NOT NULL,
    source_unit TEXT,
    stored_unit TEXT,
    unit_status TEXT NOT NULL CHECK(unit_status IN ('documented','source_scale','unknown','conflict')),
    sign_convention TEXT,
    basis_id TEXT,
    conversion_json TEXT NOT NULL DEFAULT '{"operation":"identity"}',
    evidence_snapshot_path TEXT,
    evidence_text TEXT,
    n_finite INTEGER,
    is_constant INTEGER CHECK(is_constant IN (0,1)),
    PRIMARY KEY(recording_id,column_index),
    UNIQUE(recording_id,column_name)
);

CREATE TABLE jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dedupe_key TEXT NOT NULL UNIQUE,        -- action + target + source version + pipeline version
    action TEXT NOT NULL CHECK(action IN ('discover','screen','resolve','download','validate','parse','export')),
    target_type TEXT NOT NULL CHECK(target_type IN ('query','paper','dataset','file','recording','catalog')),
    target_id TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued'
      CHECK(status IN ('queued','running','retry_wait','done','review','failed','cancelled')),
    priority INTEGER NOT NULL DEFAULT 50,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(max_attempts > 0),
    next_run_at TEXT NOT NULL,
    lease_token TEXT,
    lease_until TEXT,
    heartbeat_at TEXT,
    reserved_bytes INTEGER NOT NULL DEFAULT 0 CHECK(reserved_bytes >= 0),
    checkpoint_json TEXT NOT NULL DEFAULT '{}',
    last_error_code TEXT,
    last_error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(status <> 'running' OR (lease_token IS NOT NULL AND lease_until IS NOT NULL))
);
CREATE INDEX jobs_due ON jobs(status,next_run_at,priority);

-- CSV publication happens after consistent snapshot export, by one writer only.
CREATE VIEW v_paper_overview AS
SELECT p.paper_id,p.doi,p.pmid,p.title,p.publication_year,p.screening_state,
 (SELECT count(*) FROM paper_datasets pd WHERE pd.paper_id=p.paper_id) AS linked_datasets
FROM papers p;

CREATE VIEW v_dataset_overview AS
SELECT d.dataset_id,d.source_key,d.repository,d.version,d.collection_decision,d.declared_record_count,
 (SELECT count(*) FROM recordings r WHERE r.dataset_id=d.dataset_id) AS observed_record_count,
 (SELECT sum(r.n_rows) FROM recordings r WHERE r.dataset_id=d.dataset_id AND r.duplicate_of_recording_id IS NULL AND r.qc_state IN ('usable','partial')) AS observed_rows
FROM datasets d;

CREATE VIEW v_distinct_objects AS
SELECT sha256,object_relpath,MAX(downloaded_bytes) AS size_bytes
FROM files WHERE file_state='verified' GROUP BY sha256,object_relpath;
