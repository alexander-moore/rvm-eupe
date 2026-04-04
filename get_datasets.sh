#!/usr/bin/env bash
# =============================================================================
# get_datasets.sh — Reproducible dataset downloader for RVM-EUPE
# =============================================================================
#
# Downloads all training and evaluation datasets needed to reproduce the
# RVM-EUPE paper experiments.
#
# USAGE
#   bash get_datasets.sh [OPTIONS] [DATASETS...]
#
# DATASETS (default: all automatable ones)
#   training    Download all training datasets
#   eval        Download all evaluation datasets
#   all         Everything (training + eval, automatable only)
#   davis       DAVIS 2017 480p          [eval,   ~2.3 GB,  free]
#   kinetics700 Kinetics-700             [train,  ~450 GB,  free]
#   kinetics400 Kinetics-400             [eval,   ~140 GB,  free]
#   perception  Perception Test          [eval,   ~100 GB,  free]
#   ssv2        Something-Something-v2   [train+eval, ~19 GB, REGISTRATION REQUIRED]
#   jhmdb       JHMDB keypoints          [eval,   ~13 GB,   REGISTRATION REQUIRED]
#
# OPTIONS
#   --data-dir DIR     Where to store datasets (default: ~/data)
#   --parallel N       Parallel download connections (default: 16)
#   --dry-run          Print what would be downloaded, don't download
#   --skip-index       Skip building index JSON files after download
#
# GATED DATASETS (require manual registration before running)
#   SSv2:  Register at https://www.qualcomm.com/developer/software/something-something-data-set
#          Then: SSV2_DOWNLOAD_DIR=/path/to/parts bash get_datasets.sh ssv2
#
#   JHMDB: Register at http://jhmdb.is.tue.mpg.de/
#          Then: JHMDB_DOWNLOAD_URL=<url-from-email> bash get_datasets.sh jhmdb
#
# EXAMPLES
#   # Download everything automatable (recommended first run)
#   bash get_datasets.sh all
#
#   # Download to a custom location with more parallelism
#   bash get_datasets.sh --data-dir /mnt/storage --parallel 32 all
#
#   # Evaluation sets only (no training data)
#   bash get_datasets.sh eval
#
#   # Resume a previous run (all downloads are resumable)
#   bash get_datasets.sh all
#
#   # After SSv2 registration — point to downloaded zip directory
#   SSV2_DOWNLOAD_DIR=~/Downloads/ssv2_parts bash get_datasets.sh ssv2
#
# RESUME SUPPORT
#   All downloads are resumable. Re-running the script will skip completed
#   files and continue any interrupted downloads.
#
# AFTER DOWNLOAD
#   Run once all datasets are present:
#     python get_datasets.py --build-index --data-dir ~/data
#   Then update configs/data/mixed.yaml with the generated index paths.
# =============================================================================
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-$HOME/data}"
PARALLEL="${PARALLEL:-16}"
DRY_RUN=false
SKIP_INDEX=false
REQUESTED=()

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)  DATA_DIR="$2";  shift 2 ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true;   shift   ;;
        --skip-index) SKIP_INDEX=true; shift  ;;
        --help|-h)
            sed -n '/^# USAGE/,/^# =\{10\}/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        all)      REQUESTED+=(davis kinetics700 kinetics400 perception); shift ;;
        training) REQUESTED+=(kinetics700 ssv2);                          shift ;;
        eval)     REQUESTED+=(davis kinetics400 perception jhmdb ssv2);   shift ;;
        *)        REQUESTED+=("$1"); shift ;;
    esac
done

# Default: all automatable datasets
[[ ${#REQUESTED[@]} -eq 0 ]] && REQUESTED=(davis kinetics700 kinetics400 perception)

# ─── Helpers ──────────────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

log()     { echo -e "${BOLD}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()      { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
err()     { echo -e "${RED}✗${NC} $*" >&2; }
section() { echo -e "\n${BOLD}━━━ $* ━━━${NC}"; }

require_tool() {
    if ! command -v "$1" &>/dev/null; then
        err "Required tool not found: $1"
        echo "  Install with: $2"
        return 1
    fi
}

# Check if a file is fully downloaded by comparing Content-Length
is_complete() {
    local file="$1" url="$2"
    [[ -f "$file" ]] || return 1
    local remote_size local_size
    remote_size=$(curl -sI "$url" | grep -i '^content-length' | tail -1 | awk '{print $2}' | tr -d $'\r\n ')
    local_size=$(stat -c%s "$file" 2>/dev/null || echo 0)
    [[ -n "$remote_size" && "$local_size" == "$remote_size" ]]
}

# Download a single large file — aria2c preferred, wget fallback
download() {
    local url="$1" dest="$2" label="${3:-}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would download: $url → $dest"
        return
    fi

    if is_complete "$dest" "$url"; then
        ok "${label:-$(basename "$dest")} already complete"
        return
    fi

    local dest_dir
    dest_dir=$(dirname "$dest")
    mkdir -p "$dest_dir"

    log "Downloading ${label:-$(basename "$dest")}..."

    if command -v aria2c &>/dev/null; then
        aria2c \
            --max-connection-per-server="$PARALLEL" \
            --split="$PARALLEL" \
            --min-split-size=10M \
            --continue=true \
            --console-log-level=warn \
            --summary-interval=30 \
            --out="$(basename "$dest")" \
            --dir="$dest_dir" \
            "$url"
    else
        wget -c --show-progress -q "$url" -O "$dest"
    fi
    ok "$(basename "$dest") complete ($(du -sh "$dest" | cut -f1))"
}

# Download a list of URLs in parallel (for Kinetics tarballs)
download_list() {
    local list_url="$1" dest_dir="$2" label="$3"
    mkdir -p "$dest_dir"

    local list_file
    list_file=$(mktemp)
    wget -q "$list_url" -O "$list_file"
    local total
    total=$(grep -c . "$list_file" || echo 0)

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would download $total files from $list_url → $dest_dir"
        rm "$list_file"
        return
    fi

    local already
    already=$(ls "$dest_dir"/*.tar.gz 2>/dev/null | wc -l || echo 0)
    if [[ "$already" -ge "$total" ]]; then
        ok "$label already complete ($already / $total tars)"
        rm "$list_file"
        return
    fi

    log "$label: $already / $total tars present, downloading remainder ($PARALLEL parallel)..."

    python3 - "$list_file" "$dest_dir" "$PARALLEL" <<'PYEOF'
import sys, subprocess, concurrent.futures
from pathlib import Path

list_file, out_dir, parallel = sys.argv[1], sys.argv[2], int(sys.argv[3])
urls = [u.strip() for u in open(list_file) if u.strip()]

def download_one(url):
    fname = Path(out_dir) / Path(url).name
    if fname.exists() and fname.stat().st_size > 100_000:
        return "skip"
    r = subprocess.run(
        ["wget", "-q", "-c", url, "-P", out_dir, "--no-clobber"],
        capture_output=True, timeout=600
    )
    return "ok" if r.returncode == 0 else "fail"

ok = skip = fail = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
    futures = {pool.submit(download_one, u): u for u in urls}
    for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
        r = fut.result()
        if   r == "ok":   ok   += 1
        elif r == "skip": skip += 1
        else:             fail += 1
        if i % 100 == 0 or i == len(urls):
            print(f"  {i}/{len(urls)}  ok={ok}  skip={skip}  fail={fail}", flush=True)
print(f"Done: {ok+skip}/{len(urls)} tars  ({fail} failed)")
PYEOF
    rm "$list_file"
}

extract_tars() {
    local tar_dir="$1" out_dir="$2" label="$3"
    [[ "$DRY_RUN" == true ]] && return
    mkdir -p "$out_dir"
    local tars
    tars=$(find "$tar_dir" -maxdepth 1 -name '*.tar.gz' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$tars" -eq 0 ]]; then warn "No tars to extract in $tar_dir"; return; fi
    log "Extracting $tars tars ($label)..."
    find "$tar_dir" -maxdepth 1 -name '*.tar.gz' | while read -r tar; do
        tar xzf "$tar" -C "$out_dir" --strip-components=1 2>/dev/null || \
        tar xzf "$tar" -C "$out_dir" 2>/dev/null || true
    done
    ok "Extraction complete: $out_dir"
}

# ─── Dependency check ─────────────────────────────────────────────────────────
section "Checking dependencies"

if ! command -v aria2c &>/dev/null; then
    warn "aria2c not found — using wget (slower for large files)"
    warn "Install with: conda install -c conda-forge aria2"
else
    ok "aria2c $(aria2c --version | head -1 | awk '{print $3}')"
fi

require_tool wget  "apt install wget / brew install wget" || exit 1
require_tool curl  "apt install curl / brew install curl" || exit 1
require_tool python3 "conda install python"              || exit 1

mkdir -p "$DATA_DIR"
log "Data directory: $DATA_DIR"
log "Parallel connections: $PARALLEL"
log "Datasets requested: ${REQUESTED[*]}"

# ─── Storage estimate ─────────────────────────────────────────────────────────
FREE_GB=$(df -BG "$DATA_DIR" | tail -1 | awk '{print $4}' | tr -d G)
log "Free space: ${FREE_GB} GB"
if [[ "$FREE_GB" -lt 50 ]]; then
    err "Less than 50 GB free — aborting. At least 250 GB recommended."
    exit 1
fi

# =============================================================================
# Dataset: DAVIS 2017
# =============================================================================
download_davis() {
    section "DAVIS 2017 (~2.3 GB)"
    local out="$DATA_DIR/davis"
    mkdir -p "$out"

    local base="https://data.vision.ee.ethz.ch/csergi/share/davis"
    download "$base/DAVIS-2017-trainval-480p.zip"  "$out/DAVIS-2017-trainval-480p.zip"  "trainval 480p"
    download "$base/DAVIS-2017-test-dev-480p.zip"  "$out/DAVIS-2017-test-dev-480p.zip"  "test-dev 480p"

    if [[ ! -d "$out/DAVIS/JPEGImages" ]]; then
        log "Extracting DAVIS..."
        unzip -q -o "$out/DAVIS-2017-trainval-480p.zip" -d "$out"
        unzip -q -o "$out/DAVIS-2017-test-dev-480p.zip"  -d "$out"
        ok "DAVIS extracted"
    else
        ok "DAVIS already extracted"
    fi

    ok "DAVIS 2017 ready at $out/DAVIS/"
}

# =============================================================================
# Dataset: Kinetics-700
# =============================================================================
download_kinetics700() {
    section "Kinetics-700 (~450 GB compressed, ~600 GB extracted)"
    local out="$DATA_DIR/kinetics700"
    local tars="$out/_tars"
    mkdir -p "$out" "$tars"

    local base="https://s3.amazonaws.com/kinetics/700_2020"
    download_list "$base/train/k700_2020_train_path.txt" "$tars/train" "K700 train"
    download_list "$base/val/k700_2020_val_path.txt"     "$tars/val"   "K700 val"

    # Annotations
    if [[ ! -f "$out/train.csv" ]]; then
        log "Downloading K700 annotations..."
        local ann_tar="$tars/kinetics700_2020.tar.gz"
        download "https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020.tar.gz" \
                 "$ann_tar" "K700 annotations"
        tar xzf "$ann_tar" -C "$out" 2>/dev/null || true
    fi

    extract_tars "$tars/train" "$out/train" "K700 train"
    extract_tars "$tars/val"   "$out/val"   "K700 val"

    ok "Kinetics-700 ready at $out"
}

# =============================================================================
# Dataset: Kinetics-400
# =============================================================================
download_kinetics400() {
    section "Kinetics-400 (~140 GB compressed)"
    local out="$DATA_DIR/kinetics400"
    local tars="$out/_tars"
    mkdir -p "$out" "$tars"

    local base="https://s3.amazonaws.com/kinetics/400"
    download_list "$base/train/k400_train_path.txt" "$tars/train" "K400 train"
    download_list "$base/val/k400_val_path.txt"     "$tars/val"   "K400 val"

    extract_tars "$tars/train" "$out/train" "K400 train"
    extract_tars "$tars/val"   "$out/val"   "K400 val"

    ok "Kinetics-400 ready at $out"
}

# =============================================================================
# Dataset: Perception Test
# =============================================================================
download_perception() {
    section "Perception Test (~100 GB)"
    local out="$DATA_DIR/perception_test"
    mkdir -p "$out"

    local base="https://storage.googleapis.com/dm-perception-test"

    # Annotations (small, fast)
    download "$base/zip_data/train_annotations.zip" "$out/train_annotations.zip" "PT train annotations"
    download "$base/zip_data/valid_annotations.zip" "$out/valid_annotations.zip" "PT valid annotations"

    # Point tracking ID lists
    wget -q -c "$base/misc/point_tracking_train_id_list.csv" -O "$out/point_tracking_train_ids.csv" 2>/dev/null || true
    wget -q -c "$base/misc/point_tracking_valid_id_list.csv" -O "$out/point_tracking_valid_ids.csv" 2>/dev/null || true

    # Extract annotations
    [[ ! -f "$out/all_train.json" ]] && unzip -q -o "$out/train_annotations.zip" -d "$out" 2>/dev/null || true
    [[ ! -f "$out/all_valid.json" ]] && unzip -q -o "$out/valid_annotations.zip" -d "$out" 2>/dev/null || true

    # Videos
    download "$base/zip_data/train_videos.zip" "$out/train_videos.zip" "PT train videos (26.5 GB)"
    download "$base/zip_data/valid_videos.zip" "$out/valid_videos.zip" "PT valid videos (70.2 GB)"

    # Extract videos (only once zip is fully downloaded)
    if [[ "$DRY_RUN" == false ]]; then
        if is_complete "$out/train_videos.zip" "$base/zip_data/train_videos.zip"; then
            if [[ ! -d "$out/train" || -z "$(ls -A "$out/train" 2>/dev/null)" ]]; then
                log "Extracting PT train videos..."
                unzip -q -o "$out/train_videos.zip" -d "$out/train"
            else
                ok "PT train videos already extracted"
            fi
        else
            warn "train_videos.zip still downloading — extraction will happen on next run"
        fi
        if is_complete "$out/valid_videos.zip" "$base/zip_data/valid_videos.zip"; then
            if [[ ! -d "$out/valid" || -z "$(ls -A "$out/valid" 2>/dev/null)" ]]; then
                log "Extracting PT valid videos..."
                unzip -q -o "$out/valid_videos.zip" -d "$out/valid"
            else
                ok "PT valid videos already extracted"
            fi
        else
            warn "valid_videos.zip still downloading — extraction will happen on next run"
        fi
    fi

    ok "Perception Test ready at $out"
}

# =============================================================================
# Dataset: Something-Something-v2  (REGISTRATION REQUIRED)
# =============================================================================
download_ssv2() {
    section "Something-Something-v2 (~19 GB) — REGISTRATION REQUIRED"

    if [[ -z "${SSV2_DOWNLOAD_DIR:-}" ]]; then
        echo ""
        warn "SSv2 requires free registration to download."
        echo ""
        echo "  1. Register at:"
        echo "       https://www.qualcomm.com/developer/software/something-something-data-set"
        echo ""
        echo "  2. Download all 20 zip parts + label JSONs to a local directory"
        echo "       (files named: 20bn-something-something-v2-XX.zip)"
        echo ""
        echo "  3. Re-run with:"
        echo "       SSV2_DOWNLOAD_DIR=/path/to/parts bash get_datasets.sh ssv2"
        echo ""
        return 0
    fi

    local out="$DATA_DIR/ssv2"
    local frames="$out/frames"
    local labels="$out/labels"
    mkdir -p "$out" "$frames" "$labels"

    # Copy label JSONs
    for f in "$SSV2_DOWNLOAD_DIR"/*.json; do
        [[ -f "$f" ]] && cp "$f" "$labels/"
    done

    # Extract frames from zip parts
    log "Extracting SSv2 zip parts (this takes ~30 min)..."
    for part in "$SSV2_DOWNLOAD_DIR"/20bn-something-something-v2-*.zip; do
        [[ -f "$part" ]] || continue
        log "  Extracting $(basename "$part")..."
        unzip -q -o "$part" -d "$out/raw_webm"
    done

    # Convert webm → JPEG frames
    local n_webm
    n_webm=$(ls "$out/raw_webm"/*.webm 2>/dev/null | wc -l || echo 0)
    local n_done
    n_done=$(ls -d "$frames"/*/ 2>/dev/null | wc -l || echo 0)

    if [[ "$n_done" -lt "$n_webm" ]]; then
        log "Converting $n_webm webm files → JPEG frames (~2h)..."
        python3 - "$out/raw_webm" "$frames" <<'PYEOF'
import sys, av, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

raw_dir, frames_dir = Path(sys.argv[1]), Path(sys.argv[2])
webm_files = sorted(raw_dir.glob("*.webm"))

def extract(webm):
    vid_id = webm.stem
    out_dir = frames_dir / vid_id
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0:
        return "skip"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        container = av.open(str(webm))
        for i, frame in enumerate(container.decode(video=0)):
            frame.to_image().save(out_dir / f"{i:05d}.jpg", quality=85)
        return "ok"
    except Exception as e:
        return f"fail: {e}"

ok = skip = fail = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(extract, w): w for w in webm_files}
    for i, fut in enumerate(as_completed(futures), 1):
        r = fut.result()
        if r == "ok":     ok   += 1
        elif r == "skip": skip += 1
        else:             fail += 1
        if i % 1000 == 0 or i == len(webm_files):
            print(f"  {i}/{len(webm_files)}  ok={ok} skip={skip} fail={fail}", flush=True)
PYEOF
    else
        ok "Frames already extracted ($n_done dirs)"
    fi

    ok "SSv2 ready at $out"
}

# =============================================================================
# Dataset: JHMDB  (REGISTRATION REQUIRED)
# =============================================================================
download_jhmdb() {
    section "JHMDB (~13 GB) — REGISTRATION REQUIRED"

    if [[ -z "${JHMDB_DOWNLOAD_URL:-}" ]]; then
        echo ""
        warn "JHMDB requires free registration to download."
        echo ""
        echo "  1. Register at:"
        echo "       http://jhmdb.is.tue.mpg.de/"
        echo ""
        echo "  2. You will receive a download URL by email"
        echo ""
        echo "  3. Re-run with:"
        echo "       JHMDB_DOWNLOAD_URL=<url-from-email> bash get_datasets.sh jhmdb"
        echo ""
        return 0
    fi

    local out="$DATA_DIR/jhmdb"
    local base="${JHMDB_DOWNLOAD_URL%/}"
    mkdir -p "$out"

    download "$base/joint_positions.tar.bz2"  "$out/joint_positions.tar.bz2"  "JHMDB joint annotations"
    download "$base/Rename_Images.tar.gz"      "$out/Rename_Images.tar.gz"     "JHMDB frames"
    download "$base/splits.tar.bz2"            "$out/splits.tar.bz2"           "JHMDB splits" 2>/dev/null || \
    download "$base/test_train_splits.tar.bz2" "$out/splits.tar.bz2"           "JHMDB splits"

    if [[ ! -d "$out/joint_positions" ]]; then
        log "Extracting JHMDB..."
        tar xjf "$out/joint_positions.tar.bz2"  -C "$out"
        tar xzf "$out/Rename_Images.tar.gz"      -C "$out"
        tar xjf "$out/splits.tar.bz2"            -C "$out" 2>/dev/null || true
    fi

    ok "JHMDB ready at $out"
}

# =============================================================================
# Build index files
# =============================================================================
build_indexes() {
    section "Building dataset index files"
    local script
    script="$(cd "$(dirname "$0")" && pwd)/get_datasets.py"
    if [[ -f "$script" ]]; then
        python3 "$script" --build-index --data-dir "$DATA_DIR"
    else
        python3 "$(dirname "$0")/../data/build_index.py" --datasets all 2>/dev/null || \
        warn "Index builder not found — run manually: python get_datasets.py --build-index"
    fi
}

# ─── Summary of what was requested ────────────────────────────────────────────
print_summary() {
    section "Download summary"

    local sizes=(
        "davis:2.3 GB"
        "kinetics700:~450 GB"
        "kinetics400:~140 GB"
        "perception:~100 GB"
        "ssv2:~19 GB"
        "jhmdb:~13 GB"
    )

    for entry in "${sizes[@]}"; do
        local name="${entry%%:*}"
        local size="${entry##*:}"
        local dir="$DATA_DIR/$name"
        [[ "$name" == "perception" ]] && dir="$DATA_DIR/perception_test"

        for req in "${REQUESTED[@]}"; do
            if [[ "$req" == "$name" ]]; then
                if [[ -d "$dir" ]]; then
                    local used
                    used=$(du -sh "$dir" 2>/dev/null | cut -f1)
                    ok "$name ($size) → $dir  [on disk: $used]"
                else
                    warn "$name ($size) → not yet downloaded"
                fi
            fi
        done
    done
}

# =============================================================================
# Main
# =============================================================================
section "RVM-EUPE Dataset Downloader"
echo "  Data dir:  $DATA_DIR"
echo "  Parallel:  $PARALLEL connections"
echo "  Datasets:  ${REQUESTED[*]}"

for ds in "${REQUESTED[@]}"; do
    case "$ds" in
        davis)       download_davis       ;;
        kinetics700) download_kinetics700 ;;
        kinetics400) download_kinetics400 ;;
        perception)  download_perception  ;;
        ssv2)        download_ssv2        ;;
        jhmdb)       download_jhmdb       ;;
        *)           warn "Unknown dataset: $ds — skipping" ;;
    esac
done

if [[ "$SKIP_INDEX" == false && "$DRY_RUN" == false ]]; then
    build_indexes 2>/dev/null || warn "Index build failed — run manually after all downloads complete"
fi

print_summary

section "Done"
echo ""
echo "Next steps:"
echo "  1. Register for SSv2 and JHMDB if needed (see warnings above)"
echo "  2. Once all datasets are present, update the training config:"
echo "     $DATA_DIR → configs/data/mixed.yaml"
echo ""
