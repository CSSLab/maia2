#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Download monthly standard-rated Lichess PGN archives.

Usage:
  fetch_data.sh [OUTPUT_DIR] [START_MONTH] [END_MONTH]

Arguments:
  OUTPUT_DIR   Download directory (default: ./data)
  START_MONTH  First month in YYYY-MM form (default: 2018-05)
  END_MONTH    Last month in YYYY-MM form (default: 2023-11)

The default date range matches the released Maia-2 training configuration.
Existing files are resumed with wget --continue.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

output_dir="${1:-./data}"
start_month="${2:-2018-05}"
end_month="${3:-2023-11}"

validate_month() {
    local value="$1"
    if [[ ! "$value" =~ ^[0-9]{4}-(0[1-9]|1[0-2])$ ]]; then
        echo "Invalid month '$value'; expected YYYY-MM." >&2
        exit 2
    fi
}

month_key() {
    local value="$1"
    local year="${value%-*}"
    local month="${value#*-}"
    echo $((10#$year * 12 + 10#$month - 1))
}

validate_month "$start_month"
validate_month "$end_month"

start_key="$(month_key "$start_month")"
end_key="$(month_key "$end_month")"
if ((start_key > end_key)); then
    echo "START_MONTH must not be later than END_MONTH." >&2
    exit 2
fi

if ! command -v wget >/dev/null 2>&1; then
    echo "wget is required to download the Lichess archives." >&2
    exit 127
fi

mkdir -p "$output_dir"

base_url="https://database.lichess.org/standard"
for ((key = start_key; key <= end_key; key++)); do
    year=$((key / 12))
    month=$((key % 12 + 1))
    printf -v year_month '%04d-%02d' "$year" "$month"

    # The original Maia-2 pipeline intentionally excludes this month.
    if [[ "$year_month" == "2019-12" ]]; then
        echo "Skipping 2019-12 to match the released Maia-2 training run."
        continue
    fi

    filename="lichess_db_standard_rated_${year_month}.pgn.zst"
    url="${base_url}/${filename}"
    echo "Downloading ${url}"
    wget --continue --directory-prefix="$output_dir" "$url"
done
