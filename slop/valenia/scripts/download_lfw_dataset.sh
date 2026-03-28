#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/lfw"
ARCHIVE_PATH="${DATA_DIR}/lfw-funneled.tgz"
# Primary URLs follow scikit-learn's maintained LFW mirrors (Figshare IDs).
LFW_URL="https://ndownloader.figshare.com/files/5976015"
LFW_URL_FALLBACK="https://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
LFW_SHA256="b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"

PAIRS_TRAIN_URL="https://ndownloader.figshare.com/files/5976012"
PAIRS_TEST_URL="https://ndownloader.figshare.com/files/5976009"
PAIRS_VIEW2_URL="https://ndownloader.figshare.com/files/5976006"
PAIRS_TRAIN_URL_FALLBACK="https://vis-www.cs.umass.edu/lfw/pairsDevTrain.txt"
PAIRS_TEST_URL_FALLBACK="https://vis-www.cs.umass.edu/lfw/pairsDevTest.txt"
PAIRS_VIEW2_URL_FALLBACK="https://vis-www.cs.umass.edu/lfw/pairs.txt"
PAIRS_TRAIN_SHA256="1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa"
PAIRS_TEST_SHA256="7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c"
PAIRS_VIEW2_SHA256="ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"

mkdir -p "${DATA_DIR}"

download_with_fallback() {
  local primary_url="$1"
  local fallback_url="$2"
  local output_path="$3"
  if ! curl -fL "${primary_url}" -o "${output_path}"; then
    echo "Primary download failed, trying fallback: ${fallback_url}"
    curl -fL "${fallback_url}" -o "${output_path}"
  fi
}

verify_sha256() {
  local expected="$1"
  local file_path="$2"
  local actual
  actual="$(sha256sum "${file_path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Checksum mismatch for ${file_path}"
    echo "Expected: ${expected}"
    echo "Actual:   ${actual}"
    exit 1
  fi
}

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Downloading LFW funneled archive..."
  download_with_fallback "${LFW_URL}" "${LFW_URL_FALLBACK}" "${ARCHIVE_PATH}"
  verify_sha256 "${LFW_SHA256}" "${ARCHIVE_PATH}"
else
  echo "LFW archive already present: ${ARCHIVE_PATH}"
fi

if [[ ! -d "${DATA_DIR}/lfw_funneled" ]]; then
  echo "Extracting LFW images..."
  tar -xzf "${ARCHIVE_PATH}" -C "${DATA_DIR}"
else
  echo "LFW images already extracted: ${DATA_DIR}/lfw_funneled"
fi

for entry in \
  "${PAIRS_TRAIN_URL}|${PAIRS_TRAIN_URL_FALLBACK}|${DATA_DIR}/pairsDevTrain.txt|${PAIRS_TRAIN_SHA256}" \
  "${PAIRS_TEST_URL}|${PAIRS_TEST_URL_FALLBACK}|${DATA_DIR}/pairsDevTest.txt|${PAIRS_TEST_SHA256}" \
  "${PAIRS_VIEW2_URL}|${PAIRS_VIEW2_URL_FALLBACK}|${DATA_DIR}/pairs.txt|${PAIRS_VIEW2_SHA256}"
do
  url="${entry%%|*}"
  rest="${entry#*|}"
  fallback_url="${rest%%|*}"
  rest="${rest#*|}"
  out="${rest%%|*}"
  checksum="${rest#*|}"
  if [[ ! -f "${out}" ]]; then
    echo "Downloading $(basename "${out}")..."
    download_with_fallback "${url}" "${fallback_url}" "${out}"
  else
    echo "Already present: ${out}"
  fi
  verify_sha256 "${checksum}" "${out}"
done

echo "LFW dataset is ready under ${DATA_DIR}"
