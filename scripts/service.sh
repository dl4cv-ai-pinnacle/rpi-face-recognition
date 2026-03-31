#!/usr/bin/env bash
# Manage the face recognition pipeline as a systemd service on Raspberry Pi.
#
# First run handles full setup: system deps, uv, Python venv, models, service unit.
# Subsequent runs just manage the service.
#
# Usage:
#   bash scripts/service.sh setup    # Full setup from zero + start
#   bash scripts/service.sh up       # Start (creates unit if needed)
#   bash scripts/service.sh down     # Stop
#   bash scripts/service.sh restart  # Restart
#   bash scripts/service.sh status   # Show status
#   bash scripts/service.sh logs     # Follow logs
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVICE_NAME="face-recognition.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
SERVICE_GROUP="$(id -gn "${SERVICE_USER}")"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
fail()  { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*" >&2; exit 1; }

need_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------

install_system_deps() {
  info "Installing system packages..."
  need_sudo apt update -qq
  need_sudo apt install -y -qq \
    python3-picamera2 \
    python3-opencv \
    python3-libcamera \
    libcap-dev \
    curl \
    unzip \
    git
  ok "System packages installed"
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    ok "uv already installed ($(uv --version))"
    return
  fi
  info "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Source the env so uv is available in this session.
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    ok "uv installed ($(uv --version))"
  else
    fail "uv installation failed"
  fi
}

setup_venv() {
  cd "${REPO_ROOT}"
  if [[ -d .venv ]] && .venv/bin/python3 -c "import picamera2" 2>/dev/null; then
    ok "venv already exists with system-site-packages"
    return
  fi
  info "Creating venv with system-site-packages..."
  # Remove existing venv if it can't see picamera2.
  rm -rf .venv
  uv venv --python 3.13 --system-site-packages
  ok "venv created"

  info "Installing Python dependencies..."
  uv sync --python 3.13
  ok "Dependencies installed"

  info "Installing insightface..."
  uv pip install insightface
  ok "insightface installed"
}

download_models() {
  cd "${REPO_ROOT}"
  if [[ -f models/buffalo_sc/det_500m.onnx && -f models/version-slim-320.onnx ]]; then
    ok "Models already downloaded"
    return
  fi
  info "Downloading models..."
  bash scripts/download_models.sh
  ok "Models ready"
}

full_setup() {
  info "=== Full setup from zero ==="
  install_system_deps
  install_uv
  setup_venv
  download_models
  ok "=== Setup complete ==="
  echo ""
  cmd_up
}

# ---------------------------------------------------------------------------
# Systemd service management
# ---------------------------------------------------------------------------

render_service_unit() {
  local uv_bin
  uv_bin="$(command -v uv)"
  cat <<UNIT
[Unit]
Description=Face Recognition Pipeline — Live Camera Server
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${REPO_ROOT}
Environment=PYTHONUNBUFFERED=1
Environment=PATH=${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=${uv_bin} run --python 3.13 python -m server.app
Restart=never
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
}

ensure_service_file() {
  local tmp_file action
  tmp_file="$(mktemp)"
  render_service_unit > "${tmp_file}"
  action="unchanged"

  if [[ ! -f "${SERVICE_PATH}" ]]; then
    info "Creating ${SERVICE_PATH}"
    action="created"
  elif ! cmp -s "${tmp_file}" "${SERVICE_PATH}"; then
    info "Updating ${SERVICE_PATH}"
    action="updated"
  fi

  if [[ "${action}" != "unchanged" ]]; then
    need_sudo install -m 0644 "${tmp_file}" "${SERVICE_PATH}"
    need_sudo systemctl daemon-reload
  fi
  rm -f "${tmp_file}"
  printf '%s\n' "${action}"
}

ensure_enabled() {
  if ! need_sudo systemctl is-enabled --quiet "${SERVICE_NAME}" 2>/dev/null; then
    need_sudo systemctl enable "${SERVICE_NAME}"
  fi
}

cmd_up() {
  local service_action
  service_action="$(ensure_service_file)"
  ensure_enabled
  if need_sudo systemctl is-active --quiet "${SERVICE_NAME}"; then
    if [[ "${service_action}" == "updated" ]]; then
      need_sudo systemctl restart "${SERVICE_NAME}"
      ok "Restarted ${SERVICE_NAME} (config updated)"
    else
      ok "${SERVICE_NAME} is already running"
    fi
  else
    need_sudo systemctl start "${SERVICE_NAME}"
    ok "Started ${SERVICE_NAME}"
  fi
  echo ""
  echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8080/"
  echo "  Logs:      bash scripts/service.sh logs"
  echo ""
}

cmd_down() {
  if need_sudo systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
    need_sudo systemctl stop "${SERVICE_NAME}"
    ok "Stopped ${SERVICE_NAME}"
  else
    warn "${SERVICE_NAME} is not running"
  fi
}

cmd_restart() {
  ensure_service_file >/dev/null
  ensure_enabled
  need_sudo systemctl restart "${SERVICE_NAME}"
  ok "Restarted ${SERVICE_NAME}"
}

cmd_status() {
  need_sudo systemctl status --no-pager "${SERVICE_NAME}" 2>/dev/null || warn "Service not found"
}

cmd_logs() {
  need_sudo journalctl -u "${SERVICE_NAME}" -f --no-hostname -o short-iso
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command>

Commands:
  setup    Full setup from zero: install deps, create venv, download models, start service
  up       Start the service (creates systemd unit if needed)
  down     Stop the service
  restart  Restart the service
  status   Show service status
  logs     Follow service logs (Ctrl+C to stop)
USAGE
}

main() {
  if [[ $# -ne 1 ]]; then
    usage >&2
    exit 1
  fi

  case "$1" in
    setup)   full_setup ;;
    up)      cmd_up ;;
    down)    cmd_down ;;
    restart) cmd_restart ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    -h|--help|help) usage ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
