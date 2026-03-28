#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SETUP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SETUP_ROOT}/.." && pwd)"
SERVICE_NAME="valenia-live-camera.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
SERVICE_USER="${SUDO_USER:-$(id -un)}"
SERVICE_GROUP="$(id -gn "${SERVICE_USER}")"
PYTHON_BIN="$(command -v python3)"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python3 not found in PATH" >&2
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO_CMD=()
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to manage ${SERVICE_NAME}" >&2
    exit 1
  fi
  SUDO_CMD=(sudo)
fi

SERVICE_ARGS=(
  --fps 30
  --det-every 2
  --track-max-missed 3
  --track-iou-thresh 0.3
  --track-smoothing 0.65
  --gallery-dir data/gallery
  --match-threshold 0.228
  --embed-refresh-frames 5
  --embed-refresh-iou 0.85
  --metrics-json data/metrics/live_camera_metrics.json
  --ram-cap-mb 4096
)

usage() {
  cat <<USAGE
Usage: $(basename "$0") <up|down|restart|status|logs>

Commands:
  up       Create the systemd unit if missing, enable it, and start it if not running
  down     Stop the service if running
  restart  Restart the service (creating the unit first if needed)
  status   Show systemd status for the service
  logs     Follow service logs with journalctl
USAGE
}

render_service_unit() {
  cat <<UNIT
[Unit]
Description=Valenia Live Camera MJPEG Server
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${REPO_ROOT}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON_BIN} ${SETUP_ROOT}/scripts/live_camera_server.py ${SERVICE_ARGS[*]}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT
}

ensure_service_file() {
  tmp_file="$(mktemp)"
  render_service_unit > "${tmp_file}"
  local action="unchanged"
  if [[ ! -f "${SERVICE_PATH}" ]]; then
    echo "Creating ${SERVICE_PATH}" >&2
    action="created"
  elif ! cmp -s "${tmp_file}" "${SERVICE_PATH}"; then
    echo "Updating ${SERVICE_PATH}" >&2
    action="updated"
  fi

  if [[ "${action}" != "unchanged" ]]; then
  "${SUDO_CMD[@]}" install -m 0644 "${tmp_file}" "${SERVICE_PATH}"
    "${SUDO_CMD[@]}" systemctl daemon-reload
  fi
  rm -f "${tmp_file}"
  printf '%s\n' "${action}"
}

ensure_enabled() {
  if ! "${SUDO_CMD[@]}" systemctl is-enabled --quiet "${SERVICE_NAME}"; then
    "${SUDO_CMD[@]}" systemctl enable "${SERVICE_NAME}"
  fi
}

require_existing_service() {
  if [[ -f "${SERVICE_PATH}" ]]; then
    return 0
  fi
  echo "${SERVICE_PATH} does not exist yet. Run: $(basename "$0") up" >&2
  exit 1
}

cmd_up() {
  local service_action
  service_action="$(ensure_service_file)"
  ensure_enabled
  if "${SUDO_CMD[@]}" systemctl is-active --quiet "${SERVICE_NAME}"; then
    if [[ "${service_action}" == "updated" ]]; then
      "${SUDO_CMD[@]}" systemctl restart "${SERVICE_NAME}"
      echo "Restarted ${SERVICE_NAME} to apply updated configuration"
    else
      echo "${SERVICE_NAME} is already running"
    fi
  else
    "${SUDO_CMD[@]}" systemctl start "${SERVICE_NAME}"
    echo "Started ${SERVICE_NAME}"
  fi
}

cmd_down() {
  require_existing_service
  if "${SUDO_CMD[@]}" systemctl is-active --quiet "${SERVICE_NAME}"; then
    "${SUDO_CMD[@]}" systemctl stop "${SERVICE_NAME}"
    echo "Stopped ${SERVICE_NAME}"
  else
    echo "${SERVICE_NAME} is not running"
  fi
}

cmd_restart() {
  ensure_service_file >/dev/null
  ensure_enabled
  "${SUDO_CMD[@]}" systemctl restart "${SERVICE_NAME}"
  echo "Restarted ${SERVICE_NAME}"
}

cmd_status() {
  require_existing_service
  "${SUDO_CMD[@]}" systemctl status --no-pager "${SERVICE_NAME}"
}

cmd_logs() {
  require_existing_service
  "${SUDO_CMD[@]}" journalctl -u "${SERVICE_NAME}" -f
}

main() {
  if [[ $# -ne 1 ]]; then
    usage >&2
    exit 1
  fi

  case "$1" in
    up) cmd_up ;;
    down) cmd_down ;;
    restart) cmd_restart ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    -h|--help|help) usage ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
