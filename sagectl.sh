#!/usr/bin/env bash
# SAGE one-command launcher and background service controller.
#
# Quick run:
#   chmod +x sagectl.sh
#   ./sagectl.sh
#
# Common controls:
#   ./sagectl.sh start|stop|restart|status|logs|open

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SAGE_HOST="${SAGE_HOST:-127.0.0.1}"
SAGE_PORT="${SAGE_PORT:-5010}"
SAGE_VENV="${SAGE_VENV:-$ROOT_DIR/.venv}"
SAGE_RUNTIME_DIR="${SAGE_RUNTIME_DIR:-$ROOT_DIR/.sage_runtime}"
SAGE_AUTO_OPEN="${SAGE_AUTO_OPEN:-1}"

LOG_DIR="$SAGE_RUNTIME_DIR/logs"
PID_FILE="$SAGE_RUNTIME_DIR/sage_web.pid"
ENV_FILE="$SAGE_RUNTIME_DIR/sage.env"
WEB_LOG="$LOG_DIR/web_${SAGE_PORT}.log"
OLLAMA_PID_FILE="$SAGE_RUNTIME_DIR/ollama.pid"
OLLAMA_LOG="$LOG_DIR/ollama.log"

mkdir -p "$LOG_DIR"

info() { printf '\033[1;34m[info]\033[0m %s\n' "$*"; }
ok() { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
SAGE control script

Usage:
  ./sagectl.sh                 Install deps, configure, start in background, open browser
  ./sagectl.sh up              Same as default
  ./sagectl.sh setup           Create .venv and install Python dependencies
  ./sagectl.sh configure       Run lightweight SAGE backend auto-configuration
  ./sagectl.sh start|on        Start the web app in background
  ./sagectl.sh stop|off        Stop the background web app
  ./sagectl.sh restart         Restart the web app
  ./sagectl.sh status          Show web/Ollama/runtime status
  ./sagectl.sh logs            Tail web logs
  ./sagectl.sh open            Open http://$SAGE_HOST:$SAGE_PORT
  ./sagectl.sh doctor          Check Python, dependencies, port and backend status
  ./sagectl.sh ollama-start    Start Ollama in background if installed
  ./sagectl.sh ollama-stop     Stop Ollama only if started by this script
  ./sagectl.sh ollama-status   Show Ollama status

Environment overrides:
  SAGE_PORT=5011 ./sagectl.sh start
  SAGE_HOST=0.0.0.0 ./sagectl.sh start
  SAGE_AUTO_OPEN=0 ./sagectl.sh up
  SAGE_PYTHON=/Users/anaconda3/bin/python ./sagectl.sh start
  SAGE_VENV=/path/to/venv ./sagectl.sh setup

Runtime files:
  $SAGE_RUNTIME_DIR
EOF
}

python_candidates() {
  local seen="" py
  for py in "${SAGE_PYTHON:-}" "$(command -v python 2>/dev/null || true)" "$(command -v python3 2>/dev/null || true)" "$SAGE_VENV/bin/python"; do
    [[ -n "$py" && -x "$py" ]] || continue
    case ":$seen:" in
      *":$py:"*) ;;
      *) seen="${seen:+$seen:}$py"; printf '%s\n' "$py" ;;
    esac
  done
}

python_supported() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)
PY
}

python_version_text() {
  local py="$1"
  "$py" - <<'PY' 2>/dev/null || true
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}

deps_ready_for() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import flask
import flask_cors
import numpy
import pandas
import scipy
import requests
import matplotlib
import plotly
import pdfminer
import fitz
import jieba
import openai
import ollama
PY
}

python_cmd() {
  local py first_supported="" first_any=""
  while IFS= read -r py; do
    [[ -n "$first_any" ]] || first_any="$py"
    if python_supported "$py"; then
      [[ -n "$first_supported" ]] || first_supported="$py"
      if deps_ready_for "$py"; then
        printf '%s\n' "$py"
        return
      fi
    fi
  done < <(python_candidates)

  if [[ -n "$first_supported" ]]; then
    printf '%s\n' "$first_supported"
    return
  fi
  if [[ -n "$first_any" ]]; then
    warn "Found Python $(python_version_text "$first_any"), but SAGE dependencies are best supported on Python 3.9-3.12."
    printf '%s\n' "$first_any"
    return
  fi
  fail "Python not found. Please install Python 3.9-3.12 first."
}

ensure_venv() {
  local py
  if [[ -x "$SAGE_VENV/bin/python" ]] && python_supported "$SAGE_VENV/bin/python"; then
    ok "Using virtualenv: $SAGE_VENV"
    return
  fi
  if [[ -x "$SAGE_VENV/bin/python" ]]; then
    warn "Existing virtualenv uses Python $(python_version_text "$SAGE_VENV/bin/python"), which is not supported by several SAGE dependencies. Recreating it."
    rm -rf "$SAGE_VENV"
  fi
  py=""
  while IFS= read -r candidate; do
    if python_supported "$candidate"; then
      py="$candidate"
      break
    fi
  done < <(python_candidates)
  [[ -n "$py" ]] || fail "No supported Python found. Please use Python 3.9-3.12, e.g. conda activate base, then rerun."
  info "Creating virtualenv: $SAGE_VENV"
  "$py" -m venv "$SAGE_VENV"
  ok "Virtualenv created"
}

deps_ready() {
  local py
  py="$(python_cmd)"
  python_supported "$py" && deps_ready_for "$py"
}

write_env_file() {
  cat > "$ENV_FILE" <<EOF
SAGE_HOST=$SAGE_HOST
SAGE_PORT=$SAGE_PORT
SAGE_URL=http://$SAGE_HOST:$SAGE_PORT
SAGE_VENV=$SAGE_VENV
SAGE_PYTHON=$(python_cmd)
SAGE_RUNTIME_DIR=$SAGE_RUNTIME_DIR
WEB_LOG=$WEB_LOG
PID_FILE=$PID_FILE
EOF
}

install_deps() {
  if deps_ready; then
    ok "Python environment already has required web dependencies: $(python_cmd)"
    write_env_file
    return
  fi
  ensure_venv
  local py
  py="$SAGE_VENV/bin/python"
  info "Upgrading pip toolchain"
  "$py" -m pip install --upgrade pip setuptools wheel

  info "Installing root requirements.txt"
  "$py" -m pip install -r "$ROOT_DIR/requirements.txt"

  if [[ -f "$ROOT_DIR/web_app/requirements.txt" ]]; then
    info "Installing web_app/requirements.txt"
    "$py" -m pip install -r "$ROOT_DIR/web_app/requirements.txt"
  fi

  write_env_file
  ok "Dependencies installed"
}

configure_sage() {
  local py
  py="$(python_cmd)"
  write_env_file
  info "Running lightweight backend auto-configuration"
  if [[ -f "$ROOT_DIR/seismic_cli.py" ]]; then
    "$py" "$ROOT_DIR/seismic_cli.py" backend auto || warn "Backend auto-configuration skipped or failed; you can still configure models in the web Config page."
    "$py" "$ROOT_DIR/seismic_cli.py" backend status || true
  else
    warn "seismic_cli.py not found; skipping CLI backend configuration"
  fi
}

is_pid_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

saved_pid() {
  [[ -f "$PID_FILE" ]] && tr -d '[:space:]' < "$PID_FILE" || true
}

port_pid() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$SAGE_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
  else
    true
  fi
}

start_web() {
  local pid existing py url
  pid="$(saved_pid)"
  if is_pid_alive "$pid"; then
    ok "SAGE web already running: PID $pid"
    printf 'URL: http://%s:%s\n' "$SAGE_HOST" "$SAGE_PORT"
    return
  fi
  rm -f "$PID_FILE"

  existing="$(port_pid)"
  if [[ -n "$existing" ]]; then
    fail "Port $SAGE_PORT is already in use by PID $existing. Stop it or use SAGE_PORT=5011 ./sagectl.sh start"
  fi

  if ! deps_ready; then
    warn "Python environment is not ready; running setup first"
    install_deps
  fi
  py="$(python_cmd)"
  url="http://$SAGE_HOST:$SAGE_PORT"
  write_env_file

  info "Starting SAGE web in background"
  info "Log: $WEB_LOG"
  nohup "$py" "$ROOT_DIR/web_app/app.py" --host "$SAGE_HOST" --port "$SAGE_PORT" >> "$WEB_LOG" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" > "$PID_FILE"

  for _ in $(seq 1 30); do
    if is_pid_alive "$pid" && command -v curl >/dev/null 2>&1 && curl -fsS "$url" >/dev/null 2>&1; then
      ok "SAGE web is ready: $url"
      return
    fi
    if ! is_pid_alive "$pid"; then
      rm -f "$PID_FILE"
      tail -n 80 "$WEB_LOG" >&2 || true
      fail "SAGE web exited during startup"
    fi
    sleep 1
  done
  warn "Process started but readiness check did not complete yet: PID $pid"
  printf 'URL: %s\n' "$url"
}

stop_web() {
  local pid
  pid="$(saved_pid)"
  if ! is_pid_alive "$pid"; then
    rm -f "$PID_FILE"
    warn "SAGE web is not running"
    return
  fi
  info "Stopping SAGE web: PID $pid"
  kill "$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    is_pid_alive "$pid" || break
    sleep 0.5
  done
  if is_pid_alive "$pid"; then
    warn "Graceful stop timed out; forcing PID $pid"
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$PID_FILE"
  ok "SAGE web stopped"
}

status_web() {
  local pid ppid
  pid="$(saved_pid)"
  if is_pid_alive "$pid"; then
    ok "SAGE web running: PID $pid"
    printf 'URL: http://%s:%s\n' "$SAGE_HOST" "$SAGE_PORT"
  else
    ppid="$(port_pid)"
    if [[ -n "$ppid" ]]; then
      warn "Port $SAGE_PORT is in use by PID $ppid, but it is not managed by $PID_FILE"
    else
      warn "SAGE web is stopped"
    fi
  fi
  printf 'Runtime: %s\n' "$SAGE_RUNTIME_DIR"
  printf 'Log: %s\n' "$WEB_LOG"
}

open_web() {
  local url="http://$SAGE_HOST:$SAGE_PORT"
  if command -v open >/dev/null 2>&1; then
    open "$url"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
  else
    warn "No browser opener found. Visit $url manually."
  fi
}

tail_web_logs() {
  touch "$WEB_LOG"
  info "Tailing $WEB_LOG"
  tail -n 120 -f "$WEB_LOG"
}

ollama_http_alive() {
  command -v curl >/dev/null 2>&1 && curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1
}

ollama_start() {
  if ! command -v ollama >/dev/null 2>&1; then
    warn "Ollama is not installed. Install it from https://ollama.com, then run: ollama pull qwen3:8b"
    return
  fi
  if ollama_http_alive; then
    ok "Ollama is already running"
    return
  fi
  info "Starting Ollama in background"
  nohup ollama serve >> "$OLLAMA_LOG" 2>&1 &
  printf '%s\n' "$!" > "$OLLAMA_PID_FILE"
  for _ in $(seq 1 20); do
    ollama_http_alive && { ok "Ollama is ready"; return; }
    sleep 1
  done
  warn "Ollama start requested, but readiness check did not complete yet. Log: $OLLAMA_LOG"
}

ollama_stop() {
  local pid
  pid="$([[ -f "$OLLAMA_PID_FILE" ]] && tr -d '[:space:]' < "$OLLAMA_PID_FILE" || true)"
  if is_pid_alive "$pid"; then
    info "Stopping script-managed Ollama: PID $pid"
    kill "$pid" >/dev/null 2>&1 || true
    rm -f "$OLLAMA_PID_FILE"
    ok "Ollama stopped"
  else
    warn "No script-managed Ollama process found. If Ollama was started elsewhere, stop it from that terminal or app."
  fi
}

ollama_status() {
  if ollama_http_alive; then
    ok "Ollama API is reachable at http://127.0.0.1:11434"
    if command -v ollama >/dev/null 2>&1; then
      ollama list || true
    fi
  else
    warn "Ollama API is not reachable"
  fi
}

doctor() {
  local py
  py="$(python_cmd)"
  info "Repository: $ROOT_DIR"
  info "Python: $("$py" --version 2>&1)"
  info "Pip: $("$py" -m pip --version 2>&1)"
  status_web
  ollama_status
  if [[ -f "$ROOT_DIR/seismic_cli.py" ]]; then
    "$py" "$ROOT_DIR/seismic_cli.py" backend status || true
  fi
}

up() {
  if deps_ready; then
    ok "Python environment is ready: $(python_cmd)"
    write_env_file
  else
    install_deps
  fi
  configure_sage
  start_web
  if [[ "$SAGE_AUTO_OPEN" == "1" ]]; then
    open_web
  fi
}

cmd="${1:-up}"
case "$cmd" in
  up) up ;;
  setup) install_deps ;;
  configure|config) configure_sage ;;
  start|on) start_web ;;
  stop|off) stop_web ;;
  restart) stop_web; start_web ;;
  status) status_web ;;
  logs|log) tail_web_logs ;;
  open) open_web ;;
  doctor) doctor ;;
  ollama-start) ollama_start ;;
  ollama-stop) ollama_stop ;;
  ollama-status) ollama_status ;;
  help|-h|--help) usage ;;
  *)
    usage
    fail "Unknown command: $cmd"
    ;;
esac
