#!/usr/bin/env bash
set -euo pipefail

KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
HOST="${SCHOOL_HOST:-103.73.232.201}"
PORT="${SCHOOL_PORT:-62161}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/vipragsent-repro}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

test -f "$KEY" || { echo "Missing SSH private key: $KEY" >&2; exit 2; }
ssh -i "$KEY" -p "$PORT" root@"$HOST" "mkdir -p '$REMOTE_ROOT'"
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude 'outputs/' --exclude 'results/predictions/' \
  -e "ssh -i $KEY -p $PORT" "$LOCAL_ROOT/" root@"$HOST":"$REMOTE_ROOT/"
ssh -i "$KEY" -p "$PORT" root@"$HOST" "chmod +x '$REMOTE_ROOT'/scripts/*.sh && '$REMOTE_ROOT/scripts/bootstrap_h100_server.sh' '$REMOTE_ROOT'"
