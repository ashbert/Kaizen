#!/usr/bin/env bash
#
# Run the py_to_go demo with AgentFS workspace isolation.
#
# Prerequisites:
#   - agentfs CLI installed (curl -fsSL https://agentfs.ai/install | bash)
#   - KAIZEN_MODEL_URL set to an OpenAI-compatible endpoint
#
# Usage:
#   KAIZEN_MODEL_URL="https://..." ./demo/py_to_go/run_with_agentfs.sh
#
set -euo pipefail

if ! command -v agentfs &>/dev/null; then
    echo "Error: agentfs CLI not found."
    echo "Install: curl -fsSL https://agentfs.ai/install | bash"
    exit 1
fi

if [[ -z "${KAIZEN_MODEL_URL:-}" ]]; then
    echo "Error: KAIZEN_MODEL_URL must be set."
    echo "Example: KAIZEN_MODEL_URL=https://your-endpoint.modal.run $0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Initialize an AgentFS filesystem
agentfs init kaizen-demo --force
echo "AgentFS filesystem initialized: kaizen-demo"

export KAIZEN_MODEL_URL
export KAIZEN_MODEL_NAME="${KAIZEN_MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
export KAIZEN_API_KEY="${KAIZEN_API_KEY:-}"

echo "Running demo with:"
echo "  KAIZEN_MODEL_URL=$KAIZEN_MODEL_URL"
echo "  KAIZEN_MODEL_NAME=$KAIZEN_MODEL_NAME"
echo ""

# Use agentfs exec to mount, run the demo with cwd as workspace, then auto-unmount.
# $PWD inside the exec context is the mounted AgentFS directory.
agentfs exec kaizen-demo \
    bash -c 'KAIZEN_WORKSPACE="$PWD" exec python3 "'"$PROJECT_ROOT"'/demo/py_to_go/run_demo.py"'

echo ""
echo "View agent timeline:"
echo "  agentfs timeline kaizen-demo"
echo ""
echo "Re-mount to inspect files:"
echo "  mkdir -p /tmp/kaizen-workspace && agentfs mount kaizen-demo /tmp/kaizen-workspace"
