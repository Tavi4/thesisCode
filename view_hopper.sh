#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XML_PATH="${1:-$SCRIPT_DIR/hopper.xml}"

if [[ ! -f "$XML_PATH" ]]; then
  echo "Eroare: fișierul XML nu există: $XML_PATH" >&2
  echo "Utilizare: ./view_hopper.sh [cale/catre/hopper.xml]" >&2
  exit 1
fi

if ! python -c "import mujoco" >/dev/null 2>&1; then
  echo "Eroare: pachetul Python 'mujoco' nu este instalat în mediul curent." >&2
  echo "Rulează: pip install mujoco" >&2
  exit 1
fi

echo "Deschid modelul: $XML_PATH"
python -m mujoco.viewer --mjcf="$XML_PATH"