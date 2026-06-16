#!/bin/bash
# Run on Pi to verify updated code is installed.
cd "$(dirname "$0")"
echo "=== Code version check ==="
python3 -c "from version import SERVICE_VERSION; print('version:', SERVICE_VERSION)" 2>/dev/null \
  || echo "MISSING version.py — code is OLD"
grep -q "ThingsBoard mode" run_camera.py \
  && echo "run_camera.py: OK (on-demand)" \
  || echo "run_camera.py: OLD — need update"
grep -q "find_terminal_heads" detect_salt.py \
  && echo "detect_salt.py: OK (auto terminal heads)" \
  || echo "detect_salt.py: OLD — need update"
grep -q "SERVICE_VERSION" thingsboard_service.py \
  && echo "thingsboard_service.py: OK" \
  || echo "thingsboard_service.py: OLD — need update"
echo ""
echo "Run GUI on HDMI (no --headless):"
echo "  python3 thingsboard_service.py --token <TOKEN>"
