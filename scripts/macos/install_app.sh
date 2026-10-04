#!/bin/bash
# Write ~/Applications/compute.app (or --dest) that launches this checkout's venv.
set -euo pipefail

REPO=""
DEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    *) echo "usage: $0 [--repo PATH] [--dest PATH]" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "$REPO" ]]; then
  REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
REPO="$(cd "$REPO" && pwd)"

if [[ -z "$DEST" ]]; then
  mkdir -p "$HOME/Applications"
  DEST="$HOME/Applications/compute.app"
fi
# DEST gets rm -rf'd below, so only ever accept an app bundle path.
DEST="${DEST%/}"
if [[ "$DEST" != *.app || "$DEST" == "${HOME%/}" ]]; then
  echo "Refusing --dest $DEST: must be a path ending in .app" >&2
  exit 2
fi

PY="$REPO/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No venv at $PY — run: cd \"$REPO\" && uv sync --group dev" >&2
  exit 1
fi

CONTENTS="$DEST/Contents"
MACOS="$CONTENTS/MacOS"
RES="$CONTENTS/Resources"
rm -rf "$DEST"
mkdir -p "$MACOS" "$RES"

cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleDisplayName</key><string>/compute</string>
  <key>CFBundleExecutable</key><string>compute</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundleIdentifier</key><string>com.slashcompute.app</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>compute</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSUIElement</key><false/>
</dict>
</plist>
EOF

# %q-quote the paths so quotes, $ or backticks in them stay literal.
{
  echo '#!/bin/bash'
  echo 'export PATH="/usr/bin:/bin:/usr/sbin:/sbin"'
  printf 'cd %q\n' "$REPO"
  printf 'exec %q -m slashcompute.launcher.main\n' "$PY"
} > "$MACOS/compute"
chmod +x "$MACOS/compute"

ICON_SRC="$SCRIPT_DIR/icon.png"
if [[ ! -f "$ICON_SRC" ]]; then
  echo "Missing app icon at $ICON_SRC" >&2
  exit 1
fi
cp "$ICON_SRC" "$RES/icon.png"

if command -v sips >/dev/null && command -v iconutil >/dev/null; then
  ICONSET="$RES/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for pair in 16:16 32:16 32:32 64:32 128:128 256:128 256:256 512:256 512:512 1024:512; do
    px="${pair%%:*}"
    base="${pair##*:}"
    if [[ "$px" == "$base" ]]; then
      name="icon_${base}x${base}.png"
    else
      name="icon_${base}x${base}@2x.png"
    fi
    sips -z "$px" "$px" "$RES/icon.png" --out "$ICONSET/$name" >/dev/null
  done
  iconutil -c icns -o "$RES/AppIcon.icns" "$ICONSET"
  rm -rf "$ICONSET"
fi

echo "Installed $DEST"
echo "Double-click /compute, or open -a $DEST"
