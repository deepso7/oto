#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Rebuilds packaging/macos/AppIcon.icns from the artwork in packaging/brand: oto-icon.svg for
# 64 px and up, oto-icon-small.svg (fewer, heavier strokes) for 16 and 32 px. AppKit renders
# the SVGs, so this needs only the Xcode command line tools. The .icns is committed; run this
# after changing the artwork.
#
#   scripts/macos/make-icon.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BRAND="$ROOT/packaging/brand"
OUT="$ROOT/packaging/macos/AppIcon.icns"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/render.swift" <<'EOF'
import AppKit
let a = CommandLine.arguments
guard let img = NSImage(contentsOf: URL(fileURLWithPath: a[1])), let px = Int(a[2]) else {
    FileHandle.standardError.write("can't read \(a[1])\n".data(using: .utf8)!); exit(1)
}
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px,
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
NSGraphicsContext.current!.imageInterpolation = .high
img.draw(in: NSRect(x: 0, y: 0, width: px, height: px))
NSGraphicsContext.restoreGraphicsState()
try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: a[3]))
EOF
swiftc -O "$TMP/render.swift" -o "$TMP/render"

SET="$TMP/AppIcon.iconset"
mkdir "$SET"
for pt in 16 32 128 256 512; do
    for scale in 1 2; do
        px=$((pt * scale))
        svg="$BRAND/oto-icon.svg"
        [ "$px" -le 32 ] && svg="$BRAND/oto-icon-small.svg"
        name="icon_${pt}x${pt}"
        [ "$scale" = 2 ] && name="${name}@2x"
        "$TMP/render" "$svg" "$px" "$SET/$name.png"
    done
done
iconutil -c icns "$SET" -o "$OUT"
echo "==> $OUT"
