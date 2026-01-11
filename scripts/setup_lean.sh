#!/bin/bash
# scripts/setup_lean.sh

set -e

LEAN_VERSION="v2.5.15.1" # Example pinned version
TARGET_DIR="external/lean"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Cloning QuantConnect LEAN engine..."
    git clone https://github.com/QuantConnect/Lean.git "$TARGET_DIR"
fi

cd "$TARGET_DIR"
echo "Checking out version $LEAN_VERSION..."
git checkout "$LEAN_VERSION"

echo "LEAN setup complete in $TARGET_DIR."
echo "To build LEAN, ensure you have Docker or the .NET SDK installed."
echo "Refer to integrations/lean_runner/README.md for custom strategy integration."



