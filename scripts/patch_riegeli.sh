#!/bin/bash
# Patch Riegeli background_cleaning.cc for newer Abseil API compatibility
# Fixes:
#   1. absl::MutexLock expects Mutex* not Mutex&
#   2. mutex_.lock()/unlock() should be Lock()/Unlock()

RIEGELI_DIR="$1"
TARGET_FILE="${RIEGELI_DIR}/riegeli/base/background_cleaning.cc"

if [ ! -f "$TARGET_FILE" ]; then
    echo "Riegeli background_cleaning.cc not found at: $TARGET_FILE"
    exit 0
fi

# Check if already patched
if grep -q "MutexLock lock(&mutex_)" "$TARGET_FILE"; then
    echo "Riegeli already patched"
    exit 0
fi

echo "Patching Riegeli background_cleaning.cc for Abseil compatibility..."

# Fix MutexLock: mutex_ -> &mutex_
sed -i.bak 's/MutexLock lock(mutex_)/MutexLock lock(\&mutex_)/g' "$TARGET_FILE"

# Fix lowercase lock/unlock -> Lock/Unlock
sed -i.bak 's/mutex_\.lock()/mutex_.Lock()/g' "$TARGET_FILE"
sed -i.bak 's/mutex_\.unlock()/mutex_.Unlock()/g' "$TARGET_FILE"

# Clean up backup files
rm -f "${TARGET_FILE}.bak"

echo "Riegeli patched successfully"
