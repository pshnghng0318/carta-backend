#!/bin/bash
# Patch Riegeli for newer Abseil API compatibility
# Fixes:
#   1. absl::MutexLock expects Mutex* not Mutex&
#   2. absl::ReleasableMutexLock expects Mutex* not Mutex&
#   3. mutex_.lock()/unlock() should be Lock()/Unlock()

RIEGELI_DIR="$1"

patch_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return
    fi
    
    # Check if file contains any MutexLock patterns that need patching
    if ! grep -qE "MutexLock lock\([^&]" "$file" && ! grep -qE "ReleasableMutexLock lock\([^&]" "$file" && ! grep -q "mutex_\.lock()" "$file"; then
        return
    fi
    
    echo "Patching $(basename "$file")..."
    
    # Fix MutexLock: any_mutex_var -> &any_mutex_var (generic pattern)
    # Matches: MutexLock lock(foo) -> MutexLock lock(&foo) where foo doesn't start with &
    sed -i.bak 's/MutexLock lock(\([^&)][^)]*\))/MutexLock lock(\&\1)/g' "$file"
    
    # Fix ReleasableMutexLock: same pattern
    sed -i.bak 's/ReleasableMutexLock lock(\([^&)][^)]*\))/ReleasableMutexLock lock(\&\1)/g' "$file"
    
    # Fix lowercase lock/unlock -> Lock/Unlock
    sed -i.bak 's/mutex_\.lock()/mutex_.Lock()/g' "$file"
    sed -i.bak 's/mutex_\.unlock()/mutex_.Unlock()/g' "$file"
    
    # Clean up backup files
    rm -f "${file}.bak"
}

if [ ! -d "${RIEGELI_DIR}/riegeli" ]; then
    echo "Riegeli directory not found at: $RIEGELI_DIR"
    exit 0
fi

echo "Patching Riegeli for Abseil compatibility..."

# Patch all files that use MutexLock with mutex_
patch_file "${RIEGELI_DIR}/riegeli/base/background_cleaning.cc"
patch_file "${RIEGELI_DIR}/riegeli/base/parallelism.cc"
patch_file "${RIEGELI_DIR}/riegeli/base/recycling_pool.h"
patch_file "${RIEGELI_DIR}/riegeli/records/record_writer.cc"
patch_file "${RIEGELI_DIR}/riegeli/bytes/reader_factory.cc"
patch_file "${RIEGELI_DIR}/riegeli/zstd/zstd_dictionary.cc"
patch_file "${RIEGELI_DIR}/riegeli/csv/csv_record.cc"
patch_file "${RIEGELI_DIR}/riegeli/csv/csv_record.h"

echo "Riegeli patched successfully"
