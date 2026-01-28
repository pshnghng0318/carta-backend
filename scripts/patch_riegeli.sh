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
    
    # Check if already patched (check for either pattern)
    if grep -q "MutexLock lock(&mutex_)" "$file" || grep -q "MutexLock lock(&shared_->mutex)" "$file"; then
        return
    fi
    
    echo "Patching $(basename "$file")..."
    
    # Fix MutexLock: mutex_ -> &mutex_
    sed -i.bak 's/MutexLock lock(mutex_)/MutexLock lock(\&mutex_)/g' "$file"
    
    # Fix MutexLock: shared_->mutex -> &shared_->mutex
    sed -i.bak 's/MutexLock lock(shared_->mutex)/MutexLock lock(\&shared_->mutex)/g' "$file"
    
    # Fix ReleasableMutexLock: mutex_ -> &mutex_
    sed -i.bak 's/ReleasableMutexLock lock(mutex_)/ReleasableMutexLock lock(\&mutex_)/g' "$file"
    
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

echo "Riegeli patched successfully"
