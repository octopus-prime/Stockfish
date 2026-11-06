/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "memory.h"

#include <cstdlib>
#include <iostream>  // std::cerr

#include <errno.h>
#include <sys/mman.h>
// IWYU pragma: no_include <bits/mman-map-flags-generic.h>
#include <cstring>
#include <mutex>
#include <map>


namespace Stockfish {

// Memory allocated with std_aligned_alloc() must be freed with std_aligned_free().

void* std_aligned_alloc(usize alignment, usize size) { return aligned_alloc(alignment, size); }

void std_aligned_free(void* ptr) { free(ptr); }

// aligned_large_pages_alloc() will return suitably aligned memory,
// if possible using large pages.

static std::map<void*, usize> large_page_sizes;
static std::mutex             large_page_sizes_mtx;

static void* try_huge_pages_alloc(usize allocSize) {
    usize size = ((allocSize + HugePageSize - 1) / HugePageSize) * HugePageSize;
    void* mem  = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT), -1, 0);

    if (mem == MAP_FAILED)
        return nullptr;

    std::lock_guard lg(large_page_sizes_mtx);
    large_page_sizes[mem] = size;
    return mem;
}

void* aligned_large_pages_alloc_with_hint(usize allocSize, [[maybe_unused]] bool hugePageHint) {
    if (hugePageHint && allocSize >= HugePageSize)
    {
        void* mem = try_huge_pages_alloc(allocSize);
        if (mem)
            return mem;
    }

    constexpr usize alignment = 2 * 1024 * 1024;  // 2MB page size assumed

    // Round up to multiples of alignment
    usize size = ((allocSize + alignment - 1) / alignment) * alignment;
    void* mem  = mmap_huge_aligned(size, MAP_PRIVATE | MAP_ANONYMOUS);
    if (mem != MAP_FAILED)
    {
        madvise(mem, size, MADV_HUGEPAGE);
        std::lock_guard lg(large_page_sizes_mtx);
        large_page_sizes[mem] = size;
    }
    else
        mem = nullptr;
    return mem;
}

void* aligned_large_pages_alloc(usize size) {
    return aligned_large_pages_alloc_with_hint(size, false);
}

bool has_large_pages() { return true; }


// aligned_large_pages_free() will free the previously memory allocated
// by aligned_large_pages_alloc(). The effect is a nop if mem == nullptr.

void aligned_large_pages_free(void* mem) {
    if (!mem)
        return;

    {
        std::lock_guard lg(large_page_sizes_mtx);
        if (auto it = large_page_sizes.find(mem); it != large_page_sizes.end())
        {
            if (munmap(mem, it->second) != 0)
            {
                std::cerr << "munmap failed: " << strerror(errno) << std::endl;
                exit(EXIT_FAILURE);
            }
            large_page_sizes.erase(it);
            return;
        }
    }

    std_aligned_free(mem);
}

}  // namespace Stockfish
