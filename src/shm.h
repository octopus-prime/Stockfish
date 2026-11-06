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

#ifndef SHM_H_INCLUDED
#define SHM_H_INCLUDED

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "shm_unix.h"

#include "types.h"

#include "memory.h"

#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


#include <limits.h>
#include <unistd.h>


namespace Stockfish {

// argv[0] CANNOT be used because we need to identify the executable.
// argv[0] contains the command used to invoke it, which does not involve the full path.
// Just using a path is not fully resilient either, as the executable could
// have changed if it wasn't locked by the OS. Ideally we would hash the executable
// but it's not really that important at this point.
// If the path is longer than 4095 bytes the hash will be computed from an unspecified
// amount of bytes of the path; in particular it can a hash of an empty string.

inline std::string getExecutablePathHash() {
    char  executable_path[4096] = {0};
    usize path_length           = 0;

    ssize_t len = readlink("/proc/self/exe", executable_path, sizeof(executable_path) - 1);
    if (len >= 0)
    {
        executable_path[len] = '\0';
        path_length          = len;
    }

    // In case of any error the path will be empty.
    return std::string(executable_path, path_length);
}

enum class SystemWideSharedConstantAllocationStatus {
    NoAllocation,
    LocalMemory,
    SharedMemory
};

template<typename T>
class SharedMemoryBackend {
   public:
    SharedMemoryBackend() = default;

    SharedMemoryBackend(const std::string& shm_name, const T& value) :
        shm1(shm::create_shared<T>(shm_name, value)) {}

    void* get() const {
        const T* ptr = &shm1->get();
        return reinterpret_cast<void*>(const_cast<T*>(ptr));
    }

    bool is_valid() const { return shm1 && shm1->is_mapped() && shm1->is_serving(); }

    SystemWideSharedConstantAllocationStatus get_status() const {
        return is_valid() ? SystemWideSharedConstantAllocationStatus::SharedMemory
                          : SystemWideSharedConstantAllocationStatus::NoAllocation;
    }

    std::optional<std::string> get_error_message() const {
        if (!shm1)
            return "Shared memory not initialized";

        if (!shm1->is_mapped())
            return "Shared memory is not mapped";

        if (!shm1->is_serving())
            return "Shared memory is not serving to other processes";

        return std::nullopt;
    }

   private:
    std::optional<shm::SharedMemory<T>> shm1;
};

template<typename T>
struct SharedMemoryBackendFallback {
    SharedMemoryBackendFallback() = default;

    SharedMemoryBackendFallback(const std::string&, const T& value) :
        fallback_object(make_unique_large_page<T>(value)) {}

    void* get() const { return fallback_object.get(); }

    SharedMemoryBackendFallback(const SharedMemoryBackendFallback&)            = delete;
    SharedMemoryBackendFallback& operator=(const SharedMemoryBackendFallback&) = delete;

    SharedMemoryBackendFallback(SharedMemoryBackendFallback&& other) noexcept :
        fallback_object(std::move(other.fallback_object)) {}

    SharedMemoryBackendFallback& operator=(SharedMemoryBackendFallback&& other) noexcept {
        fallback_object = std::move(other.fallback_object);
        return *this;
    }

    SystemWideSharedConstantAllocationStatus get_status() const {
        return fallback_object == nullptr ? SystemWideSharedConstantAllocationStatus::NoAllocation
                                          : SystemWideSharedConstantAllocationStatus::LocalMemory;
    }

    std::optional<std::string> get_error_message() const {
        if (fallback_object == nullptr)
            return "Not initialized";

        return "Shared memory not supported by the OS. Local allocation fallback.";
    }

   private:
    LargePagePtr<T> fallback_object;
};

// Platform-independent wrapper
template<typename T>
struct SystemWideSharedConstant {
   private:
    static std::string createHashString(const std::string& input) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%016" PRIx64, hash_string(input));
        return buf;
    }

   public:
    // We can't run the destructor because it may be in a completely different process.
    // The object stored must also be obviously in-line but we can't check for that, other than some basic checks that cover most cases.
    static_assert(std::is_trivially_destructible_v<T>);
    static_assert(std::is_trivially_move_constructible_v<T>);
    static_assert(std::is_trivially_copy_constructible_v<T>);

    SystemWideSharedConstant() = default;


    // Content is addressed by its hash. An additional discriminator can be added to account for differences
    // that are not present in the content, for example NUMA node allocation.
    SystemWideSharedConstant(const T& value, usize discriminator = 0) {
        usize content_hash    = std::hash<T>{}(value);
        usize executable_hash = hash_string(getExecutablePathHash());

        char buf[1024];
        std::snprintf(buf, sizeof(buf), "Local\\sf_%zu$%zu$%zu", content_hash, executable_hash,
                      discriminator);
        std::string shm_name = buf;

        // POSIX shared memory names must start with a slash
        shm_name = "/sf_" + createHashString(shm_name);

        // hash name and make sure it is not longer than SF_MAX_SEM_NAME_LEN
        if (shm_name.size() > SF_MAX_SEM_NAME_LEN)
        {
            shm_name = shm_name.substr(0, SF_MAX_SEM_NAME_LEN - 1);
        }

        SharedMemoryBackend<T> shm_backend(shm_name, value);

        if (shm_backend.is_valid())
        {
            backend = std::move(shm_backend);
        }
        else
        {
            backend = SharedMemoryBackendFallback<T>(shm_name, value);
        }
    }

    SystemWideSharedConstant(const SystemWideSharedConstant&)            = delete;
    SystemWideSharedConstant& operator=(const SystemWideSharedConstant&) = delete;

    SystemWideSharedConstant(SystemWideSharedConstant&& other) noexcept :
        backend(std::move(other.backend)) {}

    SystemWideSharedConstant& operator=(SystemWideSharedConstant&& other) noexcept {
        backend = std::move(other.backend);
        return *this;
    }

    const T& operator*() const { return *std::launder(reinterpret_cast<const T*>(get_ptr())); }

    bool operator==(std::nullptr_t) const noexcept { return get_ptr() == nullptr; }

    bool operator!=(std::nullptr_t) const noexcept { return get_ptr() != nullptr; }

    SystemWideSharedConstantAllocationStatus get_status() const {
        return std::visit(
          [](const auto& end) -> SystemWideSharedConstantAllocationStatus {
              if constexpr (std::is_same_v<std::decay_t<decltype(end)>, std::monostate>)
              {
                  return SystemWideSharedConstantAllocationStatus::NoAllocation;
              }
              else
              {
                  return end.get_status();
              }
          },
          backend);
    }

    std::optional<std::string> get_error_message() const {
        return std::visit(
          [](const auto& end) -> std::optional<std::string> {
              if constexpr (std::is_same_v<std::decay_t<decltype(end)>, std::monostate>)
              {
                  return std::nullopt;
              }
              else
              {
                  return end.get_error_message();
              }
          },
          backend);
    }

   private:
    auto get_ptr() const {
        return std::visit(
          [](const auto& end) -> void* {
              if constexpr (std::is_same_v<std::decay_t<decltype(end)>, std::monostate>)
              {
                  return nullptr;
              }
              else
              {
                  return end.get();
              }
          },
          backend);
    }

    std::variant<std::monostate, SharedMemoryBackend<T>, SharedMemoryBackendFallback<T>> backend;
};


}  // namespace Stockfish

#endif  // #ifndef SHM_H_INCLUDED
