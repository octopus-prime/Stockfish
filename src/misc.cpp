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

#include "misc.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <string_view>

#include "types.h"
#include "build_info.h"

namespace Stockfish {

namespace fs = std::filesystem;

namespace {

// Version number or dev.
constexpr std::string_view version = "dev";

// Our fancy logging facility. The trick here is to replace cin.rdbuf() and
// cout.rdbuf() with two Tie objects that tie cin and cout to a file stream. We
// can toggle the logging of std::cout and std::cin at runtime whilst preserving
// usual I/O functionality, all without changing a single line of code!
// Idea from http://groups.google.com/group/comp.lang.c++/msg/1d941c0f26ea0d81

struct Tie: public std::streambuf {

    Tie(std::streambuf* b, std::streambuf* l) :
        buf(b),
        logBuf(l) {}

    int sync() override { return logBuf->pubsync(), buf->pubsync(); }
    int overflow(int c) override { return log(buf->sputc(char(c)), "<< "); }
    int underflow() override { return buf->sgetc(); }
    int uflow() override { return log(buf->sbumpc(), ">> "); }

    std::streambuf *buf, *logBuf;

    int log(int c, const char* prefix) {

        static int last = '\n';  // Single log file

        if (last == '\n')
            logBuf->sputn(prefix, 3);

        return last = logBuf->sputc(char(c));
    }
};

class Logger {

    Logger() :
        in(std::cin.rdbuf(), file.rdbuf()),
        out(std::cout.rdbuf(), file.rdbuf()) {}
    ~Logger() { start(""); }

    std::ofstream file;
    Tie           in, out;

   public:
    static void start(const fs::path& fname) {

        static Logger l;

        if (l.file.is_open())
        {
            std::cout.rdbuf(l.out.buf);
            std::cin.rdbuf(l.in.buf);
            l.file.close();
        }

        if (!fname.empty())
        {
            l.file.open(fname, std::ifstream::out);

            if (!l.file.is_open())
            {
                std::cerr << "Unable to open debug log file " << fname << std::endl;
                exit(EXIT_FAILURE);
            }

            std::cin.rdbuf(&l.in);
            std::cout.rdbuf(&l.out);
        }
    }
};

}  // namespace


// Returns the full name of the current Stockfish version.
//
// For local dev compiles we try to append the commit SHA and
// commit date from git. If that fails only the local compilation
// date is set and "nogit" is specified:
//      Stockfish dev-YYYYMMDD-SHA
//      or
//      Stockfish dev-YYYYMMDD-nogit
//
// For releases (non-dev builds) we only include the version number:
//      Stockfish version
std::string engine_version_info() {
    std::stringstream ss;
    ss << "Stockfish " << version << std::setfill('0');

    if constexpr (version == "dev")
    {
        ss << "-" << BuildInfo::Date;
        if constexpr (BuildInfo::Modified)
            ss << "-m";
        ss << "-" << BuildInfo::Revision;
    }

    return ss.str();
}

std::string engine_info(bool to_uci) {
    return engine_version_info() + (to_uci ? "\nid author " : " by ")
         + "the Stockfish developers (see AUTHORS file)";
}


// Returns a string trying to describe the compiler we use
std::string compiler_info() {

    std::string compiler = "\nCompiled by                : ";
    compiler += BuildInfo::Compiler;
    compiler += " on Linux";

    compiler += "\nCompilation architecture   : Ryzen 9800X3D (Zen 5)";
    compiler += "\nCompilation settings       : 64bit AVX512 VNNI VBMI2 BMI2 POPCNT";

#if !defined(NDEBUG)
    compiler += " DEBUG";
#endif

    compiler += "\nCompiler __VERSION__ macro : ";
    compiler += __VERSION__;

    compiler += "\n";

    return compiler;
}


// Debug functions used mainly to collect run-time statistics
constexpr int MaxDebugSlots = 32;

namespace {

template<usize N>
struct DebugInfo {
    std::array<std::atomic<i64>, N> data = {0};

    [[nodiscard]] constexpr std::atomic<i64>& operator[](usize index) {
        assert(index < N);
        return data[index];
    }

    constexpr DebugInfo& operator=(const DebugInfo& other) {
        for (usize i = 0; i < N; i++)
            data[i].store(other.data[i].load());
        return *this;
    }
};

struct DebugExtremes: public DebugInfo<3> {
    DebugExtremes() {
        data[1] = std::numeric_limits<i64>::min();
        data[2] = std::numeric_limits<i64>::max();
    }
};

std::array<DebugInfo<2>, MaxDebugSlots>  hit;
std::array<DebugInfo<2>, MaxDebugSlots>  mean;
std::array<DebugInfo<3>, MaxDebugSlots>  stdev;
std::array<DebugInfo<6>, MaxDebugSlots>  correl;
std::array<DebugExtremes, MaxDebugSlots> extremes;

}  // namespace

bool dbg_hit_on(bool cond, int slot) {

    ++hit.at(slot)[0];
    if (cond)
        ++hit.at(slot)[1];

    return cond;
}

i64 dbg_mean_of(i64 value, int slot) {

    ++mean.at(slot)[0];
    mean.at(slot)[1] += value;

    return value;
}

i64 dbg_stdev_of(i64 value, int slot) {

    ++stdev.at(slot)[0];
    stdev.at(slot)[1] += value;
    stdev.at(slot)[2] += value * value;

    return value;
}

i64 dbg_extremes_of(i64 value, int slot) {
    ++extremes.at(slot)[0];

    i64 current_max = extremes.at(slot)[1].load();
    while (current_max < value && !extremes.at(slot)[1].compare_exchange_weak(current_max, value))
    {}

    i64 current_min = extremes.at(slot)[2].load();
    while (current_min > value && !extremes.at(slot)[2].compare_exchange_weak(current_min, value))
    {}

    return value;
}

void dbg_correl_of(i64 value1, i64 value2, int slot) {

    ++correl.at(slot)[0];
    correl.at(slot)[1] += value1;
    correl.at(slot)[2] += value1 * value1;
    correl.at(slot)[3] += value2;
    correl.at(slot)[4] += value2 * value2;
    correl.at(slot)[5] += value1 * value2;
}

void dbg_print() {

    i64  n;
    auto E   = [&n](i64 x) { return double(x) / n; };
    auto sqr = [](double x) { return x * x; };

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = hit[i][0]))
            std::cerr << "Hit #" << i << ": Total " << n << " Hits " << hit[i][1]
                      << " Hit Rate (%) " << 100.0 * E(hit[i][1]) << std::endl;

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = mean[i][0]))
            std::cerr << "Mean #" << i << ": Total " << n << " Mean " << E(mean[i][1]) << std::endl;

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = stdev[i][0]))
        {
            double r = sqrt(E(stdev[i][2]) - sqr(E(stdev[i][1])));
            std::cerr << "Stdev #" << i << ": Total " << n << " Stdev " << r << std::endl;
        }

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = extremes[i][0]))
            std::cerr << "Extremity #" << i << ": Total " << n << " Min " << extremes[i][2]
                      << " Max " << extremes[i][1] << std::endl;

    for (int i = 0; i < MaxDebugSlots; ++i)
        if ((n = correl[i][0]))
        {
            double r = (E(correl[i][5]) - E(correl[i][1]) * E(correl[i][3]))
                     / (sqrt(E(correl[i][2]) - sqr(E(correl[i][1])))
                        * sqrt(E(correl[i][4]) - sqr(E(correl[i][3]))));
            std::cerr << "Correl. #" << i << ": Total " << n << " Coefficient " << r << std::endl;
        }
}

void dbg_clear() {
    hit.fill({});
    mean.fill({});
    stdev.fill({});
    correl.fill({});
    extremes.fill({});
}

// Used to serialize access to std::cout
// to avoid multiple threads writing at the same time.
std::ostream& operator<<(std::ostream& os, SyncCout sc) {

    static std::mutex m;

    if (sc == IO_LOCK)
        m.lock();

    if (sc == IO_UNLOCK)
        m.unlock();

    return os;
}

void sync_cout_start() { std::cout << IO_LOCK; }
void sync_cout_end() { std::cout << IO_UNLOCK; }

// Hash function based on public domain MurmurHash64A, by Austin Appleby.
u64 hash_bytes(const char* data, usize size) {
    const u64 m = 0xc6a4a7935bd1e995ull;
    const int r = 47;

    u64 h = size * m;

    const char* end = data + (size & ~(usize) 7);

    for (const char* p = data; p != end; p += 8)
    {
        u64 k;
        std::memcpy(&k, p, sizeof(k));

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    if (size & 7)
    {
        u64 k = 0;
        for (int i = (size & 7) - 1; i >= 0; i--)
            k = (k << 8) | u8(end[i]);

        h ^= k;
        h *= m;
    }

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

// Trampoline helper to avoid moving Logger to misc.h
void start_logger(const fs::path& fname) { Logger::start(fname); }


// This utility allows us to handle filenames and paths with UTF8 encoding
fs::path path_from_utf8(const std::string& path) { return {path}; }


// Constructor for the CommandLine class
CommandLine::CommandLine(int _argc, char** _argv) :
    argc(_argc),
    argv(_argv) {}


// Read a numerical value from the command line
std::optional<usize> str_to_size_t(const std::string& s) {
    if (s.empty() || s[0] == '-')
        return std::nullopt;
    errno                           = 0;
    char*                    endptr = nullptr;
    const unsigned long long value  = std::strtoull(s.c_str(), &endptr, 10);
    if (errno == ERANGE || (*endptr != '\0' && !std::isspace(static_cast<unsigned char>(*endptr)))
        || value > std::numeric_limits<usize>::max())
        return std::nullopt;
    return static_cast<usize>(value);
}

// Read the given file as bytes (returns std::nullopt if the file does not exist)
std::optional<std::string> read_file_to_string(const std::string& path) {
    std::ifstream f(path, std::ios_base::binary);
    if (!f)
        return std::nullopt;
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Remove all spaces from a string
void remove_whitespace(std::string& s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }),
            s.end());
}

// Test if a string has only spaces
bool is_whitespace(std::string_view s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
}


// Return the directory where our Stockfish binary sits. This is useful,
// because when the NNUE network is not embedded in the binary, this directory
// is one of the locations where we look for the NNUE file.
fs::path CommandLine::get_binary_directory(fs::path argv0) {

    auto binaryDirectory = argv0.parent_path();
    if (binaryDirectory.empty())
        binaryDirectory = fs::path(".");
    return binaryDirectory;
}


}  // namespace Stockfish
