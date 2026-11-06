# Contributing to Stockfish

Welcome to the Stockfish project! We are excited that you are interested in
contributing. This document outlines the guidelines and steps to follow when
making contributions to Stockfish.

## Table of Contents

- [Building Stockfish](#building-stockfish)
- [Making Contributions](#making-contributions)
  - [Reporting Issues](#reporting-issues)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style](#code-style)
- [Community and Communication](#community-and-communication)
- [License](#license)

## Building Stockfish

This branch supports only Linux on the Ryzen 9800X3D, using GCC 14+ or Clang 19+.
See [the build instructions](README.md#compiling-stockfish) for the fixed-target
CMake build and required test tools. Architecture and operating-system
alternatives should not be reintroduced.

Run `ctest --test-dir build --output-on-failure` after changes. Refactorings must
preserve the bench signature (1556650), including move order and network output.

## Making Contributions

### Reporting Issues

If you find a bug, please open an issue on the
[issue tracker][issue-tracker-link]. Be sure to include relevant information
like your operating system, build environment, and a detailed description of the
problem.

_Please note that Stockfish's development is not focused on adding new features.
Thus any issue regarding missing features will potentially be closed without
further discussion._

### Submitting Pull Requests

- Functional changes need to be tested on fishtest. See
  [Creating my First Test][creating-my-first-test] for more details.
  The accompanying pull request should include a link to the test results and
  the new bench.

- Non-functional changes (e.g. refactoring, code style, documentation) do not
  need to be tested on fishtest, unless they might impact performance.

- Provide a clear and concise description of the changes in the pull request
  description.

_First time contributors should add their name to [AUTHORS](./AUTHORS)._

_Stockfish's development is not focused on adding new features. Thus any pull
request introducing new features will potentially be closed without further
discussion._

## Code Style

Changes to Stockfish C++ code should respect our coding style defined by
[.clang-format](.clang-format). You can format your changes by running
`cmake --build build --target format`. This target is available when
clang-format version 20 is installed at configuration time.

## Navigate

For experienced Git users who frequently use git blame, it is recommended to
configure the blame.ignoreRevsFile setting.
This setting is useful for excluding noisy formatting commits.

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Community and Communication

- Join the [Stockfish discord][discord-link] to discuss ideas, issues, and
  development.
- Participate in the [Stockfish GitHub discussions][discussions-link] for
  broader conversations.

## License

By contributing to Stockfish, you agree that your contributions will be licensed
under the GNU General Public License v3.0. See [Copying.txt][copying-link] for
more details.

Thank you for contributing to Stockfish and helping us make it even better!

[copying-link]:           https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt
[discord-link]:           https://discord.gg/GWDRS3kU6R
[discussions-link]:       https://github.com/official-stockfish/Stockfish/discussions/new
[creating-my-first-test]: https://github.com/official-stockfish/fishtest/wiki/Creating-my-first-test#create-your-test
[issue-tracker-link]:     https://github.com/official-stockfish/Stockfish/issues
