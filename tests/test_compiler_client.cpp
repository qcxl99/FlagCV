// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <flagcv/handle.hpp>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

#include "src/runtime/compiler_client.hpp"

namespace {
using flagcv::CompilerClient;
using namespace std::chrono_literals;
constexpr auto identify = R"("op":"identify","backend":"testfixture")";

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void expect_error(const std::string& request, std::string_view message) {
  try {
    CompilerClient::instance().request(request);
  } catch (const flagcv::Error& error) {
    require(std::string_view(error.what()).find(message) != std::string_view::npos, error.what());
    return;
  }
  throw std::runtime_error("request unexpectedly succeeded");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected compiler fixture directory");
    auto& client = CompilerClient::instance();
    const auto pid_string = client.request(identify).at("identity").as_string();
    expect_error(R"("op":"compile","backend":"testfixture","kind":"compile_error")",
                 "deliberate compile error");
    require(client.request(identify).at("identity").as_string() == pid_string,
            "a normal compiler error should preserve the process");

    require(::setenv("FLAGCV_COMPILER_TIMEOUT_MS", "300", 1) == 0,
            "cannot configure timeout for fixture");
    const auto start = std::chrono::steady_clock::now();
    expect_error(R"("op":"compile","backend":"testfixture","kind":"hang")", "timed out");
    require(std::chrono::steady_clock::now() - start < 3s, "compiler request exceeded its bound");
    std::ifstream marker(std::string(argv[1]) + "/compiler.pid");
    std::string timed_out;
    marker >> timed_out;
    require(timed_out == pid_string, "compiler never entered the hang fixture");
    const pid_t pid = static_cast<pid_t>(std::stol(timed_out));
    require(::waitpid(pid, nullptr, WNOHANG) == -1 && errno == ECHILD,
            "timed-out compiler was not reaped by the client");

    require(::setenv("FLAGCV_COMPILER_TIMEOUT_MS", "3000", 1) == 0,
            "cannot configure restart timeout");
    require(client.request(identify).at("identity").as_string() != pid_string,
            "next request did not restart the compiler");
    // The child will block after seeing EOF. Exiting this test exercises
    // destructor cleanup, protected by the independent CTest timeout.
    client.request(R"("op":"compile","backend":"testfixture","kind":"shutdown_hang")");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
