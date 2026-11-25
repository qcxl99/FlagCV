/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "src/runtime/compiler_client.hpp"

#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <flagcv/handle.hpp>
#include <thread>
#include <vector>

#include "src/runtime/paths.hpp"

extern char** environ;

namespace flagcv {
namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

std::chrono::milliseconds request_timeout() {
  const char* value = std::getenv("FLAGCV_COMPILER_TIMEOUT_MS");
  if (value == nullptr) return 120s;
  const char* end = value + std::strlen(value);
  int milliseconds = 0;
  const auto parsed = std::from_chars(value, end, milliseconds);
  if (parsed.ec != std::errc{} || parsed.ptr != end || milliseconds <= 0)
    throw Error("flagcv: FLAGCV_COMPILER_TIMEOUT_MS must be a positive integer");
  return std::chrono::milliseconds(milliseconds);
}

void wait_ready(int fd, short events, Clock::time_point deadline) {
  for (;;) {
    const auto remaining = deadline - Clock::now();
    if (remaining <= Clock::duration::zero())
      throw Error("flagcv: compiler request timed out");
    const auto ms = std::chrono::ceil<std::chrono::milliseconds>(remaining);
    pollfd socket{fd, events, 0};
    const int result = ::poll(&socket, 1, static_cast<int>(ms.count()));
    if (result > 0) return;  // send/recv reports hangup or a socket error.
    if (result == 0 || errno == EINTR) continue;
    throw Error(std::string("flagcv: polling compiler socket failed: ") +
                std::strerror(errno));
  }
}
}  // namespace

CompilerClient& CompilerClient::instance() {
  static CompilerClient client;
  return client;
}

CompilerClient::~CompilerClient() { stop(); }

void CompilerClient::start() {
  if (pid_ > 0 && !reap_until(Clock::now()))
    throw Error("flagcv: previous compiler process has not exited");

  const std::string data = data_dir();
  const char* python_env = std::getenv("FLAGCV_PYTHON");
  const std::string python =
      python_env != nullptr && *python_env ? python_env : "python3";

  // The child sees the compiler package on PYTHONPATH and the data directory
  // in FLAGCV_DATA_DIR; the rest of the environment passes through untouched.
  std::vector<std::string> env_storage;
  const char* old_path = std::getenv("PYTHONPATH");
  env_storage.push_back("PYTHONPATH=" + data + "/compiler" +
                        (old_path != nullptr && *old_path
                             ? std::string(":") + old_path
                             : std::string()));
  env_storage.push_back("FLAGCV_DATA_DIR=" + data);
  for (char** e = environ; *e != nullptr; ++e) {
    if (std::strncmp(*e, "PYTHONPATH=", 11) != 0 &&
        std::strncmp(*e, "FLAGCV_DATA_DIR=", 16) != 0) {
      env_storage.emplace_back(*e);
    }
  }
  std::vector<char*> envp;
  for (auto& entry : env_storage) envp.push_back(entry.data());
  envp.push_back(nullptr);

  std::string arg_m = "-m", arg_module = "flagcv_codegen.main";
  std::string arg_python = python;
  char* argv[] = {arg_python.data(), arg_m.data(), arg_module.data(), nullptr};

  int sockets[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
    throw Error(std::string("flagcv: socketpair failed: ") +
                std::strerror(errno));
  }

  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  int rc = posix_spawn_file_actions_init(&actions);
  if (rc == 0) {
    rc = posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
    if (rc == 0)
      rc = posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
    if (rc == 0) {
      rc = posix_spawnattr_init(&attributes);
      if (rc == 0) {
        // Own a separate group so a timeout can also end a vendor assembler.
        rc = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
        if (rc == 0) rc = posix_spawnattr_setpgroup(&attributes, 0);
        if (rc == 0)
          rc = posix_spawnp(&pid_, python.c_str(), &actions, &attributes, argv,
                            envp.data());
        posix_spawnattr_destroy(&attributes);
      }
    }
    posix_spawn_file_actions_destroy(&actions);
  }
  ::close(sockets[1]);
  if (rc != 0) {
    pid_ = -1;
    ::close(sockets[0]);
    throw Error("flagcv: cannot start the compiler with '" + python +
                "': " + std::strerror(rc) +
                " (set FLAGCV_PYTHON to an interpreter with Triton installed)");
  }
  fd_ = sockets[0];
  pending_.clear();
}

bool CompilerClient::reap_until(Deadline deadline) noexcept {
  while (pid_ > 0) {
    const auto result = ::waitpid(pid_, nullptr, WNOHANG);
    if (result == pid_ || (result < 0 && errno == ECHILD)) {
      pid_ = -1;
      return true;
    }
    if ((result < 0 && errno != EINTR) || Clock::now() >= deadline) return false;
    std::this_thread::sleep_for(5ms);
  }
  return true;
}

void CompilerClient::stop(bool abandon_request) noexcept {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  pending_.clear();
  if (pid_ <= 0) return;
  if (!abandon_request && reap_until(Clock::now() + 250ms)) return;
  // Don't block indefinitely on a dying process. If reaping is delayed,
  // the next start must reap it before creating another child.
  (void)::kill(-pid_, SIGKILL);
  (void)reap_until(Clock::now() + 250ms);
}

void CompilerClient::send_line(const std::string& line, Deadline deadline) {
  std::size_t sent = 0;
  while (sent < line.size()) {
    // MSG_NOSIGNAL: a dead child must surface as an error, not SIGPIPE.

    wait_ready(fd_, POLLOUT, deadline);
    ssize_t n = ::send(fd_, line.data() + sent, line.size() - sent,
                       MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    if (n <= 0)
      throw Error(std::string("flagcv: compiler process write failed: ") +
                  std::strerror(errno));
    sent += static_cast<std::size_t>(n);
  }
}

std::string CompilerClient::receive_line(Deadline deadline) {
  for (;;) {
    auto newline = pending_.find('\n');
    if (newline != std::string::npos) {
      std::string line = pending_.substr(0, newline);
      pending_.erase(0, newline + 1);
      return line;
    }
    char buffer[65536];
    wait_ready(fd_, POLLIN, deadline);
    ssize_t n = ::recv(fd_, buffer, sizeof(buffer), MSG_DONTWAIT);
    if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    if (n <= 0)
      throw Error(
          "flagcv: compiler process exited unexpectedly (see its stderr "
          "above)");
    pending_.append(buffer, static_cast<std::size_t>(n));
  }
}

json::Value CompilerClient::request(const std::string& body) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto deadline = Clock::now() + request_timeout();
  if (fd_ < 0) start();

  const std::int64_t id = next_id_++;
  json::Value reply;
  bool ok = false;
  std::string error;
  try {
    send_line("{\"id\":" + std::to_string(id) + "," + body + "}\n", deadline);
    reply = json::Value::parse(receive_line(deadline));
    if (!reply.has("id") || !reply.at("id").is_number() ||
        reply.at("id").as_number() != static_cast<double>(id))
      throw Error("flagcv: compiler reply does not match request " +
                  std::to_string(id));
    ok = reply.at("ok").as_bool();
    if (!ok) error = reply.at("error").as_string();
  } catch (...) {
    stop(true);  // The next request uses a fresh protocol session.
    throw;
  }
  if (!ok) throw Error("flagcv: compile failed: " + error);
  return reply;
}

}  // namespace flagcv
