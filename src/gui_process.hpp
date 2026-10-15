// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <eir_ctrl/config.hpp>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

namespace eir::ctrl {
// Owned only by the ordinary main thread; never used by control or inference.
class GuiProcess {
    pid_t pid_{-1};
public:
    GuiProcess(const Config &config, const std::string &config_path) {
        const auto display = std::getenv("DISPLAY");
        const auto wayland = std::getenv("WAYLAND_DISPLAY");
        if (!config.remote_gui ||
            config.runtime.headless ||
            ((!display || !*display) && (!wayland || !*wayland)))
            return;
        const auto script = std::filesystem::read_symlink("/proc/self/exe").parent_path() /
                            "remote_gui/remote_gui.py";
        if (!std::filesystem::is_regular_file(script)) {
            std::cerr << "Remote GUI not bundled; set EIR_REMOTE_GUI_ROOT when building\n";
            return;
        }
        auto script_path = script.string();
        auto absolute_config = std::filesystem::absolute(config_path).string();
        auto parent = std::to_string(getpid());
        char python[] = "/usr/bin/python3", config_arg[] = "--config", parent_arg[] = "--parent-pid";
        char *args[] = {python, script_path.data(), config_arg, absolute_config.data(),
                        parent_arg, parent.data(), nullptr};
        // posix_spawn avoids executing Python setup after fork in a multithreaded controller.
        const int error = posix_spawn(&pid_, python, nullptr, nullptr, args, environ);
        if (error) {
            pid_ = -1;
            std::cerr << "Cannot start remote GUI: " << std::strerror(error) << '\n';
        }
    }
    GuiProcess(const GuiProcess &) = delete;
    GuiProcess &operator=(const GuiProcess &) = delete;
    ~GuiProcess() {
        if (pid_ <= 0)
            return;
        if (waitpid(pid_, nullptr, WNOHANG) == pid_)
            return;
        kill(pid_, SIGTERM);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            const auto result = waitpid(pid_, nullptr, WNOHANG);
            if (result == pid_ || (result < 0 && errno == ECHILD))
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        kill(pid_, SIGKILL);
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
    }
};
} // namespace eir::ctrl
