#include "serve/serve_options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
ninfer::serve::ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}
}

int main() {
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    const auto custom = parse({"ninfer-serve", "model.ninfer", "--chat-template", "./sharp.jinja"});
    bool empty_rejected = false;
    try { (void)parse({"ninfer-serve", "model.ninfer", "--chat-template", ""}); }
    catch (const std::invalid_argument&) { empty_rejected = true; }
    if (!defaults.chat_template_path.empty() || custom.chat_template_path != "./sharp.jinja" ||
        !empty_rejected ||
        ninfer::serve::serve_usage_text("ninfer-serve").find("--chat-template FILE") == std::string::npos) {
        std::cerr << "local chat template CLI contract failed\n";
        return 1;
    }
    std::cout << "ok\n";
}