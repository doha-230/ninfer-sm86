#include "serve/openai_common.h"

#include <iostream>
#include <string>

int main() {
    const std::string advertised = "qwen3.8-27b";
    const std::string requested = "agent-configured-alias";
    const std::string resolved = ninfer::serve::resolve_openai_model(requested, advertised);
    if (resolved != advertised || ninfer::serve::resolve_openai_model(advertised, advertised) != advertised) {
        std::cerr << "request alias did not resolve to the loaded model\n";
        return 1;
    }
    std::cout << "ok\n";
}
