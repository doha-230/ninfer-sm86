#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fi = ninfer::targets::qwen3_6::frontend_internal;

int main() {
    try {
        const auto chat = fi::CompiledChatTemplate::resolve(
            "{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}"
            "{% if add_generation_prompt %}[assistant]{% endif %}");
        if (chat.capabilities().reasoning_effort.low ||
            chat.capabilities().reasoning_effort.medium ||
            chat.capabilities().reasoning_effort.xhigh) {
            throw std::runtime_error("template ignoring effort advertised effort support");
        }
        fi::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(fi::ChatPart::text_part("hello"));
        const fi::RenderedChat result = chat.render({user});
        if (result.text != "[user]hello[assistant]") {
            std::cerr << "custom Jinja produced: " << result.text << '\n';
            return 1;
        }
        std::cout << "custom Jinja basic rendering: PASS\n";

        const auto structured = fi::CompiledChatTemplate::resolve(
            "{% for m in messages %}{{ m.role }}:{{ m.content }}"
            "{% if m.tool_calls is defined %}:{{ m.tool_calls[0].function.name }}"
            "={{ m.tool_calls[0].function.arguments.city }}{% endif %}"
            "{% if m.reasoning_content is defined %}:{{ m.reasoning_content }}{% endif %}"
            "{% endfor %}|{% for t in tools %}{{ t.function.name }}{% endfor %}"
            "|{{ enable_thinking }}|{{ reasoning_effort }}");
        fi::ChatMessage assistant;
        assistant.role = ninfer::ChatRole::Assistant;
        assistant.parts.push_back(fi::ChatPart::text_part("done"));
        assistant.reasoning_content = "plan";
        assistant.tool_calls.push_back({.id = "call", .name = "weather",
                                        .arguments_json = R"({"city":"Paris"})"});
        fi::ChatRenderOptions opts;
        opts.reasoning_effort = ninfer::ReasoningEffort::Low;
        opts.tool_jsons = {R"({"function":{"name":"weather"}})"};
        const fi::RenderedChat enriched = structured.render({assistant}, opts);
        if (enriched.text != "assistant:done:weather=Paris:plan|weather|True|low") {
            std::cerr << "structured Jinja produced: " << enriched.text << '\n';
            return 1;
        }
        std::cout << "custom Jinja structured context: PASS\n";

        bool content_literal = false;
        for (const auto span : result.literal_spans) {
            content_literal |= span.begin <= 6 && span.end >= 11;
        }
        if (!content_literal) {
            std::cerr << "custom Jinja lost source content literal span\n";
            return 1;
        }
        fi::ChatRenderOptions without_suffix;
        without_suffix.add_generation_prompt = false;
        if (chat.render({user}, without_suffix).text != "[user]hello") {
            std::cerr << "custom Jinja ignored add_generation_prompt=false\n";
            return 1;
        }
        if (result.message_boundaries.size() != 2 || result.cache_boundaries.size() != 0 ||
            result.rewrite_checkpoint) {
            std::cerr << "custom Jinja fabricated unsafe cache/rewrite metadata\n";
            return 1;
        }
        fi::ChatMessage media = user;
        media.parts.push_back(fi::ChatPart::image({}));
        bool rejected_media = false;
        try { (void)chat.render({media}); }
        catch (const std::invalid_argument&) { rejected_media = true; }
        if (!rejected_media) {
            std::cerr << "custom Jinja accepted media without a placeholder\n";
            return 1;
        }
        std::cout << "custom Jinja spans, options, and missing-placeholder guard: PASS\n";

        fi::ChatRenderOptions vision_options;
        vision_options.add_generation_prompt = false;
        vision_options.enable_thinking = false;
        const auto synthetic = fi::CompiledChatTemplate::resolve(
            "{% for m in messages %}{% for p in m.content %}"
            "{% if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
            "{% elif p.type == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
            "{% else %}{{ p.text }}{% endif %}{% endfor %}{% endfor %}");
        fi::ChatMessage synthetic_media;
        synthetic_media.parts = {fi::ChatPart::text_part("literal <|image_pad|>"),
                                 fi::ChatPart::image({}), fi::ChatPart::video({})};
        const auto synthetic_result = synthetic.render({synthetic_media}, vision_options);
        if (synthetic_result.media_placeholders.size() != 2 ||
            synthetic_result.media_placeholders[0].item_index != 0 ||
            synthetic_result.media_placeholders[0].modality != fi::Modality::Image ||
            synthetic_result.media_placeholders[1].item_index != 1 ||
            synthetic_result.media_placeholders[1].modality != fi::Modality::Video ||
            synthetic_result.media_placeholders[0].bytes.begin <=
                synthetic_result.text.find("literal <|image_pad|>")) {
            throw std::runtime_error("synthetic template lost media order or literal text");
        }
        for (const auto& item : synthetic_result.media_placeholders) {
            for (const auto& span : synthetic_result.literal_spans) {
                if (span.begin < item.bytes.end && span.end > item.bytes.begin) {
                    throw std::runtime_error("synthetic media overlaps literal span");
                }
            }
        }
        std::cout << "custom Jinja image/video and literal pad: PASS\n";
        const char* sharp_path = std::getenv("NINFER_QWEN_SHARP_TEMPLATE");
        if (sharp_path != nullptr) {
        std::ifstream source_file(sharp_path);
        if (!source_file) throw std::runtime_error("cannot open Qwen-Sharp template fixture");
        const std::string source(std::istreambuf_iterator<char>{source_file}, {});
        const auto sharp = fi::CompiledChatTemplate::resolve(source);
        const auto caps = sharp.capabilities();
        if (!caps.reasoning_effort.low || !caps.reasoning_effort.medium ||
            !caps.reasoning_effort.xhigh) {
            throw std::runtime_error("Qwen-Sharp reasoning effort not advertised");
        }
        fi::ChatRenderOptions effort_options;
        effort_options.add_generation_prompt = false;
        effort_options.reasoning_effort = ninfer::ReasoningEffort::Low;
        const auto low_prompt = sharp.render({user}, effort_options).text;
        effort_options.reasoning_effort = ninfer::ReasoningEffort::Medium;
        const auto medium_prompt = sharp.render({user}, effort_options).text;
        effort_options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
        const auto xhigh_prompt = sharp.render({user}, effort_options).text;
        if (low_prompt == medium_prompt || medium_prompt == xhigh_prompt ||
            low_prompt == xhigh_prompt) {
            throw std::runtime_error("Qwen-Sharp ignored a reasoning effort level");
        }
        std::cout << "Qwen-Sharp reasoning effort levels: PASS\n";
        const fi::RenderedChat single = sharp.render({media}, vision_options);
        const auto& first = single.media_placeholders;
        if (first.size() != 1 || first[0].modality != fi::Modality::Image ||
            first[0].item_index != 0 ||
            single.text.substr(first[0].bytes.begin, first[0].bytes.end - first[0].bytes.begin) !=
                "<|image_pad|>" ||
            single.text.find("<|vision_start|><|image_pad|><|vision_end|>") ==
                std::string::npos) {
            throw std::runtime_error("Qwen-Sharp single image placeholder missing or incorrect");
        }
        std::cout << "Qwen-Sharp image: PASS\n";

        fi::ChatMessage mixed;
        mixed.role = ninfer::ChatRole::User;
        mixed.parts = {fi::ChatPart::text_part("literal <|image_pad|> "),
                       fi::ChatPart::image({}), fi::ChatPart::text_part(" then "),
                       fi::ChatPart::video({}), fi::ChatPart::image({})};
        const fi::RenderedChat multiple = sharp.render({mixed}, vision_options);
        if (multiple.media_placeholders.size() != 3 ||
            multiple.media_placeholders[0].item_index != 0 ||
            multiple.media_placeholders[0].modality != fi::Modality::Image ||
            multiple.media_placeholders[1].item_index != 1 ||
            multiple.media_placeholders[1].modality != fi::Modality::Video ||
            multiple.media_placeholders[2].item_index != 2 ||
            multiple.media_placeholders[2].modality != fi::Modality::Image ||
            multiple.text.find("literal <|image_pad|> ") == std::string::npos) {
            throw std::runtime_error("Qwen-Sharp mixed-media order or literal text incorrect");
        }
        for (const auto& placeholder : multiple.media_placeholders) {
            for (const auto& span : multiple.literal_spans) {
                if (span.begin < placeholder.bytes.end && span.end > placeholder.bytes.begin) {
                    throw std::runtime_error("Qwen-Sharp literal span overlaps a media placeholder");
                }
            }
            const auto pad = placeholder.modality == fi::Modality::Image ?
                                 "<|image_pad|>" : "<|video_pad|>";
            if (multiple.text.substr(placeholder.bytes.begin,
                                     placeholder.bytes.end - placeholder.bytes.begin) != pad ||
                placeholder.bytes.begin <= multiple.text.find("literal <|image_pad|> ")) {
                throw std::runtime_error("Qwen-Sharp mislabeled a literal pad as media");
            }
        }
        std::cout << "Qwen-Sharp ordered mixed media and literal pad: PASS\n";

        fi::ChatRenderOptions numbered = vision_options;
        numbered.add_vision_id = true;
        const fi::RenderedChat across_messages = sharp.render({media, mixed}, numbered);
        if (across_messages.media_placeholders.size() != 4 ||
            across_messages.text.find("Picture 1: <|vision_start|><|image_pad|>") == std::string::npos ||
            across_messages.text.find("Picture 2: <|vision_start|><|image_pad|>") == std::string::npos ||
            across_messages.text.find("Video 1: <|vision_start|><|video_pad|>") == std::string::npos ||
            across_messages.text.find("Picture 3: <|vision_start|><|image_pad|>") == std::string::npos) {
            throw std::runtime_error("Qwen-Sharp numbered media across messages failed");
        }
        for (std::size_t i = 0; i < across_messages.media_placeholders.size(); ++i) {
            if (across_messages.media_placeholders[i].item_index != i) {
                throw std::runtime_error("Qwen-Sharp changed media index across messages");
            }
        }
        std::cout << "Qwen-Sharp numbered media across messages: PASS\n";
        } else {
            std::cout << "Qwen-Sharp local template: SKIP (set NINFER_QWEN_SHARP_TEMPLATE)\n";
        }

        // The template must emit each input exactly once, unmodified and in input order.
        for (const std::string& template_source : {
                 std::string("{% for m in messages %}{% for p in m.content %}"
                             "{% if p.type == 'image' %}<|image_pad|><|image_pad|>{% endif %}"
                             "{% endfor %}{% endfor %}"),
                 std::string("{% for m in messages %}{% for p in m.content %}"
                             "{% if p.type == 'image' %}{{ '<|image_pad|>' | upper }}{% endif %}"
                             "{% endfor %}{% endfor %}"),
                 std::string("{% for m in messages %}{{ m.role }}{% endfor %}"),
                 std::string("<|image_pad|>{% for m in messages %}{% for p in m.content %}"
                             "{% if p.type == 'image' %}<|image_pad|>{% endif %}"
                             "{% endfor %}{% endfor %}")}) {
            bool rejected = false;
            try { (void)fi::CompiledChatTemplate::resolve(template_source).render({media}); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("accepted unsafe image template");
        }
        std::cout << "custom Jinja transformed, duplicated, omitted and static media guards: PASS\n";

        fi::ChatMessage two_images;
        two_images.parts = {fi::ChatPart::image({}), fi::ChatPart::image({})};
        const auto reversed = fi::CompiledChatTemplate::resolve(
            "{% for m in messages %}{% for p in m.content[::-1] %}"
            "{% if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
            "{% else %}{{ p.text }}{% endif %}{% endfor %}{% endfor %}");
        bool reordered_rejected = false;
        try { (void)reversed.render({two_images}, vision_options); }
        catch (const std::invalid_argument&) { reordered_rejected = true; }
        if (!reordered_rejected) throw std::runtime_error("accepted reordered image items");
        std::cout << "custom Jinja reordered media guard: PASS\n";

        bool malformed_rejected = false;
        try { (void)fi::CompiledChatTemplate::resolve("{% if messages %}"); }
        catch (const std::invalid_argument&) { malformed_rejected = true; }
        if (!malformed_rejected) {
            std::cerr << "custom Jinja accepted invalid template syntax\n";
            return 1;
        }
        std::cout << "custom Jinja parse errors: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "custom Jinja basic rendering: FAIL: " << error.what() << '\n';
        return 1;
    }
}
