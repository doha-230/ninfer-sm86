#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include "targets/qwen3_6/impl/frontend/digest.h"
#include "text/jinja.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr Sha256Digest kThinkingToggleTemplateDigest{
    0xe8, 0x4f, 0x32, 0xa2, 0x3f, 0xdd, 0xa2, 0x76, 0x89, 0xf8, 0x68, 0xaa, 0x4a, 0x1a, 0x56, 0x21,
    0xf4, 0x11, 0x33, 0xe5, 0x1a, 0x48, 0xd7, 0xf3, 0xef, 0xcb, 0xea, 0x28, 0x39, 0x57, 0x42, 0x59,
};

constexpr Sha256Digest kReasoningEffortTemplateDigest{
    0xc3, 0xcf, 0x9e, 0x34, 0xab, 0xf4, 0xf9, 0xe3, 0x6c, 0x2d, 0x72, 0x16, 0x5a, 0xa9, 0xc1, 0x32,
    0xd3, 0xe2, 0xa7, 0x25, 0xb6, 0xc2, 0x58, 0x6a, 0xaa, 0x3a, 0x8a, 0xf9, 0xd7, 0xa8, 0x10, 0x41,
};

constexpr std::string_view kLowReasoningInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

bool is_instruction_role(ChatRole role) noexcept {
    return role == ChatRole::System || role == ChatRole::Developer;
}

void validate_instruction_message(const ChatMessage& message) {
    if (message.has_media()) {
        throw std::invalid_argument(
            "system and developer messages cannot contain images or videos");
    }
    if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
        !message.tool_call_id.empty()) {
        throw std::invalid_argument("system and developer messages may contain only text content");
    }
}

void append_literal_span(std::vector<ByteSpan>& spans, ByteSpan span) {
    if (span.begin == span.end) { return; }
    if (!spans.empty() && spans.back().end == span.begin) {
        spans.back().end = span.end;
        return;
    }
    if (!spans.empty() && spans.back().end > span.begin) {
        throw std::logic_error("rendered literal byte spans overlap");
    }
    spans.push_back(span);
}

class RenderBuilder {
public:
    void append_template(std::string_view text) { fragment_.text += text; }

    void append_literal(std::string_view text) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        append_literal_span(fragment_.literal_spans, ByteSpan{begin, fragment_.text.size()});
    }

    void append_media_placeholder(std::string_view text, Modality modality,
                                  std::size_t item_index) {
        const std::size_t begin = fragment_.text.size();
        fragment_.text += text;
        fragment_.media_placeholders.push_back(MediaPlaceholderByteSpec{
            .bytes      = ByteSpan{begin, fragment_.text.size()},
            .modality   = modality,
            .item_index = item_index,
        });
    }

    void append(RenderedFragment fragment) {
        const std::size_t offset = fragment_.text.size();
        fragment_.text += fragment.text;
        for (const ByteSpan span : fragment.literal_spans) {
            append_literal_span(fragment_.literal_spans,
                                ByteSpan{offset + span.begin, offset + span.end});
        }
        for (MediaPlaceholderByteSpec placeholder : fragment.media_placeholders) {
            placeholder.bytes.begin += offset;
            placeholder.bytes.end += offset;
            fragment_.media_placeholders.push_back(placeholder);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return fragment_.text.size(); }

    [[nodiscard]] RenderedFragment release() && { return std::move(fragment_); }

private:
    RenderedFragment fragment_;
};

RenderedFragment literal_fragment(std::string text) {
    RenderBuilder builder;
    builder.append_literal(text);
    return std::move(builder).release();
}

std::pair<std::size_t, std::size_t> trim_ascii_whitespace_bounds(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return {begin, end};
}

RenderedFragment slice_fragment(const RenderedFragment& source, std::size_t begin,
                                std::size_t end) {
    if (begin > end || end > source.text.size()) {
        throw std::logic_error("rendered fragment slice is out of range");
    }
    RenderedFragment result;
    result.text = source.text.substr(begin, end - begin);
    for (const ByteSpan span : source.literal_spans) {
        const std::size_t clipped_begin = std::max(span.begin, begin);
        const std::size_t clipped_end   = std::min(span.end, end);
        if (clipped_begin < clipped_end) {
            append_literal_span(result.literal_spans,
                                ByteSpan{clipped_begin - begin, clipped_end - begin});
        }
    }
    for (MediaPlaceholderByteSpec placeholder : source.media_placeholders) {
        if (placeholder.bytes.end <= begin || placeholder.bytes.begin >= end) { continue; }
        if (placeholder.bytes.begin < begin || placeholder.bytes.end > end) {
            throw std::logic_error("rendered fragment slice intersects a media placeholder");
        }
        placeholder.bytes.begin -= begin;
        placeholder.bytes.end -= begin;
        result.media_placeholders.push_back(placeholder);
    }
    return result;
}

RenderedFragment trim_ascii_whitespace(const RenderedFragment& fragment) {
    const auto [begin, end] = trim_ascii_whitespace_bounds(fragment.text);
    return slice_fragment(fragment, begin, end);
}

bool starts_with(const std::string& text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

long last_real_user_query(const std::vector<ChatMessage>& messages) {
    long trailing_tool_query = -1;
    for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
        const ChatMessage& message = messages[static_cast<std::size_t>(i)];
        if (message.role == ChatRole::Tool && trailing_tool_query < 0) { trailing_tool_query = i; }
        if (message.role != ChatRole::User) { continue; }
        const RenderedFragment content = trim_ascii_whitespace(message.rendered_content());
        if (!(starts_with(content.text, "<tool_response>") &&
              ends_with(content.text, "</tool_response>"))) {
            return i;
        }
    }
    // A truncated imported history may begin at the tool-result turn. That result is the current
    // query even though the user message that initiated the omitted tool call is not visible.
    if (trailing_tool_query >= 0) { return trailing_tool_query; }
    throw std::invalid_argument("no user query found in chat messages");
}

// Split an assistant turn into (reasoning, content) exactly as the Qwen3.6 jinja
// does when reasoning_content is not provided: reasoning is the text between the
// last <think> and the first </think>; content is everything after the last
// </think>. When there is no </think> the whole thing is content and reasoning is
// empty.
struct ThinkParts {
    RenderedFragment reasoning;
    RenderedFragment content;
};

ThinkParts derive_think_parts(const RenderedFragment& content) {
    ThinkParts parts;
    const std::size_t first_close = content.text.find("</think>");
    if (first_close == std::string::npos) {
        parts.content = content;
        return parts;
    }
    // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
    std::size_t before_end = first_close;
    while (before_end > 0 && content.text[before_end - 1] == '\n') { --before_end; }
    const std::size_t last_open = content.text.substr(0, before_end).rfind("<think>");
    std::size_t reasoning_begin =
        last_open == std::string::npos ? 0 : last_open + std::string("<think>").size();
    while (reasoning_begin < before_end && content.text[reasoning_begin] == '\n') {
        ++reasoning_begin;
    }
    parts.reasoning = slice_fragment(content, reasoning_begin, before_end);
    // content = content.split('</think>')[-1].lstrip('\n')
    const std::size_t last_close = content.text.rfind("</think>");
    std::size_t content_begin    = last_close + std::string("</think>").size();
    while (content_begin < content.text.size() && content.text[content_begin] == '\n') {
        ++content_begin;
    }
    parts.content = slice_fragment(content, content_begin, content.text.size());
    return parts;
}

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

std::string tojson_text(const OrderedJson& value) {
    if (value.is_array()) {
        std::string rendered = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += tojson_text(value[index]);
        }
        rendered += "]";
        return rendered;
    }
    if (value.is_object()) {
        std::string rendered = "{";
        std::size_t index    = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += OrderedJson(it.key()).dump();
            rendered += ": ";
            rendered += tojson_text(it.value());
        }
        rendered += "}";
        return rendered;
    }
    return value.dump();
}

std::string parameter_text(const OrderedJson& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return tojson_text(value);
}

RenderedFragment render_tool_call(const ToolCall& call, bool allow_empty_arguments) {
    RenderBuilder rendered;
    if (allow_empty_arguments && call.arguments_json.empty()) {
        rendered.append_template("<tool_call>\n<function=");
        rendered.append_literal(call.name);
        rendered.append_template(">\n</function>\n</tool_call>");
        return std::move(rendered).release();
    }
    OrderedJson args = OrderedJson::parse(call.arguments_json);
    if (!args.is_object()) {
        throw std::invalid_argument("tool call arguments must be a JSON object");
    }

    rendered.append_template("<tool_call>\n<function=");
    rendered.append_literal(call.name);
    rendered.append_template(">\n");
    for (auto it = args.begin(); it != args.end(); ++it) {
        rendered.append_template("<parameter=");
        rendered.append_literal(it.key());
        rendered.append_template(">\n");
        rendered.append_literal(parameter_text(it.value()));
        rendered.append_template("\n</parameter>\n");
    }
    rendered.append_template("</function>\n</tool_call>");
    return std::move(rendered).release();
}

struct RenderedToolsSystemBlock {
    RenderedFragment fragment;
    std::vector<std::size_t> tool_boundaries;
    std::optional<std::size_t> instruction_begin;
};

RenderedToolsSystemBlock render_tools_system_block(const std::vector<std::string>& tool_jsons,
                                                   const RenderedFragment& leading_instruction,
                                                   std::string_view reasoning_instructions) {
    RenderedToolsSystemBlock out;
    RenderBuilder rendered;
    out.tool_boundaries.reserve(tool_jsons.size());
    rendered.append_template("<|im_start|>system\n");
    if (!reasoning_instructions.empty()) {
        rendered.append_template(reasoning_instructions);
        rendered.append_template("\n\n");
    }
    rendered.append_template("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (const std::string& tool : tool_jsons) {
        rendered.append_template("\n");
        rendered.append_literal(tojson_text(OrderedJson::parse(tool)));
        out.tool_boundaries.push_back(rendered.size());
    }
    rendered.append_template("\n</tools>");
    rendered.append_template(kToolInstructions);
    if (!leading_instruction.text.empty()) {
        rendered.append_template("\n\n");
        out.instruction_begin = rendered.size();
        rendered.append(leading_instruction);
    }
    rendered.append_template("<|im_end|>\n");
    out.fragment = std::move(rendered).release();
    return out;
}

std::string_view resolve_reasoning_instructions(ChatTemplateSemantics semantics,
                                                const ChatRenderOptions& options) {
    if (semantics == ChatTemplateSemantics::ThinkingToggle) {
        if (options.reasoning_effort) {
            throw std::invalid_argument("loaded chat template does not support reasoning effort");
        }
        return {};
    }
    if (!options.enable_thinking) {
        if (options.reasoning_effort) {
            throw std::invalid_argument(
                "reasoning effort cannot be combined with disabled thinking");
        }
        return {};
    }

    switch (options.reasoning_effort.value_or(ReasoningEffort::XHigh)) {
    case ReasoningEffort::Low:
        return kLowReasoningInstructions;
    case ReasoningEffort::Medium:
        return {};
    case ReasoningEffort::XHigh:
        return kXHighReasoningInstructions;
    }
    throw std::invalid_argument("invalid reasoning effort");
}

RenderedChat render_custom_jinja(const text::JinjaTemplate& jinja,
                                 const std::vector<ChatMessage>& messages,
                                 const ChatRenderOptions& options) {
    OrderedJson context;
    context["messages"] = OrderedJson::array();
    OrderedJson shadow;
    shadow["messages"] = OrderedJson::array();
    struct MediaOrigin {
        Modality modality;
        std::size_t index;
        std::string shadow_text;
        std::size_t pad_begin;
        std::string_view pad;
    };
    std::vector<MediaOrigin> media;
    std::vector<text::TemplateInputRegion> regions;
    std::size_t image_count = 0;
    std::size_t video_count = 0;
    for (const ChatMessage& message : messages) {
        const std::size_t message_index = context["messages"].size();
        if (message.has_media() && is_instruction_role(message.role)) {
            throw std::invalid_argument("system and developer messages cannot contain images or videos");
        }
        OrderedJson value;
        switch (message.role) {
        case ChatRole::System: value["role"] = "system"; break;
        case ChatRole::Developer: value["role"] = "developer"; break;
        case ChatRole::User: value["role"] = "user"; break;
        case ChatRole::Assistant: value["role"] = "assistant"; break;
        case ChatRole::Tool: value["role"] = "tool"; break;
        default: throw std::invalid_argument("unsupported chat role value");
        }
        if (!message.has_media()) {
            std::string content;
            for (const ChatPart& part : message.parts) { content += part.text; }
            value["content"] = std::move(content);
        } else {
            value["content"] = OrderedJson::array();
            for (const ChatPart& part : message.parts) {
                if (part.kind == ChatPartKind::Text) {
                    value["content"].push_back({{"type", "text"}, {"text", part.text}});
                } else {
                    value["content"].push_back(
                        {{"type", part.kind == ChatPartKind::Image ? "image" : "video"}});
                }
            }
        }
        if (!message.reasoning_content.empty()) {
            value["reasoning_content"] = message.reasoning_content;
        }
        if (!message.tool_call_id.empty()) { value["tool_call_id"] = message.tool_call_id; }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = OrderedJson::array();
            for (const ToolCall& call : message.tool_calls) {
                value["tool_calls"].push_back(
                    {{"id", call.id}, {"type", "function"},
                     {"function", {{"name", call.name},
                                   {"arguments", OrderedJson::parse(call.arguments_json)}}}});
            }
        }
        context["messages"].push_back(value);
        if (message.has_media()) {
            OrderedJson shadow_value = std::move(value);
            for (std::size_t part_index = 0; part_index < message.parts.size(); ++part_index) {
                const ChatPart& part = message.parts[part_index];
                if (part.kind == ChatPartKind::Text) { continue; }
                if (media.size() >= std::numeric_limits<std::uint32_t>::max()) {
                    throw std::invalid_argument("too many chat media items");
                }
                const bool image = part.kind == ChatPartKind::Image;
                const std::size_t number = image ? ++image_count : ++video_count;
                const std::string_view pad = image ? "<|image_pad|>" : "<|video_pad|>";
                std::string prefix;
                if (options.add_vision_id) {
                    prefix = std::string(image ? "Picture " : "Video ") +
                             std::to_string(number) + ": ";
                }
                const std::size_t pad_begin = prefix.size() + std::string_view("<|vision_start|>").size();
                const std::uint32_t tag = static_cast<std::uint32_t>(media.size() + 1);
                media.push_back({image ? Modality::Image : Modality::Video, media.size(),
                                 prefix + "<|vision_start|>" + std::string(pad) + "<|vision_end|>",
                                 pad_begin, pad});
                shadow_value["content"][part_index] =
                    {{"type", "text"}, {"text", media.back().shadow_text}};
                regions.push_back({"/messages/" + std::to_string(message_index) + "/content/" +
                                       std::to_string(part_index) + "/text", tag});
            }
            shadow["messages"].push_back(std::move(shadow_value));
        } else {
            shadow["messages"].push_back(std::move(value));
        }
    }
    context["add_vision_id"] = options.add_vision_id;
    context["add_generation_prompt"] = options.add_generation_prompt &&
        options.continuation != PromptContinuationMode::ContinueFinalAssistant;
    context["enable_thinking"] = options.enable_thinking;
    context["preserve_thinking"] = options.preserve_thinking.value_or(false);
    if (options.reasoning_effort) {
        switch (*options.reasoning_effort) {
        case ReasoningEffort::Low: context["reasoning_effort"] = "low"; break;
        case ReasoningEffort::Medium: context["reasoning_effort"] = "medium"; break;
        case ReasoningEffort::XHigh: context["reasoning_effort"] = "xhigh"; break;
        }
    }
    context["tools"] = OrderedJson::array();
    for (const std::string& tool : options.tool_jsons) {
        context["tools"].push_back(OrderedJson::parse(tool));
    }
    const text::TemplateOutput output = jinja.render(context);
    RenderedChat result;
    result.text = output.text;
    for (const text::ByteSpan span : output.literal_spans) {
        result.literal_spans.push_back({span.begin, span.end});
    }
    if (!media.empty()) {
        // The real template may emit a constant vision marker rather than echo a tagged
        // input field (as Qwen-Sharp does). Render a second, equivalent text-only context:
        // each media item becomes its exact expected output, with its own input-origin tag.
        // Only identical output and intact, ordered tagged bytes establish provenance.
        for (auto it = context.begin(); it != context.end(); ++it) {
            if (it.key() != "messages") { shadow[it.key()] = it.value(); }
        }
        text::TemplateRenderOptions shadow_options;
        shadow_options.regions = regions;
        const text::TemplateOutput traced = jinja.render(shadow, shadow_options);
        if (traced.text != output.text) {
            throw std::invalid_argument("chat template transformed or omitted media placeholders");
        }
        std::vector<std::vector<std::optional<std::size_t>>> positions;
        for (const MediaOrigin& origin : media) { positions.emplace_back(origin.shadow_text.size()); }
        for (const text::TemplateOutputRegion& region : traced.regions) {
            if (region.tag == 0 || region.tag > media.size() || !region.source_offset ||
                region.end > traced.text.size() || region.begin > region.end) {
                throw std::invalid_argument("chat template lost media placeholder provenance");
            }
            const std::size_t index = region.tag - 1;
            const MediaOrigin& origin = media[index];
            const std::size_t length = region.end - region.begin;
            if (*region.source_offset > origin.shadow_text.size() ||
                length > origin.shadow_text.size() - *region.source_offset ||
                traced.text.compare(region.begin, length, origin.shadow_text,
                                    *region.source_offset, length) != 0) {
                throw std::invalid_argument("chat template transformed a media placeholder");
            }
            for (std::size_t offset = 0; offset < length; ++offset) {
                const std::size_t source = *region.source_offset + offset;
                if (positions[index][source]) {
                    throw std::invalid_argument("chat template duplicated a media placeholder");
                }
                positions[index][source] = region.begin + offset;
            }
        }
        for (std::size_t index = 0; index < media.size(); ++index) {
            const MediaOrigin& origin = media[index];
            const auto& bytes = positions[index];
            const std::size_t begin = origin.pad_begin;
            const std::size_t end = begin + origin.pad.size();
            if (end > bytes.size() || !bytes[begin]) {
                throw std::invalid_argument("chat template omitted a media placeholder");
            }
            // Require the entire marker and its framing (including optional vision id)
            // from the same item, once, without rearrangement or internal insertion.
            const std::size_t whole_begin = *bytes.front();
            for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
                if (!bytes[offset] || *bytes[offset] != whole_begin + offset) {
                    throw std::invalid_argument("chat template transformed or duplicated media");
                }
            }
            const std::size_t pad_at = *bytes[begin];
            if (!result.media_placeholders.empty() &&
                result.media_placeholders.back().bytes.end > pad_at) {
                throw std::invalid_argument("chat template reordered media placeholders");
            }
            result.media_placeholders.push_back({{pad_at, pad_at + origin.pad.size()},
                                                 origin.modality, origin.index});
        }
        // An untagged template-owned pad is not another media item. Reject it rather
        // than handing a raw vision marker to tokenization; user text is literal.
        for (const std::string_view pad : {std::string_view("<|image_pad|>"),
                                           std::string_view("<|video_pad|>")}) {
            std::size_t cursor = 0;
            while ((cursor = result.text.find(pad, cursor)) != std::string::npos) {
                const bool mapped = std::any_of(result.media_placeholders.begin(),
                                                result.media_placeholders.end(),
                                                [&](const MediaPlaceholderByteSpec& item) {
                                                    return item.bytes.begin == cursor &&
                                                           item.bytes.end == cursor + pad.size();
                                                });
                const bool literal = std::any_of(result.literal_spans.begin(),
                                                 result.literal_spans.end(),
                                                 [&](ByteSpan span) {
                                                     return span.begin <= cursor &&
                                                            cursor + pad.size() <= span.end;
                                                 });
                if (!mapped && !literal) {
                    throw std::invalid_argument("chat template emitted an unowned media marker");
                }
                cursor += pad.size();
            }
        }
    }
    // Arbitrary templates can reorder/transform messages. There is no generally valid byte
    // frontier for incremental caching or turn rewrite; leave those hints unresolved rather
    // than claiming a prefix that may change when another message is appended.
    result.message_boundaries.resize(messages.size() + 1U);
    result.cache_boundaries.resize(options.cache_markers.size());
    return result;
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    for (const ChatPart& part : parts) {
        if (part.kind != ChatPartKind::Text) { return true; }
    }
    return false;
}

RenderedFragment ChatMessage::rendered_content(bool add_vision_id, int* image_count,
                                               int* video_count, std::size_t* media_count,
                                               std::vector<std::size_t>* part_boundaries) const {
    int local_images        = 0;
    int local_videos        = 0;
    std::size_t local_media = 0;
    int& images             = image_count == nullptr ? local_images : *image_count;
    int& videos             = video_count == nullptr ? local_videos : *video_count;
    std::size_t& media      = media_count == nullptr ? local_media : *media_count;
    RenderBuilder out;
    if (part_boundaries != nullptr) {
        part_boundaries->clear();
        part_boundaries->reserve(parts.size());
    }
    for (const ChatPart& part : parts) {
        switch (part.kind) {
        case ChatPartKind::Text:
            out.append_literal(part.text);
            break;
        case ChatPartKind::Image:
            ++images;
            if (add_vision_id) { out.append_template("Picture " + std::to_string(images) + ": "); }
            out.append_template("<|vision_start|>");
            out.append_media_placeholder("<|image_pad|>", Modality::Image, media++);
            out.append_template("<|vision_end|>");
            break;
        case ChatPartKind::Video:
            ++videos;
            if (add_vision_id) { out.append_template("Video " + std::to_string(videos) + ": "); }
            out.append_template("<|vision_start|>");
            out.append_media_placeholder("<|video_pad|>", Modality::Video, media++);
            out.append_template("<|vision_end|>");
            break;
        }
        if (part_boundaries != nullptr) { part_boundaries->push_back(out.size()); }
    }
    return std::move(out).release();
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source) {
    const Sha256Digest digest = sha256(source);
    if (digest == kThinkingToggleTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle);
    }
    if (digest == kReasoningEffortTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ReasoningEffort);
    }
    auto jinja = std::make_shared<const text::JinjaTemplate>(
        std::string(source), "frontend/chat_template.jinja");
    PromptCapabilities capabilities;
    if (source.find("reasoning_effort") != std::string_view::npos) {
        // No general introspection can prove an arbitrary Jinja template uses a variable.
        // Advertise effort only if the parsed template demonstrably distinguishes all
        // three supplied levels on a representative user turn. Probe once at load time.
        try {
            ChatMessage probe;
            probe.parts.push_back(ChatPart::text_part("hello"));
            ChatRenderOptions options;
            options.reasoning_effort = ReasoningEffort::Low;
            const std::string low = render_custom_jinja(*jinja, {probe}, options).text;
            options.reasoning_effort = ReasoningEffort::Medium;
            const std::string medium = render_custom_jinja(*jinja, {probe}, options).text;
            options.reasoning_effort = ReasoningEffort::XHigh;
            const std::string xhigh = render_custom_jinja(*jinja, {probe}, options).text;
            if (low != medium && low != xhigh && medium != xhigh) {
                capabilities.reasoning_effort.low = true;
                capabilities.reasoning_effort.medium = true;
                capabilities.reasoning_effort.xhigh = true;
                capabilities.reasoning_effort.default_effort = ReasoningEffort::Medium;
            }
        } catch (const std::invalid_argument&) {
            // A template requiring a different conversation may still render normally;
            // do not assert effort support based on a probe it could not complete.
        }
    }
    return CompiledChatTemplate(std::move(jinja), capabilities);
}

PromptCapabilities CompiledChatTemplate::capabilities() const noexcept {
    if (semantics_ == ChatTemplateSemantics::CustomJinja) {
        PromptCapabilities result = custom_capabilities_;
        result.enable_thinking = true;
        return result;
    }
    PromptCapabilities result;
    result.enable_thinking = true;
    if (semantics_ == ChatTemplateSemantics::ReasoningEffort) {
        result.reasoning_effort.low            = true;
        result.reasoning_effort.medium         = true;
        result.reasoning_effort.xhigh          = true;
        result.reasoning_effort.default_effort = ReasoningEffort::XHigh;
    }
    return result;
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options) const {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }
    if (semantics_ == ChatTemplateSemantics::CustomJinja) {
        return render_custom_jinja(*jinja_, messages, options);
    }

    const bool continue_final_assistant =
        options.continuation == PromptContinuationMode::ContinueFinalAssistant;
    if (continue_final_assistant) {
        const ChatMessage& final = messages.back();
        if (final.role != ChatRole::Assistant || final.parts.empty() ||
            !final.reasoning_content.empty() || !final.tool_calls.empty()) {
            throw std::invalid_argument(
                "assistant continuation requires a final text-only assistant message");
        }
        if (final.has_media()) {
            throw std::invalid_argument(
                "assistant continuation requires a final text-only assistant message");
        }
        if (options.enable_thinking) {
            throw std::invalid_argument("assistant continuation cannot start in thinking mode");
        }
    }

    const bool effort_template = semantics_ == ChatTemplateSemantics::ReasoningEffort;
    const std::string_view reasoning_instructions =
        resolve_reasoning_instructions(semantics_, options);

    std::size_t message_begin = 0;
    RenderedFragment leading_instruction_raw;
    RenderedFragment leading_instruction;
    std::size_t leading_trim_begin = 0;
    std::size_t leading_trim_end   = 0;
    if (is_instruction_role(messages[0].role)) {
        validate_instruction_message(messages[0]);
        leading_instruction_raw = messages[0].rendered_content();
        std::tie(leading_trim_begin, leading_trim_end) =
            trim_ascii_whitespace_bounds(leading_instruction_raw.text);
        leading_instruction =
            slice_fragment(leading_instruction_raw, leading_trim_begin, leading_trim_end);
        message_begin = 1;
    }

    RenderBuilder rendered;
    std::vector<std::size_t> tool_boundaries;
    std::optional<std::size_t> instruction_begin;
    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        RenderedToolsSystemBlock preamble = render_tools_system_block(
            options.tool_jsons, leading_instruction, reasoning_instructions);
        rendered.append(std::move(preamble.fragment));
        tool_boundaries   = std::move(preamble.tool_boundaries);
        instruction_begin = preamble.instruction_begin;
    } else if (message_begin == 1) {
        if (!effort_template || !leading_instruction.text.empty() ||
            !reasoning_instructions.empty()) {
            rendered.append_template("<|im_start|>system\n");
            if (!reasoning_instructions.empty()) {
                rendered.append_template(reasoning_instructions);
                if (!leading_instruction.text.empty()) { rendered.append_template("\n\n"); }
            }
            if (!leading_instruction.text.empty()) { instruction_begin = rendered.size(); }
            rendered.append(leading_instruction);
            rendered.append_template("<|im_end|>\n");
        }
    } else if (!reasoning_instructions.empty()) {
        rendered.append_template("<|im_start|>system\n");
        rendered.append_template(reasoning_instructions);
        rendered.append_template("<|im_end|>\n");
    }

    std::vector<std::optional<std::size_t>> message_boundaries(messages.size() + 1U);
    std::vector<std::optional<std::size_t>> cache_boundaries(options.cache_markers.size());
    if (message_begin == 0) {
        message_boundaries[0] = rendered.size();
    } else {
        // The first instruction message is folded into the system preamble by this template.
        message_boundaries[1] = rendered.size();
    }

    const long last_query_index  = last_real_user_query(messages);
    const bool preserve_thinking = options.preserve_thinking.value_or(effort_template);
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
    std::vector<std::size_t> rewrite_execution_boundaries;
    const auto add_rewrite_execution_boundary = [&] {
        if (rewrite_execution_boundaries.empty() ||
            rewrite_execution_boundaries.back() != rendered.size()) {
            rewrite_execution_boundaries.push_back(rendered.size());
        }
    };

    int image_count         = 0;
    int video_count         = 0;
    std::size_t media_count = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (i < message_begin) { continue; }
        if (is_instruction_role(message.role)) { validate_instruction_message(message); }
        std::vector<std::size_t> raw_part_boundaries;
        const RenderedFragment raw_content = message.rendered_content(
            options.add_vision_id, &image_count, &video_count, &media_count, &raw_part_boundaries);
        const auto [content_trim_begin, content_trim_end] =
            trim_ascii_whitespace_bounds(raw_content.text);
        const RenderedFragment content =
            slice_fragment(raw_content, content_trim_begin, content_trim_end);
        const auto resolve_part_boundaries = [&](std::size_t content_begin) {
            for (std::size_t marker_index = 0; marker_index < options.cache_markers.size();
                 ++marker_index) {
                const PromptCacheMarker& marker = options.cache_markers[marker_index];
                if (marker.location != PromptCacheMarkerLocation::MessagePartBoundary ||
                    marker.after_message_count != i + 1U || marker.after_message_part_count == 0 ||
                    marker.after_message_part_count > raw_part_boundaries.size()) {
                    continue;
                }
                const std::size_t raw = raw_part_boundaries[marker.after_message_part_count - 1U];
                const std::size_t clamped = std::clamp(raw, content_trim_begin, content_trim_end);
                cache_boundaries[marker_index] = content_begin + clamped - content_trim_begin;
            }
        };
        if (is_instruction_role(message.role)) {
            rendered.append_template("<|im_start|>system\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::User) {
            rendered.append_template("<|im_start|>user\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("<|im_end|>\n");
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::Tool) {
            const bool opens_group = i == 0 || messages[i - 1].role != ChatRole::Tool;
            const bool closes_group =
                i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
            if (opens_group) { rendered.append_template("<|im_start|>user"); }
            rendered.append_template("\n<tool_response>\n");
            resolve_part_boundaries(rendered.size());
            rendered.append(content);
            rendered.append_template("\n</tool_response>");
            if (closes_group) { rendered.append_template("<|im_end|>\n"); }
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }

        if (message.role != ChatRole::Assistant) {
            throw std::invalid_argument("unsupported chat role value");
        }

        // assistant
        if (continue_final_assistant && i + 1U == messages.size()) {
            const std::size_t generation_begin = rendered.size();
            rewrite_checkpoint                 = RewriteCheckpointByteSpec{
                                .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
            rendered.append_template("<|im_start|>assistant\n");
            add_rewrite_execution_boundary();
            rendered.append(content);
            message_boundaries[i + 1U] = rendered.size();
            continue;
        }
        RenderedFragment reasoning;
        RenderedFragment body = content;
        if (!message.reasoning_content.empty()) {
            reasoning = literal_fragment(message.reasoning_content);
        } else if (!effort_template) {
            ThinkParts parts = derive_think_parts(content);
            reasoning        = std::move(parts.reasoning);
            body             = std::move(parts.content);
        }
        reasoning = trim_ascii_whitespace(reasoning);

        const bool keep_thinking = preserve_thinking || (static_cast<long>(i) > last_query_index);
        if (!preserve_thinking && !rewrite_checkpoint && static_cast<long>(i) > last_query_index) {
            // Closing the current turn may rewrite everything beginning with this assistant
            // segment. Keep the stable history before the opener recoverable; retaining the
            // deterministic opener itself is not worth losing the whole prefix when a caller
            // branches with a new user message instead.
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        rendered.append_template("<|im_start|>assistant\n");
        add_rewrite_execution_boundary();
        if (keep_thinking) {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
            rendered.append(reasoning);
            rendered.append_template("\n</think>\n\n");
            add_rewrite_execution_boundary();
        }
        rendered.append(body);
        if (!message.tool_calls.empty()) {
            const bool body_has_text = !trim_ascii_whitespace(body).text.empty();
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (body_has_text) { rendered.append_template("\n\n"); }
                } else {
                    rendered.append_template("\n");
                }
                rendered.append(render_tool_call(message.tool_calls[call_index], effort_template));
            }
        }
        rendered.append_template("<|im_end|>\n");
        message_boundaries[i + 1U] = rendered.size();
    }

    if (!continue_final_assistant && options.add_generation_prompt) {
        // The generation suffix is replaceable as a unit. An immediate successor may replay the
        // response, close the turn, or branch by appending a different user message directly to
        // the input history. The rolling private checkpoint must therefore precede the assistant
        // opener; placing it after the deterministic prologue makes the complete history
        // unrecoverable for the branch case merely to save a handful of prompt tokens.
        const std::size_t generation_begin = rendered.size();
        if (preserve_thinking) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::ResponseReplay, .offset = generation_begin};
        } else if (!rewrite_checkpoint) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = generation_begin};
        }
        rendered.append_template("<|im_start|>assistant\n");
        add_rewrite_execution_boundary();
        if (options.enable_thinking) {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
        } else {
            rendered.append_template("<think>\n");
            add_rewrite_execution_boundary();
            rendered.append_template("\n</think>\n\n");
            add_rewrite_execution_boundary();
        }
    }
    for (std::size_t index = 0; index < options.cache_markers.size(); ++index) {
        const PromptCacheMarker marker = options.cache_markers[index];
        switch (marker.location) {
        case PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count < message_boundaries.size()) {
                cache_boundaries[index] = message_boundaries[marker.after_message_count];
            }
            break;
        case PromptCacheMarkerLocation::MessagePartBoundary:
            // Resolved while the containing message's content offset is known. Assistant
            // reasoning/tool-call interiors deliberately remain advisory and unresolved.
            break;
        case PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (message_begin == 1 && instruction_begin &&
                marker.leading_instruction_bytes <= leading_instruction_raw.text.size()) {
                const std::size_t clamped = std::clamp<std::size_t>(
                    marker.leading_instruction_bytes, leading_trim_begin, leading_trim_end);
                cache_boundaries[index] = *instruction_begin + clamped - leading_trim_begin;
            }
            break;
        case PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count != 0 && marker.after_tool_count <= tool_boundaries.size()) {
                cache_boundaries[index] = tool_boundaries[marker.after_tool_count - 1U];
            }
            break;
        }
    }
    RenderedFragment final = std::move(rendered).release();
    return RenderedChat{.text                         = std::move(final.text),
                        .literal_spans                = std::move(final.literal_spans),
                        .media_placeholders           = std::move(final.media_placeholders),
                        .rewrite_checkpoint           = rewrite_checkpoint,
                        .rewrite_execution_boundaries = std::move(rewrite_execution_boundaries),
                        .message_boundaries           = std::move(message_boundaries),
                        .cache_boundaries             = std::move(cache_boundaries)};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
