package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"kiro-go/config"
	"net/http"
	"strings"
)

const (
	anthropicAPIBase    = "https://api.anthropic.com"
	anthropicVersion    = "2023-06-01"
	anthropicBetaHeader = "interleaved-thinking-2025-05-14"
)

// anthropicBaseURL returns the effective upstream base URL for an Anthropic
// account. When account.BaseURL is set it is used as-is (trailing slash
// stripped), enabling third-party Anthropic-compatible relay services.
func anthropicBaseURL(account *config.Account) string {
	if account != nil && strings.TrimSpace(account.BaseURL) != "" {
		return strings.TrimRight(strings.TrimSpace(account.BaseURL), "/")
	}
	return anthropicAPIBase
}

// callAnthropicAPI sends a serialized ClaudeRequest to api.anthropic.com and
// drives the KiroStreamCallback exactly as CallKiroAPI does for Kiro accounts.
// Real Anthropic cache_read/cache_creation token counts are intentionally
// ignored — the caller's promptCacheTracker provides simulated values instead,
// ensuring consistent cache-reporting behavior across all account types.
func callAnthropicAPI(account *config.Account, reqBody []byte, callback *KiroStreamCallback) error {
	if callback == nil {
		callback = &KiroStreamCallback{}
	}

	// Determine whether streaming is requested from the serialized body.
	var meta struct {
		Stream   bool        `json:"stream"`
		Thinking interface{} `json:"thinking"`
	}
	_ = json.Unmarshal(reqBody, &meta)

	url := anthropicBaseURL(account) + "/v1/messages"
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", account.AccessToken)
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	// Enable extended thinking beta if the request carries a thinking config.
	if meta.Thinking != nil {
		httpReq.Header.Set("anthropic-beta", anthropicBetaHeader)
	}
	if meta.Stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}

	proxyURL := ResolveAccountProxyURL(account)
	client := GetClientForProxy(proxyURL)
	resp, err := client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	if meta.Stream {
		return parseAnthropicSSE(resp.Body, callback)
	}
	return parseAnthropicNonStream(resp.Body, callback)
}

// anthropicBlockState tracks per-content-block state during SSE parsing.
type anthropicBlockState struct {
	blockType    string // "text", "thinking", "tool_use"
	toolUseID    string
	toolName     string
	inputBuilder strings.Builder
}

// parseAnthropicSSE reads the Anthropic streaming response and drives callbacks.
// Cache token fields (cache_read_input_tokens, cache_creation_input_tokens) are
// deliberately discarded — the caller uses simulated values from promptCacheTracker.
func parseAnthropicSSE(body io.Reader, callback *KiroStreamCallback) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB — large tool schemas can exceed 256KB

	var (
		eventType    string
		inputTokens  int
		outputTokens int
		blocks       = make(map[int]*anthropicBlockState)
	)

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event:") {
			eventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}

		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var ev map[string]interface{}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}

		switch eventType {
		case "message_start":
			msg, _ := ev["message"].(map[string]interface{})
			if msg != nil {
				usage, _ := msg["usage"].(map[string]interface{})
				if usage != nil {
					// Read real input_tokens as the total (cache-inclusive).
					// cache_read/creation are ignored — simulated values are used instead.
					if v, ok := usage["input_tokens"].(float64); ok {
						inputTokens = int(v)
					}
				}
			}

		case "content_block_start":
			idx := blockIndexFromEvent(ev)
			cb, _ := ev["content_block"].(map[string]interface{})
			if cb == nil {
				break
			}
			btype, _ := cb["type"].(string)
			state := &anthropicBlockState{blockType: btype}
			if btype == "tool_use" {
				state.toolUseID, _ = cb["id"].(string)
				state.toolName, _ = cb["name"].(string)
			}
			blocks[idx] = state

		case "content_block_delta":
			idx := blockIndexFromEvent(ev)
			state := blocks[idx]
			delta, _ := ev["delta"].(map[string]interface{})
			if delta == nil || state == nil {
				break
			}
			dtype, _ := delta["type"].(string)
			switch dtype {
			case "text_delta":
				text, _ := delta["text"].(string)
				if text != "" && callback.OnText != nil {
					callback.OnText(text, false)
				}
			case "thinking_delta":
				thinking, _ := delta["thinking"].(string)
				if thinking != "" && callback.OnText != nil {
					callback.OnText(thinking, true)
				}
			case "input_json_delta":
				partial, _ := delta["partial_json"].(string)
				state.inputBuilder.WriteString(partial)
			}

		case "content_block_stop":
			idx := blockIndexFromEvent(ev)
			state := blocks[idx]
			if state != nil && state.blockType == "tool_use" {
				var input map[string]interface{}
				_ = json.Unmarshal([]byte(state.inputBuilder.String()), &input)
				if input == nil {
					input = make(map[string]interface{})
				}
				if callback.OnToolUse != nil {
					callback.OnToolUse(KiroToolUse{
						ToolUseID: state.toolUseID,
						Name:      state.toolName,
						Input:     input,
					})
				}
			}
			delete(blocks, idx)

		case "message_delta":
			usage, _ := ev["usage"].(map[string]interface{})
			if usage != nil {
				if v, ok := usage["output_tokens"].(float64); ok {
					outputTokens = int(v)
				}
			}

		case "message_stop":
			if callback.OnComplete != nil {
				callback.OnComplete(inputTokens, outputTokens)
			}

		case "error":
			// Anthropic sends this for mid-stream errors (e.g. overload, auth failure).
			// Return an error so the caller retries or surfaces it to the client.
			if errObj, ok := ev["error"].(map[string]interface{}); ok {
				msg, _ := errObj["message"].(string)
				errType, _ := errObj["type"].(string)
				return fmt.Errorf("anthropic stream error: %s: %s", errType, msg)
			}
			return fmt.Errorf("anthropic stream error: %v", ev)
		}
	}

	return scanner.Err()
}

// parseAnthropicNonStream handles a non-streaming Anthropic response.
func parseAnthropicNonStream(body io.Reader, callback *KiroStreamCallback) error {
	var resp struct {
		Content []struct {
			Type    string `json:"type"`
			Text    string `json:"text,omitempty"`
			Thinking string `json:"thinking,omitempty"`
			ID      string `json:"id,omitempty"`
			Name    string `json:"name,omitempty"`
			Input   map[string]interface{} `json:"input,omitempty"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			// cache_read_input_tokens and cache_creation_input_tokens are
			// intentionally ignored — simulated values are used instead.
		} `json:"usage"`
	}
	data, err := io.ReadAll(body)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &resp); err != nil {
		return fmt.Errorf("decode anthropic response: %w", err)
	}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != "" && callback.OnText != nil {
				callback.OnText(block.Text, false)
			}
		case "thinking":
			if block.Thinking != "" && callback.OnText != nil {
				callback.OnText(block.Thinking, true)
			}
		case "tool_use":
			if callback.OnToolUse != nil {
				input := block.Input
				if input == nil {
					input = make(map[string]interface{})
				}
				callback.OnToolUse(KiroToolUse{
					ToolUseID: block.ID,
					Name:      block.Name,
					Input:     input,
				})
			}
		}
	}

	if callback.OnComplete != nil {
		callback.OnComplete(resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}
	return nil
}

func blockIndexFromEvent(ev map[string]interface{}) int {
	if v, ok := ev["index"].(float64); ok {
		return int(v)
	}
	return 0
}

// openAIToClaudeRequest converts an OpenAI Chat Completions request to the
// Anthropic Claude Messages format so OpenAI-format clients can be routed to
// Anthropic API accounts transparently.
func openAIToClaudeRequest(req *OpenAIRequest, thinking bool, thinkingSuffix string) *ClaudeRequest {
	model := req.Model
	// Strip thinking suffix if present so we send the clean model name.
	if thinkingSuffix != "" && strings.HasSuffix(model, thinkingSuffix) {
		model = strings.TrimSuffix(model, thinkingSuffix)
	}

	claude := &ClaudeRequest{
		Model:       model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}
	if claude.MaxTokens <= 0 {
		claude.MaxTokens = 8096
	}

	// Extract system message and convert the rest.
	var messages []ClaudeMessage
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			if text, ok := msg.Content.(string); ok {
				claude.System = text
			}
			continue
		}
		messages = append(messages, openAIMessageToClaude(msg))
	}
	claude.Messages = messages

	// Convert tools.
	for _, t := range req.Tools {
		claude.Tools = append(claude.Tools, ClaudeTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	// Inject thinking config when the model suffix indicated it.
	if thinking {
		budget := 8000
		if claude.MaxTokens > 0 && budget >= claude.MaxTokens {
			budget = claude.MaxTokens / 2
		}
		if budget < 1024 {
			budget = 1024
		}
		// Anthropic requires budget_tokens strictly less than max_tokens.
		if claude.MaxTokens > 0 && budget >= claude.MaxTokens {
			budget = claude.MaxTokens - 1
		}
		// If max_tokens is too small to fit a minimum thinking budget, skip thinking.
		if budget < 1 {
			budget = 1
		}
		if claude.MaxTokens <= 1 {
			// Cannot do thinking with max_tokens this small; skip silently.
		} else {
			claude.Thinking = &ClaudeThinkingConfig{
				Type:         "enabled",
				BudgetTokens: budget,
			}
		}
	}

	return claude
}

func openAIMessageToClaude(msg OpenAIMessage) ClaudeMessage {
	role := msg.Role
	if role == "tool" {
		role = "user"
	}

	// Tool result message (role=tool in OpenAI, tool_result content block in Claude).
	if msg.Role == "tool" {
		text := ""
		if s, ok := msg.Content.(string); ok {
			text = s
		}
		return ClaudeMessage{
			Role: "user",
			Content: []ClaudeContentBlock{
				{
					Type:      "tool_result",
					ToolUseID: msg.ToolCallID,
					Content:   text,
				},
			},
		}
	}

	// Assistant message with tool calls → tool_use blocks.
	if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
		var blocks []ClaudeContentBlock
		if text, ok := msg.Content.(string); ok && text != "" {
			blocks = append(blocks, ClaudeContentBlock{Type: "text", Text: text})
		}
		for _, tc := range msg.ToolCalls {
			var input map[string]interface{}
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
			if input == nil {
				input = make(map[string]interface{})
			}
			blocks = append(blocks, ClaudeContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
		return ClaudeMessage{Role: "assistant", Content: blocks}
	}

	// Plain string content.
	if text, ok := msg.Content.(string); ok {
		return ClaudeMessage{Role: role, Content: text}
	}

	// Array content (multimodal).
	if arr, ok := msg.Content.([]interface{}); ok {
		var blocks []ClaudeContentBlock
		for _, item := range arr {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			itemType, _ := m["type"].(string)
			switch itemType {
			case "text":
				text, _ := m["text"].(string)
				blocks = append(blocks, ClaudeContentBlock{Type: "text", Text: text})
			case "image_url":
				imgURL, _ := m["image_url"].(map[string]interface{})
				if imgURL != nil {
					if urlStr, ok := imgURL["url"].(string); ok {
						if strings.HasPrefix(urlStr, "data:") {
							// data URI → base64 source
							parts := strings.SplitN(urlStr, ",", 2)
							if len(parts) == 2 {
								mediaType := strings.TrimSuffix(strings.TrimPrefix(parts[0], "data:"), ";base64")
								blocks = append(blocks, ClaudeContentBlock{
									Type: "image",
									Source: &ImageSource{
										Type:      "base64",
										MediaType: mediaType,
										Data:      parts[1],
									},
								})
							}
						}
					}
				}
			}
		}
		return ClaudeMessage{Role: role, Content: blocks}
	}

	return ClaudeMessage{Role: role, Content: ""}
}
