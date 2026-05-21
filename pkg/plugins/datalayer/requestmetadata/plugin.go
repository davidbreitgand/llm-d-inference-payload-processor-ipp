/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package requestmetadata

import (
	"context"
	"encoding/json"
	"math"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/datastore"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
	dlsrc "github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer/datasource"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/plugin"
)

const (
	// PluginType is the identifier used when registering this extractor.
	PluginType = "request-metadata-extractor"

	// RequestMetadataAttributeKey is the attribute key written to each model's attribute store.
	RequestMetadataAttributeKey = "request-metadata"
)

// compile-time interface assertion
var _ dlsrc.Extractor = &RequestMetadataExtractor{}

// ExtractorFactory creates a RequestMetadataExtractor with a nil DataStore.
// The factory path is limited: the DataStore is not available via plugin.Handle,
// so the created extractor cannot write to the store. Use NewRequestMetadataExtractor
// directly when constructing for production use.
func ExtractorFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return NewRequestMetadataExtractor(nil).WithName(name), nil
}

// RequestMetadataCount holds in-flight request counts, in-flight max_tokens proxy totals,
// cumulative actual token usage for one model, and rollover counts for the cumulative token fields.
type RequestMetadataCount struct {
	Requests             int64
	MaxTokens            int64
	InputTokens          int64
	InputTokenRollovers  int64
	OutputTokens         int64
	OutputTokenRollovers int64
	CachedTokens         int64
	CachedTokenRollovers int64
	ThinkTokens          int64
	ThinkTokenRollovers  int64
}

func (r RequestMetadataCount) Clone() datalayer.Cloneable { return r }

// standardizedUsage holds provider-normalized token counters for one response.
type standardizedUsage struct {
	InputTokens  int64
	OutputTokens int64
	CachedTokens int64
	ThinkTokens  int64
}

// RequestMetadataExtractor tracks in-flight request counts and cumulative actual token usage per model.
// It writes RequestMetadataCount to each model's RequestMetadataAttributeKey attribute.
//
// Extract is assumed to be called from a single goroutine (the NotificationSource event loop).
// If parallel dispatch is introduced, add a sync.Mutex around counters and the DataStore write.
//
// TODO: counters leak if a request fails without a corresponding ResponseEventType (e.g. connection
// drop, upstream error, context cancellation). The call site should fire a
// synthetic ResponseEventType in its error/EOF path to keep counts accurate.
type RequestMetadataExtractor struct {
	typedName plugin.TypedName
	ds        datastore.Datastore
	counters  map[string]RequestMetadataCount
}

func NewRequestMetadataExtractor(ds datastore.Datastore) *RequestMetadataExtractor {
	return &RequestMetadataExtractor{
		typedName: plugin.TypedName{Type: PluginType, Name: PluginType},
		ds:        ds,
		counters:  make(map[string]RequestMetadataCount),
	}
}

func (e *RequestMetadataExtractor) TypedName() plugin.TypedName { return e.typedName }

// WithName sets the instance name, used by the factory when the plugin is configured by name.
func (e *RequestMetadataExtractor) WithName(name string) *RequestMetadataExtractor {
	e.typedName.Name = name
	return e
}

func (e *RequestMetadataExtractor) Extract(_ context.Context, events []dlsrc.Event) error {
	updated := map[string]RequestMetadataCount{}

	for _, ev := range events {
		switch ev.Type {
		case dlsrc.RequestEventType:
			p, ok := ev.Payload.(dlsrc.RequestPayload)
			if !ok {
				continue
			}
			model, _ := p.Request.Body["model"].(string)
			if model == "" {
				continue
			}
			ensureStreamingUsageIncluded(p.Request.Body)
			maxTokens := int64Value(p.Request.Body["max_tokens"])
			c := e.counters[model]
			c.Requests++
			c.MaxTokens += maxTokens
			e.counters[model] = c
			updated[model] = c

		case dlsrc.ResponseEventType:
			p, ok := ev.Payload.(dlsrc.ResponsePayload)
			if !ok {
				continue
			}
			model, _ := p.Request.Body["model"].(string)
			if model == "" {
				continue
			}
			c := e.counters[model]
			floorDecrement(&c.Requests, 1)

			maxTokens := int64Value(p.Request.Body["max_tokens"])
			floorDecrement(&c.MaxTokens, maxTokens)

			usage, ok := extractStandardizedUsage(p.Request.Body, p.Response.Body)
			if ok {
				addWithRollover(&c.InputTokens, &c.InputTokenRollovers, usage.InputTokens)
				addWithRollover(&c.OutputTokens, &c.OutputTokenRollovers, usage.OutputTokens)
				addWithRollover(&c.CachedTokens, &c.CachedTokenRollovers, usage.CachedTokens)
				addWithRollover(&c.ThinkTokens, &c.ThinkTokenRollovers, usage.ThinkTokens)
			}

			e.counters[model] = c
			updated[model] = c
		}
	}

	if e.ds != nil {
		for model, c := range updated {
			e.ds.GetOrCreateModel(model).GetAttributes().Put(RequestMetadataAttributeKey, c)
		}
	}
	return nil
}

// extractStandardizedUsage resolves the provider and extracts provider-normalized response usage.
// Phase 1 supports conservative detection of OpenAI-compatible response bodies only, including
// streamed final response chunks that carry a top-level usage object.
// TODO: extend to other providers
func extractStandardizedUsage(reqBody map[string]any, respBody map[string]any) (standardizedUsage, bool) {
	if !isOpenAIRequest(reqBody, respBody) {
		return standardizedUsage{}, false
	}
	return extractOpenAIUsage(respBody)
}

func isOpenAIRequest(reqBody map[string]any, respBody map[string]any) bool {
	model, _ := reqBody["model"].(string)
	if model == "" {
		return false
	}

	usage, ok := respBody["usage"].(map[string]any)
	if !ok {
		return false
	}

	_, hasPromptTokens := usage["prompt_tokens"]
	_, hasCompletionTokens := usage["completion_tokens"]
	if !hasPromptTokens || !hasCompletionTokens {
		return false
	}

	return true
}

func extractOpenAIUsage(body map[string]any) (standardizedUsage, bool) {
	usage, ok := body["usage"].(map[string]any)
	if !ok {
		return standardizedUsage{}, false
	}

	result := standardizedUsage{
		InputTokens:  int64Value(usage["prompt_tokens"]),
		OutputTokens: int64Value(usage["completion_tokens"]),
	}

	if promptDetails, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		result.CachedTokens = int64Value(promptDetails["cached_tokens"])
	}
	if completionDetails, ok := usage["completion_tokens_details"].(map[string]any); ok {
		result.ThinkTokens = int64Value(completionDetails["reasoning_tokens"])
	}

	return result, true
}

func ensureStreamingUsageIncluded(body map[string]any) {
	stream, _ := body["stream"].(bool)
	if !stream {
		return
	}

	streamOptions, ok := body["stream_options"].(map[string]any)
	if !ok {
		streamOptions = map[string]any{}
		body["stream_options"] = streamOptions
	}
	streamOptions["include_usage"] = true
}

func int64Value(v any) int64 {
	switch n := v.(type) {
	case int:
		return int64(n)
	case int64:
		return n
	case float64:
		return int64(n)
	default:
		return 0
	}
}

// floorDecrement decrements v by delta, flooring at zero.
func floorDecrement(v *int64, delta int64) {
	*v -= delta
	if *v < 0 {
		*v = 0
	}
}

// addWithRollover adds delta to v, incrementing rollovers if necessary.
// addWithRollover behavior is modulo-int64 accumulation with explicit overflow counting.
func addWithRollover(v *int64, rollovers *int64, delta int64) {
	if delta <= 0 {
		return
	}
	if *v > math.MaxInt64-delta {
		*rollovers++
		*v = delta - (math.MaxInt64 - *v) - 1
		return
	}
	*v += delta
}
