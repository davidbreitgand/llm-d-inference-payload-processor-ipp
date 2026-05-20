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
	"testing"
	"time"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/datastore"
	dlsrc "github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer/datasource"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/requesthandling"
)

// makeRequestEvent creates a RequestEventType event with model, max_tokens, and optional extra request body fields.
func makeRequestEvent(model string, maxTokens float64, extras map[string]any) dlsrc.Event {
	req := requesthandling.NewInferenceRequest()
	req.Body["model"] = model
	req.Body["max_tokens"] = maxTokens
	for k, v := range extras {
		req.Body[k] = v
	}
	return dlsrc.Event{
		Type:    dlsrc.RequestEventType,
		Payload: dlsrc.RequestPayload{Request: req},
	}
}

// makeResponseEvent creates a ResponseEventType event with model, duration, request max_tokens, request extras, and response body.
func makeResponseEvent(model string, durationMs int, maxTokens float64, requestExtras map[string]any, responseBody map[string]any) dlsrc.Event {
	req := requesthandling.NewInferenceRequest()
	req.Body["model"] = model
	req.Body["max_tokens"] = maxTokens
	for k, v := range requestExtras {
		req.Body[k] = v
	}

	resp := requesthandling.NewInferenceResponse()
	for k, v := range responseBody {
		resp.Body[k] = v
	}

	return dlsrc.Event{
		Type: dlsrc.ResponseEventType,
		Payload: dlsrc.ResponsePayload{
			Request:  req,
			Response: resp,
			Duration: time.Duration(durationMs) * time.Millisecond,
		},
	}
}

func openAIUsageResponseBody(promptTokens, completionTokens, totalTokens, cachedTokens, reasoningTokens float64) map[string]any {
	return map[string]any{
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      totalTokens,
			"prompt_tokens_details": map[string]any{
				"cached_tokens": cachedTokens,
			},
			"completion_tokens_details": map[string]any{
				"reasoning_tokens": reasoningTokens,
			},
		},
	}
}

// getInflightRequests asserts the inflight-requests attribute exists for model and returns it.
func streamedOpenAIFinalChunkResponseBody(promptTokens, completionTokens, cachedTokens, reasoningTokens float64) map[string]any {
	body := openAIUsageResponseBody(promptTokens, completionTokens, promptTokens+completionTokens, cachedTokens, reasoningTokens)
	body["object"] = "chat.completion.chunk"
	body["choices"] = []any{
		map[string]any{
			"index": 0,
			"delta": map[string]any{},
		},
	}
	return body
}

func getRequestMetadata(t testing.TB, ds datastore.Datastore, model string) RequestMetadataCount {
	t.Helper()
	val, ok := ds.GetOrCreateModel(model).GetAttributes().Get(RequestMetadataAttributeKey)
	if !ok {
		t.Fatalf("expected %q attribute for model %q", RequestMetadataAttributeKey, model)
	}
	rc, ok := val.(RequestMetadataCount)
	if !ok {
		t.Fatalf("expected RequestMetadataCount for model %q", model)
	}
	return rc
}

func newRequestMetadataTest(t *testing.T) (*RequestMetadataExtractor, datastore.Datastore) {
	t.Helper()
	ds := datastore.NewFakeDataStore()
	return NewRequestMetadataExtractor(ds), ds
}

func TestRequestIncrementsInflightCounterOnly(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{makeRequestEvent("m1", 100, nil)}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 1 {
		t.Errorf("expected Requests=1, got %d", rc.Requests)
	}
	if rc.MaxTokens != 100 {
		t.Errorf("expected MaxTokens=100, got %d", rc.MaxTokens)
	}
	if rc.InputTokens != 0 || rc.OutputTokens != 0 || rc.CachedTokens != 0 || rc.ThinkTokens != 0 {
		t.Errorf("expected zero cumulative token counts on request, got %+v", rc)
	}
}

func TestStreamingRequestMutatesIncludeUsage(t *testing.T) {
	ext, _ := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeRequestEvent("m1", 100, map[string]any{
			"stream": true,
			"stream_options": map[string]any{
				"existing": "value",
			},
		}),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	payload := batch[0].Payload.(dlsrc.RequestPayload)
	streamOptions, ok := payload.Request.Body["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("expected stream_options map to be created")
	}
	if includeUsage, ok := streamOptions["include_usage"].(bool); !ok || !includeUsage {
		t.Fatalf("expected stream_options.include_usage=true, got %#v", streamOptions["include_usage"])
	}
	if got := streamOptions["existing"]; got != "value" {
		t.Fatalf("expected existing stream_options keys preserved, got %#v", got)
	}
}

func TestStreamingFinalChunkAccumulatesUsage(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeRequestEvent("m1", 300, map[string]any{"stream": true}),
		makeResponseEvent("m1", 50, 300, map[string]any{"stream": true}, streamedOpenAIFinalChunkResponseBody(100, 200, 25, 75)),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 0 {
		t.Errorf("expected Requests=0, got %d", rc.Requests)
	}
	if rc.MaxTokens != 0 {
		t.Errorf("expected MaxTokens=0 after matching response, got %d", rc.MaxTokens)
	}
	if rc.InputTokens != 100 || rc.OutputTokens != 200 || rc.CachedTokens != 25 || rc.ThinkTokens != 75 {
		t.Errorf("unexpected cumulative usage from streamed final chunk: %+v", rc)
	}
}

func TestResponseDecrementsInflightAndAccumulatesUsage(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeRequestEvent("m1", 300, nil),
		makeResponseEvent("m1", 50, 300, nil, openAIUsageResponseBody(100, 200, 123, 25, 75)),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 0 {
		t.Errorf("expected Requests=0, got %d", rc.Requests)
	}
	if rc.MaxTokens != 0 {
		t.Errorf("expected MaxTokens=0 after matching response, got %d", rc.MaxTokens)
	}
	if rc.InputTokens != 100 || rc.OutputTokens != 200 || rc.CachedTokens != 25 || rc.ThinkTokens != 75 {
		t.Errorf("unexpected cumulative usage: %+v", rc)
	}
}

func TestCounterFloorsAtZeroAndAccumulatesUsage(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeResponseEvent("m1", 50, 300, nil, openAIUsageResponseBody(10, 20, 30, 3, 4)),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 0 {
		t.Errorf("expected Requests=0, got %d", rc.Requests)
	}
	if rc.MaxTokens != 0 {
		t.Errorf("expected MaxTokens floored to zero, got %d", rc.MaxTokens)
	}
	if rc.InputTokens != 10 || rc.OutputTokens != 20 || rc.CachedTokens != 3 || rc.ThinkTokens != 4 {
		t.Errorf("unexpected cumulative usage: %+v", rc)
	}
}

func TestResponseMissingUsageOnlyDecrementsInflight(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeRequestEvent("m1", 125, nil),
		makeResponseEvent("m1", 50, 125, nil, map[string]any{}),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 0 {
		t.Errorf("expected Requests=0, got %d", rc.Requests)
	}
	if rc.MaxTokens != 0 {
		t.Errorf("expected MaxTokens=0 after fallback decrement, got %d", rc.MaxTokens)
	}
	if rc.InputTokens != 0 || rc.OutputTokens != 0 || rc.CachedTokens != 0 || rc.ThinkTokens != 0 {
		t.Errorf("expected zero cumulative usage without usage payload, got %+v", rc)
	}
}

func TestResponseUsageWithoutOpenAIShapeDoesNotAccumulateUsage(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeRequestEvent("m1", 125, nil),
		makeResponseEvent("m1", 50, 125, nil, map[string]any{
			"usage": map[string]any{
				"input_tokens":  float64(11),
				"output_tokens": float64(22),
			},
		}),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.Requests != 0 {
		t.Errorf("expected Requests=0, got %d", rc.Requests)
	}
	if rc.MaxTokens != 0 {
		t.Errorf("expected MaxTokens=0 after fallback decrement, got %d", rc.MaxTokens)
	}
	if rc.InputTokens != 0 || rc.OutputTokens != 0 || rc.CachedTokens != 0 || rc.ThinkTokens != 0 {
		t.Errorf("expected zero cumulative usage for non-OpenAI-shaped usage payload, got %+v", rc)
	}
}

func TestResponseMissingNestedUsageDetailsDefaultsToZero(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeResponseEvent("m1", 50, 40, nil, map[string]any{
			"usage": map[string]any{
				"prompt_tokens":     float64(11),
				"completion_tokens": float64(22),
			},
		}),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := getRequestMetadata(t, ds, "m1")
	if rc.MaxTokens != 0 {
		t.Errorf("expected MaxTokens=0 after request max_tokens decrement, got %d", rc.MaxTokens)
	}
	if rc.InputTokens != 11 || rc.OutputTokens != 22 || rc.CachedTokens != 0 || rc.ThinkTokens != 0 {
		t.Errorf("unexpected cumulative usage: %+v", rc)
	}
}

func TestRequestMetadataMultipleModels(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{
		makeRequestEvent("m1", 10, nil),
		makeRequestEvent("m2", 20, nil),
		makeResponseEvent("m1", 10, 10, nil, openAIUsageResponseBody(1, 2, 3, 0, 0)),
		makeResponseEvent("m2", 20, 20, nil, openAIUsageResponseBody(3, 4, 7, 1, 2)),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc1 := getRequestMetadata(t, ds, "m1")
	if rc1.Requests != 0 || rc1.MaxTokens != 0 || rc1.InputTokens != 1 || rc1.OutputTokens != 2 || rc1.CachedTokens != 0 || rc1.ThinkTokens != 0 {
		t.Errorf("m1: unexpected metadata %+v", rc1)
	}

	rc2 := getRequestMetadata(t, ds, "m2")
	if rc2.Requests != 0 || rc2.MaxTokens != 0 || rc2.InputTokens != 3 || rc2.OutputTokens != 4 || rc2.CachedTokens != 1 || rc2.ThinkTokens != 2 {
		t.Errorf("m2: unexpected metadata %+v", rc2)
	}
}

func TestRequestMetadataUnknownEventTypeIgnored(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	batch := []dlsrc.Event{{Type: "unknown"}}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	modelCount := len(ds.Models())
	if modelCount != 0 {
		t.Errorf("expected no models in datastore, got %d", modelCount)
	}
}

func TestRequestMetadataMissingModelFieldIgnored(t *testing.T) {
	ext, ds := newRequestMetadataTest(t)

	// Payload without a "model" key — no counter should be updated.
	req := requesthandling.NewInferenceRequest()
	batch := []dlsrc.Event{
		{Type: dlsrc.RequestEventType, Payload: dlsrc.RequestPayload{Request: req}},
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	modelCount := len(ds.Models())
	if modelCount != 0 {
		t.Errorf("expected no models in datastore, got %d", modelCount)
	}
}

func TestExtractWithNilDatastoreDoesNotPanic(t *testing.T) {
	ext := NewRequestMetadataExtractor(nil)

	batch := []dlsrc.Event{
		makeRequestEvent("m1", 100, nil),
		makeResponseEvent("m1", 50, 100, nil, openAIUsageResponseBody(10, 20, 30, 3, 4)),
	}
	if err := ext.Extract(context.Background(), batch); err != nil {
		t.Fatalf("Extract failed: %v", err)
	}

	rc := ext.counters["m1"]
	if rc.Requests != 0 || rc.MaxTokens != 0 || rc.InputTokens != 10 || rc.OutputTokens != 20 || rc.CachedTokens != 3 || rc.ThinkTokens != 4 {
		t.Fatalf("unexpected in-memory counters with nil datastore: %+v", rc)
	}
}
