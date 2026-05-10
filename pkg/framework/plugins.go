/*
Copyright 2026 The Kubernetes Authors.

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

package framework

import (
	"context"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/datalayer"
)

// ScorerCategory indicates the preference a scorer applies when scoring candidate models
type ScorerCategory string

const (
	// Cheapest indicates a scorer that prefers models with the lowest USD/(1M tokens) price
	Cheapest ScorerCategory = "Cheapest"

	// TODO: Add other cost-related scorer categories here after the simplest "Cheapest" category scorer is implemented
	// Value indicates a scorer that prefers models sitting on the efficiency curve
	// A model is better than another model only if it is both more accurate and cheaper
	// Value = (normalized accuracy rank)/(actual total cost)
	// Value ScorerCategory = "Value"

	// PredictiveCost indicates a scorer that prefers models for which predicted cost is lowest
	// PredictiveCost ScorerCategory = "PredictiveCost"

	// PredictiveValue indicates a scorer that prefers models for which predicted value is highest
	// PredictiveValue ScorerCategory = "PredictiveValue"

	// TODO: Add other scorer categories here for other types of scoring
)

// Plugin defines the interface for a plugin.
// This interface should be embedded in all plugins across the code.
type Plugin interface {
	// TypedName returns the type and name tuple of this plugin instance.
	TypedName() TypedName
}

type RequestProcessor interface {
	Plugin
	// ProcessRequest runs the RequestProcessor plugin.
	// RequestProcessor can mutate the headers and/or the body of the request.
	ProcessRequest(ctx context.Context, cycleState *CycleState, request *InferenceRequest) error
}

type ResponseProcessor interface {
	Plugin
	// ProcessResponse runs the ResponseProcessor plugin.
	// ResponseProcessor can mutate the headers and/or the body of the response.
	ProcessResponse(ctx context.Context, cycleState *CycleState, response *InferenceResponse) error
}

// Scorer defines the interface for a scorer plugin that scores a list of models based on the category and context.
// Scorer implementation MUST return a score in [0,1] with 1 being the highest score
type Scorer interface {
	Plugin
	// Category returns the category of the scorer
	Category() ScorerCategory
	// Score returns a score for each input model
	Score(ctx context.Context, cycleState *CycleState, models []*datalayer.Model) map[string]float64
}
