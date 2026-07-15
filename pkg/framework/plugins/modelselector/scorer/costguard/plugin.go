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

// Package costguard implements a cost-minimising scorer for the ModelSelector
// framework. It routes inference to the model with the lowest observed cost
// via an epsilon-Greedy Multi-Arm Bandit, using a risk-aware rank that
// combines each model's body cost (trimmed mean) and Conditional Tail
// Expectation of the cost distribution. See the package README.md file for the full
// algorithm and configuration reference.
package costguard

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/modelselector"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/requesthandling"
)

const (
	// PluginType is the identifier used when registering this scorer.
	PluginType = "costguard"

	// z95 is the standard normal quantile for a 95% confidence interval, used
	// in sampleThreshold to derive the per-model minimum sample size from the
	// configured percentile margin of error.
	z95 = 1.96

	// neutralScore is returned for any model that the scorer has no strong opinion
	// on (missing / empty / under-explored cost data, degenerate sigmoid
	// inputs). It composes with other scorers in the pipeline without pushing
	// selection either way.
	neutralScore = 0.5

	defaultEpsilon               = 0.1
	defaultAlpha                 = 0.95
	defaultLambda                = 1.0
	defaultWindowDuration        = "2h"
	defaultPercentileMarginError = 0.03
)

// compile-time interface assertion
var _ modelselector.Scorer = &CostGuardScorer{}

// Config defines the JSON configuration for the plugin. All fields have
// sensible defaults, so an empty config is valid.
type Config struct {
	// Epsilon is the probability of random exploration on each request. Must
	// be in [0, 1]. Defaults to 0.1. Setting Epsilon to 1.0 forces full random
	// selection (with replacement) on every request.
	Epsilon float64 `json:"epsilon"`

	// Alpha is the quantile that separates the body of the cost distribution
	// from the tail. Must be in (0, 1). Defaults to 0.95.
	Alpha float64 `json:"alpha"`

	// Lambda is the penalty weight applied to the tail cost contribution
	// (Conditional Tail Expectation). Must be >= 0. Defaults to 1.0.
	Lambda float64 `json:"lambda"`

	// WindowDuration is the epoch length. Parsed as a Go duration string
	// (e.g. "2h", "30m"). Must be > 0. Defaults to "2h".
	//
	// TODO: costguard-epoch: consume in the final PR for costguard epoch.
	// Parsed and validated today, but not consulted at scoring time.
	// The requestcostmetadata extractor plugin will own the window
	// in the concluding PR for CostGuard.
	WindowDuration string `json:"windowDuration"`

	// PercentileMarginError is the error around the alpha-quantile
	// of the observed cost distribution. Must be in (0, 1).
	// Defaults to 0.03. Smaller values require quadratically more
	// samples to consider a model sufficiently explored.
	PercentileMarginError float64 `json:"percentileMarginError"`
}

// CostGuardScorer will consume per-model cost samples via the datalayer
// (published by the requestcostmetadata extractor) and score candidate models
// to minimize typical cost and risk of the high cost in the tail.
// See README.md for a description of how it works.
//
// At this time only the configuration surface, plugin registration, and a stub
// scorer that returns a neutral score for every model exists.
type CostGuardScorer struct {
	typedName       plugin.TypedName
	epsilon         float64
	alpha           float64
	lambda          float64
	sampleThreshold uint64
	windowDuration  time.Duration
}

// ScorerFactory creates a CostGuardScorer from the given raw parameters.
// It validates every field of Config and precomputes the sample threshold so
// Score can rank candidate models with high statistical fidelity.
func ScorerFactory(name string, rawParameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	config := Config{
		Epsilon:               defaultEpsilon,
		Alpha:                 defaultAlpha,
		Lambda:                defaultLambda,
		WindowDuration:        defaultWindowDuration,
		PercentileMarginError: defaultPercentileMarginError,
	}
	if len(rawParameters) > 0 {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("costguard %q: failed to parse parameters: %w", name, err)
		}
	}

	if config.Epsilon < 0 || config.Epsilon > 1 {
		return nil, fmt.Errorf("costguard %q: epsilon must be in [0, 1], got %f", name, config.Epsilon)
	}
	if config.Alpha <= 0 || config.Alpha >= 1 {
		return nil, fmt.Errorf("costguard %q: alpha must be in (0, 1), got %f", name, config.Alpha)
	}
	if config.Lambda < 0 {
		return nil, fmt.Errorf("costguard %q: lambda must be >= 0, got %f", name, config.Lambda)
	}
	if config.PercentileMarginError <= 0 || config.PercentileMarginError >= 1 {
		return nil, fmt.Errorf("costguard %q: percentileMarginError must be in (0, 1), got %f", name, config.PercentileMarginError)
	}
	windowDuration, err := time.ParseDuration(config.WindowDuration)
	if err != nil {
		return nil, fmt.Errorf("costguard %q: invalid windowDuration %q: %w", name, config.WindowDuration, err)
	}
	if windowDuration <= 0 {
		return nil, fmt.Errorf("costguard %q: windowDuration must be > 0, got %s", name, windowDuration)
	}

	return NewCostGuardScorer(
		config.Epsilon,
		config.Alpha,
		config.Lambda,
		windowDuration,
		config.PercentileMarginError,
	).WithName(name), nil
}

// NewCostGuardScorer constructs a scorer with the given parameters.
// It performs no range checks; callers are responsible for passing values in
// the ranges documented on Config.
// ScorerFactory is the intended path for validated construction from raw JSON parameters.
func NewCostGuardScorer(epsilon, alpha, lambda float64, windowDuration time.Duration, percentileMarginError float64) *CostGuardScorer {
	return &CostGuardScorer{
		typedName:       plugin.TypedName{Type: PluginType, Name: PluginType},
		epsilon:         epsilon,
		alpha:           alpha,
		lambda:          lambda,
		sampleThreshold: sampleThreshold(alpha, percentileMarginError),
		windowDuration:  windowDuration,
	}
}

// TypedName returns the typed name of the plugin instance.
func (s *CostGuardScorer) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the instance name.
func (s *CostGuardScorer) WithName(name string) *CostGuardScorer {
	s.typedName.Name = name
	return s
}

// Score returns the neutral score for every model. The exploit and explore
// branches follow in subsequent PRs.
func (s *CostGuardScorer) Score(_ context.Context, _ *plugin.CycleState, _ *requesthandling.InferenceRequest, models []datalayer.Model) map[datalayer.Model]float64 {
	scores := make(map[datalayer.Model]float64, len(models))
	for _, m := range models {
		scores[m] = neutralScore
	}
	return scores
}

// sampleThreshold returns the minimum sample count for a model's alpha-quantile
// estimate to have a two-sided 95% CI of half-width w (Wald formula).
func sampleThreshold(alpha, w float64) uint64 {
	return uint64(math.Ceil(z95 * z95 * alpha * (1 - alpha) / (w * w)))
}
