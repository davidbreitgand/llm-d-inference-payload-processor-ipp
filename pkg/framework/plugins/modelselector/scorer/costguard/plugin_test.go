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

package costguard

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-payload-processor/pkg/framework/interface/requesthandling"
)

func newTestScorer(t *testing.T) *CostGuardScorer {
	t.Helper()
	p, err := ScorerFactory("test-cg", nil, nil)
	require.NoError(t, err)
	s, ok := p.(*CostGuardScorer)
	require.True(t, ok)
	return s
}

// TestFactory_DefaultConfig verifies factory behavior with nil parameters and
// with an empty JSON object; both must produce the same defaulted scorer.
func TestFactory_DefaultConfig(t *testing.T) {
	tests := []struct {
		name string
		raw  json.RawMessage
	}{
		{"nil parameters", nil},
		{"empty object", json.RawMessage(`{}`)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := ScorerFactory("test-cg", tt.raw, nil)
			require.NoError(t, err)
			s, ok := p.(*CostGuardScorer)
			require.True(t, ok)
			assert.Equal(t, defaultEpsilon, s.epsilon)
			assert.Equal(t, defaultAlpha, s.alpha)
			assert.Equal(t, defaultLambda, s.lambda)
			assert.Equal(t, 2*time.Hour, s.windowDuration)
			assert.Equal(t, sampleThreshold(defaultAlpha, defaultPercentileMarginError), s.sampleThreshold)
			assert.Equal(t, PluginType, s.TypedName().Type)
			assert.Equal(t, "test-cg", s.TypedName().Name)
		})
	}
}

// TestFactory_CustomConfig verifies that custom parameters override defaults.
func TestFactory_CustomConfig(t *testing.T) {
	raw := json.RawMessage(`{"epsilon":0.2,"alpha":0.9,"lambda":2.0,"windowDuration":"30m","percentileMarginError":0.05}`)
	p, err := ScorerFactory("custom", raw, nil)
	require.NoError(t, err)
	s := p.(*CostGuardScorer)
	assert.Equal(t, 0.2, s.epsilon)
	assert.Equal(t, 0.9, s.alpha)
	assert.Equal(t, 2.0, s.lambda)
	assert.Equal(t, 30*time.Minute, s.windowDuration)
	assert.Equal(t, sampleThreshold(0.9, 0.05), s.sampleThreshold)
}

// TestFactory_ValidationErrors verifies that each out-of-range or malformed
// parameter causes the factory to return an error.
func TestFactory_ValidationErrors(t *testing.T) {
	tests := []struct {
		name string
		raw  string
	}{
		{"malformed json", `{invalid`},
		{"epsilon below 0", `{"epsilon":-0.1}`},
		{"epsilon above 1", `{"epsilon":1.1}`},
		{"alpha zero", `{"alpha":0}`},
		{"alpha one", `{"alpha":1}`},
		{"alpha above 1", `{"alpha":1.5}`},
		{"lambda negative", `{"lambda":-0.1}`},
		{"pme zero", `{"percentileMarginError":0}`},
		{"pme one", `{"percentileMarginError":1}`},
		{"windowDuration unparsable", `{"windowDuration":"not-a-duration"}`},
		{"windowDuration zero", `{"windowDuration":"0s"}`},
		{"windowDuration negative", `{"windowDuration":"-1s"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ScorerFactory("bad", json.RawMessage(tt.raw), nil)
			require.Error(t, err)
		})
	}
}

// TestFactory_AcceptedBoundaries verifies that the boundary values documented
// as accepted stay accepted; pins down current strict-vs-inclusive semantics
// so a refactor cannot flip a guard without a failing test.
func TestFactory_AcceptedBoundaries(t *testing.T) {
	tests := []struct {
		name  string
		raw   string
		check func(*testing.T, *CostGuardScorer)
	}{
		{
			"epsilon at lower bound",
			`{"epsilon":0}`,
			func(t *testing.T, s *CostGuardScorer) { assert.Equal(t, 0.0, s.epsilon) },
		},
		{
			"epsilon at upper bound",
			`{"epsilon":1}`,
			func(t *testing.T, s *CostGuardScorer) { assert.Equal(t, 1.0, s.epsilon) },
		},
		{
			"lambda at lower bound",
			`{"lambda":0}`,
			func(t *testing.T, s *CostGuardScorer) { assert.Equal(t, 0.0, s.lambda) },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := ScorerFactory("boundary", json.RawMessage(tt.raw), nil)
			require.NoError(t, err)
			tt.check(t, p.(*CostGuardScorer))
		})
	}
}

// --- WithName test ---

func TestWithName(t *testing.T) {
	s := newTestScorer(t)
	result := s.WithName("custom-name")
	assert.Same(t, s, result, "WithName should return the same instance for chaining")
	assert.Equal(t, "custom-name", s.TypedName().Name)
}

// --- Helper tests ---

// TestSampleThreshold verifies the Wald CI sample-size formula against the
// values documented in the CostGuard README.
func TestSampleThreshold(t *testing.T) {
	assert.Equal(t, uint64(203), sampleThreshold(0.95, 0.03))
	assert.Equal(t, uint64(73), sampleThreshold(0.95, 0.05))
}

// --- Score tests (PR 1 stub returns neutral for every model) ---

func TestScore_AllNeutral(t *testing.T) {
	s := newTestScorer(t)
	models := []datalayer.Model{
		datalayer.NewModel("m1"),
		datalayer.NewModel("m2"),
		datalayer.NewModel("m3"),
	}
	scores := s.Score(context.Background(), plugin.NewCycleState(), requesthandling.NewInferenceRequest(), models)
	require.Len(t, scores, len(models))
	for _, m := range models {
		assert.Equal(t, neutralScore, scores[m], "model %s expected neutralScore", m.GetName())
	}
}

func TestScore_EmptyModels(t *testing.T) {
	s := newTestScorer(t)
	scores := s.Score(context.Background(), plugin.NewCycleState(), requesthandling.NewInferenceRequest(), nil)
	assert.Empty(t, scores)
}
