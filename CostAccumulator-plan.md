# CostAccumulator Multi-PR Implementation Plan

## Overview

This is a 3-PR phased delivery for integrating the request-cost-metadata extractor into the llm-d payload processor framework. The extractor accumulates per-model cost samples using t-digests and publishes distribution snapshots to the model's attribute map, enabling cost-aware scoring and observability.

**Completed PRs:**

- PR 1: Cost t-digest accumulator infrastructure (`CostDigest` type, attribute keys) — MERGED
- PR 2: RequestCostMetadataExtractor plugin implementation — MERGED

**In Progress:**

- PR 3: Configuration and registration (THIS DOCUMENT)

---

## PR 3: Configuration and Registration

### Objective

Wire the requestcostmetadata extractor into the plugin registry and enable end-to-end configuration from YAML, making it available for operators to deploy.
Wire the requestcostmetadata extractor into the plugin registry and enable end-to-end configuration from YAML, making it available for operators to deploy.

### Scope

#### 1. Plugin Registration (cmd/runner/runner.go)

**Status:** TODO

Add the requestcostmetadata extractor to the global plugin registry in the `registerInTreePlugins()` method at line 306.

**Changes:**

- Import `requestcostmetadata` package
- Call `plugin.Register(requestcostmetadata.PluginType, requestcostmetadata.ExtractorFactory)` after the requestmetadata registration

**Why:** The plugin registry is a map consulted at config load time. Plugins must be registered before any configuration can instantiate them.

**Testing:** Unit test in runner_test.go verifies the registration succeeds and can be looked up.

---

#### 2. Integration Example and Documentation
**Status:** TODO

Update the example payload processor config and add plugin documentation following the established pattern.

**Files to update:**
- [deploy/examples/payloadprocessorconfig.yaml](deploy/examples/payloadprocessorconfig.yaml) — add example datalayer extractor config section
- [pkg/framework/plugins/datalayer/requestcostmetadata/README.md](pkg/framework/plugins/datalayer/requestcostmetadata/README.md) — create new plugin README following the pattern from other plugins (e.g., random picker)
- [docs/plugins.md](docs/plugins.md) — add requestcostmetadata to the data-layer extractors section (if it exists)

**Example config block (YAML) for deploy/examples/payloadprocessorconfig.yaml:**

```yaml
plugins:
- name: cost-extractor
  type: model-cost-extractor
  parameters:
    compression: 200
    flushIntervalDuration: "5s"

datalayer:
  extractors:
  - pluginRef: cost-extractor
```

**Example plugin README (pkg/framework/plugins/datalayer/requestcostmetadata/README.md):**

```markdown
# Request Cost Metadata Extractor

Accumulates per-model cost samples from inference responses and publishes cost distribution 
snapshots (t-digests) to the model's attribute map. Enables cost-aware scoring and observability.

It is registered as type `model-cost-extractor` and runs as a data-layer extractor.

## What it does

1. Extracts prompt and completion token counts from each response's usage metadata.
2. Looks up per-token pricing for the model from the model's attribute map.
3. Computes request cost = (prompt_tokens × input_price) + (completion_tokens × output_price).
4. Adds the cost sample to a per-model t-digest (compressed distribution).
5. Publishes a t-digest snapshot to the model's `CostDigest` attribute on the configured flush interval.

## Behavioral Intent

Provides a memory-efficient (constant space per model) view of the cost distribution across requests. 
Enables operators to track cost trends, detect anomalies, and drive cost-aware model selection.

## Inputs consumed

- `request.body["model"]` — the model name (string)
- `response.body["usage"]["prompt_tokens"]` and `response.body["usage"]["completion_tokens"]` — token counts (floats, must be > 0)
- Model's `TokenPrices` attribute (set by the pricing data source) — pricing per token for input and output

## Outputs published

- Model's `CostDigest` attribute (updated on flush interval) — a t-digest snapshot with per-request costs

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compression` | float | 200.0 | T-digest compression factor. Higher values trade memory for accuracy. Must be > 0. |
| `flushIntervalDuration` | string | "5s" | Aggregation window before publishing a cost snapshot. Set to "0s" to publish on every event (test only). |

Example:

```yaml
plugins:
- name: cost-extractor
  type: model-cost-extractor
  parameters:
    compression: 200
    flushIntervalDuration: "5s"

datalayer:
  extractors:
  - pluginRef: cost-extractor
```
```

**Why:** Operators need example configurations and documentation to discover and use this feature. Consistent plugin README patterns make the documentation discoverable and searchable. An unmapped plugin that's not documented is effectively invisible.

**Testing:** No unit tests (docs don't fail tests), but validated by running end-to-end with the example config.

---

#### 3. End-to-End Test (if time permits)
**Status:** OPTIONAL / DEFERRED

If PR 2 left room and this PR's scope allows, add an integration test that:
1. Loads the example config with the cost extractor registered
2. Sends a mock request–response pair with usage and model fields
3. Verifies a cost digest appears in the model's attribute map

This is a confidence-building smoke test; functional correctness of the extractor was validated in PR 2's unit tests.

**Files:**
- [pkg/config/loader/loader_test.go](pkg/config/loader/loader_test.go) or new `integration_test.go` if pattern exists
- Add a test case under a `TestConfigLoad*` or `TestIntegration*` suite

**Why:** Config loading is a separate layer from plugin instantiation. An integration test ensures the YAML deserialization and plugin factory wiring all work together.

**Testing:** Integration test runs as part of `go test ./...`

---

### Implementation Checklist

- [ ] **Import requestcostmetadata in runner.go**
  - Add import statement at top of file
  - Target: line ~40 (with other plugin imports)

- [ ] **Register plugin in registerInTreePlugins()**
  - Add `plugin.Register(requestcostmetadata.PluginType, requestcostmetadata.ExtractorFactory)`
  - Target: line ~306 (after requestmetadata registration)
  - Run `go build ./cmd/runner` to verify no import errors

- [ ] **Add test for registration**
  - Update [cmd/runner/runner_test.go](cmd/runner/runner_test.go) to verify the plugin is discoverable
  - Confirm `plugin.Registry[requestcostmetadata.PluginType]` is not nil

- [ ] **Update example config**
  - Edit [deploy/examples/payloadprocessorconfig.yaml](deploy/examples/payloadprocessorconfig.yaml)
  - Add a `plugins` entry for cost-extractor
  - Add a `datalayer.extractors` reference to cost-extractor

- [ ] **Create plugin README**
  - Create [pkg/framework/plugins/datalayer/requestcostmetadata/README.md](pkg/framework/plugins/datalayer/requestcostmetadata/README.md)
  - Follow the pattern from other plugins (e.g., random picker)
  - Include: what it does, behavioral intent, inputs/outputs, configuration

- [ ] **Update docs/plugins.md** (if it lists extractors)
  - Add requestcostmetadata to the data-layer extractors section
  - Link to the plugin README

- [ ] **(Optional) End-to-end config test**
  - Create integration test verifying YAML→plugin→cost sample flow
  - Defer if time-boxed

- [ ] **Run full test suite**
  - `go test ./...`
  - `go build ./cmd/runner`
  - No regressions in existing tests

---

### Files Changed

**Must:**

- `cmd/runner/runner.go` — import and register plugin
- `cmd/runner/runner_test.go` — test registration
- `deploy/examples/payloadprocessorconfig.yaml` — add example config
- `pkg/framework/plugins/datalayer/requestcostmetadata/README.md` — create plugin documentation

**Should:**

- `docs/plugins.md` — add extractor to documentation (if it lists plugins)

**Nice to have:**

- Integration test (can be a follow-up)

---

### Verification Checklist

**Before submitting PR:**

1. ✅ Plugin registry lookup returns requestcostmetadata's factory
2. ✅ Example YAML config parses without error
3. ✅ `go test ./...` passes
4. ✅ `go build ./cmd/runner` builds successfully
5. ✅ No new linting failures (`golangci-lint run ./...`)

**Manual smoke test (optional but recommended):**

1. Start the payload processor with the example config (if deployment instructions exist)
2. Send a request–response pair with a model name and usage fields
3. Observe that cost samples are accumulated (check logs at DEBUG level)
4. Verify that the model's cost digest appears in the datastore attribute map

---

### Success Criteria

This PR is complete when:
1. The requestcostmetadata extractor is registered and discoverable via the plugin registry
2. Example YAML configuration shows operators how to deploy it
3. All tests pass and no regressions are introduced
4. An operator can follow the example config to enable cost accumulation without code changes

---

### Known Risks / Assumptions

- **Plugin registry immutability:** The global `Registry` map is written to once at startup. If a plugin is loaded twice (e.g., in tests), the second registration silently overwrites the first. This is expected; no conflict is anticipated here.

- **Configuration schema:** The plugin uses `RequestCostMetadataExtractorConfig` with optional fields defaulting to proposal values. Operators unfamiliar with t-digests may not understand `compression`; consider adding comments to the example or docs.

- **Empty model creation:** The `lookupTokenPrices` function uses `GetOrCreateModel`, which creates an empty model in the datastore if pricing is absent. This is a known side-effect noted in the code comment; deferred to a follow-up PR for a read-only `GetModel` method.

---

### Next Steps (After PR 3)

1. **Observability:** Add metrics/logging for cost digest publishing (e.g., histogram of cost quantiles)
2. **Scorer Integration:** Wire a cost-aware scorer (e.g., CostGuard) to consume cost digest snapshots
3. **Configuration Schema:** CRD or config validation to catch invalid `compression` or `flushIntervalDuration` values earlier
4. **Test Expansion:** E2E test with real inference mock and multi-model scaling

---

## Appendix: Related Code Locations

- Plugin interface: `pkg/framework/interface/plugin/plugin.go`
- Registry: `pkg/framework/interface/plugin/registry.go`
- Extractor interface: `pkg/framework/interface/datalayer/datasource/datasource.go`
- Accumulator types: `pkg/framework/interface/datalayer/accumulator/accumulator.go`
- Runner config loading: `cmd/runner/runner.go` (lines 240–310)
- Example configs: `deploy/examples/`
- Test utils: `pkg/config/loader/testdata_test.go`

