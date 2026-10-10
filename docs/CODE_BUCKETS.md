# User Story: Deterministic Code Complexity Scoring and Tier Classification

**Story Type:** Feature  
**Component:** Data Processing Pipeline  
**Priority:** High  
**Status:** Proposed

## User Story

**As a** data pipeline engineer,  
**I want** source-code samples to be automatically analyzed, scored, and classified into four empirical complexity tiers using deterministic static-analysis metrics,  
**So that** downstream datasets, benchmarks, and evaluation workflows can consistently segment code by structural complexity without relying on LLM-generated judgments or arbitrary complexity thresholds.

## Background

Our data processing pipeline requires a reproducible method for quantifying the structural complexity of source-code samples.

The classification system must operate programmatically using language-aware parsing, Abstract Syntax Trees (ASTs), token streams, and, where supported, control-flow and data-flow analysis.

The system will produce a normalized composite complexity score and assign each sample to one of four tiers based on a fixed reference population.

The classification is intended to represent **relative structural code complexity**, not a direct or independently validated measure of LLM task difficulty.

## Functional Requirements

### FR-1: Source-Code Analysis

For each eligible source-code sample, the pipeline shall:

- Identify the programming language.
- Parse the sample using a supported language-aware parser.
- Determine the configured analysis unit (function/method by default).
- Extract deterministic structural metrics.
- Record parser and analyzer versions.
- Handle unsupported languages, malformed code, and incomplete snippets without terminating the overall pipeline.

Generated, vendored, and minified code shall be excluded or separately classified according to configurable preprocessing rules.

### FR-2: Complexity Metric Extraction

The pipeline shall calculate the following metrics where supported:

| Metric | Description | Usage |
|---|---|---|
| Cyclomatic Complexity | Independent control-flow paths | Primary |
| Cognitive Complexity | Nesting and interruptions to linear control flow | Primary |
| Control-Flow Nesting Depth | Maximum semantic nesting of control structures | Primary |
| State Mutation Complexity | Variable assignments, reassignment, and state changes | Primary |
| Halstead Volume | Lexical operator and operand volume | Primary |
| Token Count | Source-code length | Context feature |
| AST Node Count | Parse-tree size | Diagnostic |
| Expression Nesting Depth | Nested calls and expressions | Diagnostic |
| Dependency / Call Count | External references and function calls | Context feature |

Metric definitions and language-specific counting rules must be documented and version-controlled.

Unavailable metrics shall be explicitly marked rather than silently treated as zero.

### FR-3: Normalization and Composite Scoring

The pipeline shall normalize primary metrics using empirical percentile ranks derived from a fixed, versioned reference population.

The initial proposed composite score is:

\[
S = 0.25P(CC) + 0.30P(CogC) + 0.20P(D) + 0.15P(State) + 0.10P(V)
\]

Where:

- \(P\) represents the reference-population percentile rank.
- \(CC\) represents cyclomatic complexity.
- \(CogC\) represents cognitive complexity.
- \(D\) represents control-flow nesting depth.
- \(State\) represents state mutation complexity.
- \(V\) represents Halstead volume.

Weights shall be configurable and versioned. These initial weights are provisional and subject to empirical validation.

The pipeline shall also retain token count and dependency-related context features separately from the structural composite.

Missing primary metrics must follow an explicit, documented scoring policy.

### FR-4: Four-Tier Classification

The pipeline shall classify samples using quartile boundaries derived from the reference population.

| Tier | Reference Percentile | Classification |
|---|---|---|
| Tier 1 | 0–25th | Low Structural Complexity |
| Tier 2 | 25th–50th | Mild Structural Complexity |
| Tier 3 | 50th–75th | Moderate Structural Complexity |
| Tier 4 | 75th–100th | High Structural Complexity |

Classification boundaries shall remain fixed for a given scoring configuration.

Samples with identical composite scores must receive identical tiers. Exact 25% representation in each tier is not required when ties prevent it.

The system shall support language-specific calibration when necessary to avoid misleading cross-language comparisons.

### FR-5: Output Schema

Each processed sample shall include a structured complexity record containing:

```json
{
  "sample_id": "sample_001",
  "language": "python",
  "analysis_unit": "function",
  "metrics": {
    "cyclomatic_complexity": 5,
    "cognitive_complexity": 8,
    "control_flow_depth": 3,
    "state_mutation_complexity": 4,
    "halstead_volume": 125.4,
    "token_count": 180,
    "ast_node_count": 92,
    "expression_depth": 5,
    "dependency_count": 3
  },
  "structural_complexity_score": 0.63,
  "complexity_tier": 3,
  "scoring_version": "1.0.0",
  "reference_population_version": "1.0.0",
  "analysis_status": "success"
}
```

*Values above are illustrative.*

The production schema shall additionally support nullable metrics, analysis errors, parser/analyzer versions, and provenance metadata.

### FR-6: Reproducibility and Versioning

The pipeline shall:

- Produce identical results for identical inputs under the same configuration and analyzer versions.
- Version the metric definitions, normalization reference population, scoring weights, and tier boundaries.
- Support reprocessing datasets when scoring methodologies change.
- Preserve sufficient metadata to reproduce historical classifications.
- Prevent changes to the current processing batch from automatically shifting established tier boundaries.

### FR-7: Validation and Quality Monitoring

The scoring system shall be evaluated against simpler baselines, including token count alone, cyclomatic complexity alone, and cognitive complexity alone.

Validation shall examine:

- Correlation and redundancy between metrics.
- Sensitivity to scoring weights.
- Tier stability under minor code changes.
- Language-specific distribution differences.
- Relationships between structural complexity and downstream task outcomes, where available.

The pipeline shall report tier distributions, analysis failure rates, missing-metric rates, and scoring-version usage.

## Acceptance Criteria

### AC-1: Deterministic Extraction

**Given** a valid source-code sample in a supported language,  
**When** the sample enters the complexity analysis stage,  
**Then** the system extracts the supported structural metrics without making an LLM inference call.

### AC-2: Reproducible Scoring

**Given** identical source code and identical analyzer, scoring, and reference-population versions,  
**When** the sample is processed multiple times,  
**Then** the resulting metric values, composite score, and tier assignment are identical.

### AC-3: Empirical Classification

**Given** a successfully scored sample,  
**When** its composite score is compared against the configured reference-population boundaries,  
**Then** it is assigned exactly one tier between 1 and 4.

### AC-4: Consistent Tie Handling

**Given** two samples with identical composite scores under the same scoring configuration,  
**When** their tiers are assigned,  
**Then** both receive the same tier.

### AC-5: Size Independence

**Given** samples with different token counts,  
**When** their structural complexity scores are calculated,  
**Then** token count does not directly contribute to the structural composite, and is retained as a separate context feature.

### AC-6: Graceful Error Handling

**Given** a malformed sample or unsupported language,  
**When** complexity analysis cannot be completed,  
**Then** the pipeline records a structured failure or unsupported status, does not fabricate metric values, and continues processing other samples.

### AC-7: Versioned Calibration

**Given** an established scoring configuration,  
**When** new samples are added to the dataset,  
**Then** the existing reference-population percentile mappings and tier boundaries remain unchanged until an explicitly versioned recalibration occurs.

### AC-8: Observable Outputs

**Given** a completed processing batch,  
**When** pipeline results are inspected,  
**Then** individual complexity records and aggregate tier-distribution statistics are available for downstream consumption.

### AC-9: Quality Validation

**Given** a representative validation dataset,  
**When** the scoring methodology is evaluated,  
**Then** the pipeline produces a report covering feature correlations, tier distributions, baseline comparisons, and scoring stability.

## Non-Functional Requirements

**Performance:** Complexity analysis must support batch processing at dataset scale. Throughput and latency targets will be established through profiling.

**Reliability:** Failure to analyze an individual sample must not cause the entire processing batch to fail.

**Extensibility:** New programming languages, metrics, and scoring configurations must be addable without redesigning the overall pipeline.

**Auditability:** Every assigned tier must be traceable to its underlying metric values, scoring configuration, and calibration version.

**Determinism:** No LLM inference or nondeterministic judgment may participate in the production complexity scoring path.

## Out of Scope

- LLM-generated complexity assessments.
- Semantic correctness verification.
- Runtime performance profiling.
- Security vulnerability classification.
- Automatic code refactoring.
- Claims that structural tiers directly measure LLM reasoning difficulty.
- Training a predictive difficulty model as part of the initial implementation.

## Definition of Done

- [ ] Supported languages and analysis units are documented.
- [ ] Deterministic metric extractors are implemented and tested.
- [ ] Normalization and composite scoring are implemented.
- [ ] A versioned reference population is established.
- [ ] Four-tier classification is implemented with deterministic tie handling.
- [ ] Structured output records are integrated into the data pipeline.
- [ ] Parsing failures and missing metrics are handled explicitly.
- [ ] Scoring configuration and analyzer versions are recorded.
- [ ] Unit and integration tests verify reproducibility.
- [ ] Baseline comparison and tier-distribution reports are available.
- [ ] Pipeline documentation and operational monitoring are complete.

## Implementation Notes

A phased implementation is recommended.

**Phase 1 — Baseline:** Implement language-aware parsing, cyclomatic complexity, cognitive complexity, control-flow depth, Halstead volume, token counts, and the initial output schema.

**Phase 2 — Scoring and Calibration:** Establish the reference population, implement percentile normalization, configure composite weights, and assign four tiers.

**Phase 3 — Enhanced Analysis:** Add state mutation and dependency metrics, finalize missing-feature policies, and evaluate whether these additions materially improve the classification. The full proposed composite should not be activated until its required metrics are available.

**Phase 4 — Validation and Operationalization:** Evaluate metric redundancy, classification stability, language effects, and downstream task-performance relationships. Finalize scoring configuration and integrate monitoring.

## Expected Outcome

The data processing pipeline produces a deterministic, auditable, and reproducible complexity classification for every eligible source-code sample.

These classifications enable consistent dataset segmentation, stratified sampling, balanced benchmark construction, and downstream analysis of how structural code complexity relates to model performance.

The methodology can evolve through explicit versioned calibration without sacrificing reproducibility of previously processed datasets.

## Notes

- Keep in mind that our source data is still in `../../rust-code`, but has been pre-chunked into `./training-cache/code.bin` (by cache_samples.rs) so we can pull straight from `code.bin`
- The sorted buckets of data can be in our `./training-cache/code_buckets/bucket_name.bin`
- When running the sorting program (a Rust file in bin folder), we should be able to see some of samples and their destination bucket to verify correctness
- This is for a human-like curriculum training approach where we do the simpler data for epoch 1, and add in more complex data in future epochs