Feature: Deterministic Rust curriculum buckets
  Scenario: Frozen empirical scoring
    Given a versioned reference calibrated from complete Rust samples
    When an unchanged sample is analyzed repeatedly
    Then its metrics, composite score and tier are identical
    And adding samples to the processing batch does not change the reference

  Scenario: Ties and context features
    Given two samples with identical primary metrics
    When their token and call context counts differ
    Then their composite scores and tiers are identical
    And scores exactly on a boundary remain in the lower tier

  Scenario: Unmeasurable samples
    Given malformed Rust, an unsupported language, or an unexpanded macro
    When samples are analyzed
    Then unavailable metrics are null and no structural tier is assigned
    And other samples continue to be processed

  Scenario: Observable training-compatible outputs
    Given a prepared raw Rust cache and a frozen reference
    When sort_code_buckets processes the cache
    Then every input appears in exactly one destination cache
    And records and a validation report identify calibration and sample provenance
