Feature: Yumon Code cached Rust pretraining
  # Executable acceptance coverage: tests/code_corpus/mod.rs (CPU only).
  Scenario: Recursively cache Rust files and hold out entire files
    Given nested Rust sources and excluded build output
    When a raw-code cache is prepared and loaded
    Then token tensors are unchanged and train and validation files are disjoint
    And prompt-reply loading and mismatched context lengths are rejected

  Scenario: Preserve complete items and supervise every selected token
    Given Rust with mixed case, Unicode, CRLF, tabs and a literal control-token name
    When it is chunked at contexts from 2 to 512 tokens
    Then only complete structs and functions that fit are emitted
    And oversized items are skipped rather than truncated
    And next-token targets cover all selected code tokens and the final EOS

  Scenario: Parse logical boundaries instead of counting braces
    Given attributes, doc comments, inline modules, impl methods and trait methods
    And braces inside nested comments and raw strings
    Then each selected item retains its complete body and attributes
    And methods retain an impl or trait header and a closing brace
    And macro bodies and abstract method signatures are not selected

  Scenario: Fail explicitly instead of silently losing source
    Given duplicate source files and invalid UTF-8 input
    Then duplicate files are removed and invalid input fails preparation
    And existing tokenizers and caches cannot be overwritten
