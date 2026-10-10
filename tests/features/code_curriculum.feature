Feature: Cumulative code curriculum
  Scenario: Only low, mild and moderate tiers may be used
    Given frozen low, mild and moderate bucket caches
    And high and unscored caches are unavailable or corrupt
    When code training runs for five epochs
    Then epoch one trains on low samples
    And epoch two trains on low and mild samples
    And epoch three and subsequent epochs train on low, mild and moderate samples
    And no high, excluded, unscored or original mixed-cache samples are loaded

  Scenario: Fixed validation across expanding training pools
    Given source files with samples in multiple eligible tiers
    When the curriculum is prepared
    Then the file-level validation split is computed once across all eligible tiers
    And every sample from each validation file stays out of every training epoch

  Scenario: Resume at the correct curriculum epoch
    Given a checkpoint with two completed epochs
    When code training resumes with a five-epoch budget
    Then it starts at epoch three using low, mild and moderate samples
    And learning rate progress accounts for the smaller preceding epochs

  Scenario: RunPod image only includes eligible training data
    When the Docker image is built
    Then only low, mild and moderate caches and calibration metadata are copied
    And the CPU-only config check requires curriculum mode
