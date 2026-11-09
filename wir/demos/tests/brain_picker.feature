Feature: Yumon Ant and Yumon Park load superposed brains
  The pages carry the exact compiled brain plus the variants from `wir ant|park --variant`
  (superposed brains from `wir superpose-choice-layer`). Driven by brain_picker.mjs in Chrome.

  Scenario: Ant offers its brains and runs a superposed one
    Given the Yumon Ant page built from ant_demo.json with two superposed variants
    When the page opens
    Then the Brain menu lists the exact brain and both superposed brains, exact selected
    When the viewer picks "Superposed, 6 shared neurons"
    Then the page reloads with #brain=1 and the networks table shows the bottleneck layer
    And the colony runs, with the check note saying the brain shares neurons
    And no script errors were logged

  Scenario: Park offers its brains in the ecosystem setup
    Given the Yumon Park page built from park_demo.json with one superposed variant
    When the viewer opens Ecosystem setup, picks "Superposed, 6 shared neurons" and saves
    Then the park restarts with that brain named under the source
    And the load check says the brain shares neurons
    And no script errors were logged
