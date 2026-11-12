Feature: WIR Beam page
  A written network picks the survivors of every beam search step; the host runs the loop.

  Scenario: The page opens on the first step
    Given the WIR Beam page is open
    Then the grid shows 8 beam rows and 16 word columns
    And width 4 is selected and 4 cells are highlighted as survivors
    And the network check says its ranks equal a plain sort
    And no script errors were raised

  Scenario: Stepping takes the network's survivors
    When I press Step
    Then four beams are listed, best first
    And the grid shows the next step for those beams

  Scenario: Running to the end finds the exact best sentence
    When I run the search from the start at width 4
    Then it stops when the cheapest beam has ended
    And the best sentence is "yumon sleeps" at -2.74 nats

  Scenario: Greedy gets it wrong and a wider beam fixes it
    When I choose width 1 from the start
    Then the search ends on a sentence that is not the exact best
    When I choose width 3
    Then the search ends on the exact best sentence

  Scenario: A coarse cost grain changes the picks
    When I choose 2 cost units per nat
    Then the reference table still lists every width

  Scenario: Scoring every prompt
    When I score all prompts
    Then greedy finds the exact best for 8 of 16
    And width 8 finds it for 16 of 16

  Scenario: Phone width and dark theme
    Then the page does not scroll sideways and the grid scrolls inside its frame
