Feature: WIR Chess page
  The network judges, the host plays by the rules and searches.

  Scenario: The page opens on the starting position
    Given the WIR Chess page is open
    Then the board shows 64 squares and 32 pieces
    And the status says white is to move
    And the instinct panel ranks five network moves
    And no script errors were raised

  Scenario: The network answers a move
    When I play e2 to e4 by clicking
    Then the network replies within the think time
    And the move list shows two moves
    And the search ran in a worker

  Scenario: The lens shows what the network attends to
    When I turn on the attack map and the ray heads
    And I point at the white queen
    Then attacker counts and ray lines are drawn

  Scenario: Style weights are the last layer
    When I move the mobility weight to 0
    Then the judgment total changes

  Scenario: Playing black at Instinct level
    When I choose Instinct and play black
    Then the network opens the game

  Scenario: Phone width and dark theme
    Given a 400 px wide dark viewport
    Then the page does not scroll sideways
