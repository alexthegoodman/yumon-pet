Feature: Yumon Portal is simulations on the left and chat on the right

  Scenario: The portal opens with a simulation beside the chat
    Given the portal page is loaded with a stub host
    Then the Yumon Ant simulation is on the left and the chat is on the right
    And there is no Products, Paint or 3D universe view

  Scenario: Chat round trip goes through the host
    Given the portal page is loaded with a stub host
    When the host reports Yumon is ready
    And I type "Hello Yumon" and press Enter
    Then the host receives a chat message "Hello Yumon"
    When the host answers with "Hello friend!"
    Then the conversation shows "You" and "Yumon" bubbles

  Scenario: The simulation can be switched between Yumon Ant and WIR Chess
    Given the portal page is loaded with a stub host
    Then the Yumon Ant page runs inside the portal
    When I pick WIR Chess
    Then the WIR Chess board has 64 squares and the network ranks candidate moves
    And the chat is still on the right
