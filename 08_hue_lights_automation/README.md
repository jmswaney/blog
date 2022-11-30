# Automating Phillips Hue lights in Python

Value: Learn how to progressively do cooler things with the Phillips Hue lights.

- Run script to change the light color in a sinusoidal pattern
    - Use to automatically change the color temp during the day
- React to webhook events with certain light programs
    - CI pipeline went down give a RED ALERT!
    - CI pipeline integration successful gives GREEN!

Probably need to make a websocket connection to make this happen without creating an endpoint. Slack API's Socket Mode may work just fine without having to expose a public endpoint.
