# Yumon Pet

Yumon is a tabletop ePet that responds to various inputs with short text replies and emotes.

We will be leveraging the Rust Burn framework for this task.

Architecture:
- Computer Vision CNN: 2 heads, one for classification and one for emote detection
- Small Language Model LSTM: Outputs short messages (perhaps tweet-sized)
- Final Model: Include CNN classification and emote probabilities in LSTM input, with the text message and Yumon's own emote response

It is up to you the details of this architecture.

Available Datasets:
- `./data/cifar-100-binary`
    - coarse_label_names.txt
    - fine_label_names.txt
    - test.bin
    - train.bin

- `./data/fer2013-archive`
    - /test/[emote]/ImageName_123.jpg
    - /train/[emote]/ImageName_345.jpg

- `./data/simplewiki-latest-pages-articles.xml`

- Later, we can create specialized fine-tuning reply-emote data, but for now, we can associate classification and emotes with wiki data to provide more dry responses and emotes.
Yumon emotes for now can be purely empathetic, based on user emotes.