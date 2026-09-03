# Yumon Pet

Yumon is a tabletop ePet that responds to various inputs with short text replies and emotes.

## Hyperparameters

### Medium

```
pub const EMBED_DIM:    usize = 256;
pub const HIDDEN_UNITS: usize = 256;
pub const ATTN_HEADS:   usize = 2;
pub const N_LAYERS:     usize = 2;
pub const FF_DIM:       usize = 1024;
pub const TEMPERATURE:  f32   = 0.9;
pub const TOP_K:        usize = 10;
pub const MAX_SEQ_LEN:  usize = 280;
```

### Small

```
pub const EMBED_DIM:    usize = 64;
pub const HIDDEN_UNITS: usize = 64;
pub const ATTN_HEADS:   usize = 2;
pub const N_LAYERS:     usize = 2;
pub const FF_DIM:       usize = 256;
pub const TEMPERATURE:  f32   = 0.9;
pub const TOP_K:        usize = 10;
pub const MAX_SEQ_LEN:  usize = 320;
```

## Get Started

You may need clang for chat_web.

- `cargo run --release --bin yumon-pet -- train-brain` to start training on the provided (or your own) dataset
- `cargo run --release --bin chat_ui` to get started chatting
- `cargo run --release --bin yumon_world` to start a Yumon World simulation
- `cargo run --release --bin endless_data` TUI to answer endless questions in order to generate some data
- `cargo run --release --bin train_bpe` train your tokenizer on your data

- `trunk serve --release` for chat web (or `trunk build --release` for deployment)

### Training on RunPod (Docker)

The training path (`brain::train::run`, driven by `train-brain`) builds headless with
`--no-default-features` - this skips the desktop/GUI feature (native window, webview,
3D engine, gamepad) that the other bins use, so the Docker image needs no GTK/WebKit/
udev packages. It still uses burn's `Wgpu` backend (Vulkan), so all it needs at
runtime is `libvulkan1` + the GPU's driver.

1. **Build and push the image** (from a machine with Docker and the full repo checked
   out, since the image bakes in `yumon_bpe/` and the text training data):

   ```
   docker build -t alexthegoodman/yumon-brain:latest .
   docker push alexthegoodman/yumon-brain:latest
   ```

2. **Create a RunPod Network Volume** (Storage → Network Volumes) sized for your
   checkpoints, in the same region as the pod you'll launch.

3. **Launch a GPU Pod** from your custom image (`<registry>/yumon-brain:latest`),
   attaching the Network Volume at `/workspace`. The container's default command runs
   `train-brain` and writes checkpoints to `/workspace/checkpoints/brain/<run-name>/`
   (`model.bin`, `metadata.json`, tokenizer copy, and a loss-chart PNG per stage) -
   same run configs (model sizes/epochs/stages) as `src/brain/train.rs` uses locally,
   since the CLI's `--epochs`/`--batch-size`/`--max-articles` flags are legacy no-ops
   for this path; edit the `runs` vec in that file to change them. Training resumes
   automatically from whatever's already in a run's checkpoint directory.

4. **Pull checkpoints down** once you're happy with a run (or periodically - it saves
   every epoch and every 500 batches): easiest is the RunPod web File Manager on the
   pod/volume, or `runpodctl send`/`scp` if you've enabled SSH on the pod, to copy
   `/workspace/checkpoints/brain/` to your machine.

If wgpu can't find a GPU at startup, exec into the pod and run `vulkaninfo --summary`
to confirm the NVIDIA Vulkan ICD is visible (RunPod's nvidia-container-toolkit should
mount it automatically given `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`,
already set in the image).

### Datasets

- Custom / Bespoke
- https://www.kaggle.com/datasets/lmsysorg/chatbot-arena-conversations
- https://www.kaggle.com/datasets/thedevastator/distillchat-v1-mixture-of-conversations-dataset

## Evaluation

### Yumon characteristics

Primary:
- Teachable (do this, go there, get that + reward signals and lesson cache)
- Conversational (what do you think about... + memory strength)
- Smart (model parameters + data)

Secondary:
- Loyal
- Connective
- Entertaining
- Organizational
- Affordable

## TODO

- Training UI (train tokenizer, organize data, run structured and unstructured training sessions, etc)