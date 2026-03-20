// why no work?

// use burn::{backend::Wgpu, tensor::Device};
// use floem::prelude::*;
// use yumon_pet::{brain::model::YumonBrain, vision::{CIFAR_CLASSES, EMOTE_CLASSES}};
// use std::{io, sync::mpsc, thread, time::{Duration, Instant}};

// fn main() {
//     let (tx_prompt, rx_prompt) = crossbeam_channel::bounded::<String>(1);
//     let (tx_reply, rx_reply) = crossbeam_channel::bounded::<String>(1);

//     // Spawn model thread exactly as before
//     let device: Device<Wgpu> = Default::default();
//     thread::spawn(move || {
            
//             let (model, tokenizer) = YumonBrain::<Wgpu>::load(
//                 "checkpoints/brain", &device
//             ).expect("Failed to load");

//             // Neutral context for now
//             let class_probs = vec![1.0 / CIFAR_CLASSES as f32; CIFAR_CLASSES];
//             let emote_probs = vec![1.0 / EMOTE_CLASSES as f32; EMOTE_CLASSES];

//             while let Ok(prompt) = rx_prompt.recv() {
//                 let result = model.generate(
//                     &tokenizer,
//                     &class_probs,
//                     &emote_probs,
//                     4,
//                     &prompt,
//                     80,
//                     &device,
//                 );
//                 tx_reply.send(result.reply).unwrap();
//             }
//     });

//     floem::launch(move || app_view(tx_prompt, rx_reply));
// }

// use floem::prelude::*;

// fn app_view(
//     tx: crossbeam_channel::Sender<String>,
//     rx: crossbeam_channel::Receiver<String>,
// ) -> impl IntoView {
//     let input = RwSignal::new(String::new());
//     let messages = RwSignal::new(Vec::<(usize, bool, String)>::new()); // (id, is_yumon, text)
//     let thinking = RwSignal::new(false);
//     let emote = RwSignal::new("neutral");
//     let next_id = RwSignal::new(0usize);

//     // Poll for replies from model thread
//     floem::ext_event::create_signal_from_channel(rx)
//         .with(move |reply| {
//             let id = next_id.get();

//             if let Some(reply) = reply {
//                 next_id.update(|n| *n += 1);
//                 messages.update(|m| m.push((id, true, reply.clone())));
//             }

//             thinking.set(false);
//         });

//     v_stack((
//         // Chat history
//         scroll(
//             dyn_stack(
//                 move || messages.get(),
//                 |(id, _, _)| *id,
//                 |(_, is_yumon, text)| {
//                     label(move || if is_yumon {
//                         format!("Yumon: {text}")
//                     } else {
//                         format!("You: {text}")
//                     })
//                 },
//             )
//         ).style(|s| s.flex_grow(1.0)),

//         // Yumon face
//         label(move || match emote.get() {
//             "happy" => "^‿^",
//             "sad"   => "ಥ_ಥ",
//             _       => "•‿•",
//         }).style(|s| s.font_size(48.0).padding(20)),

//         // Input row
//         h_stack((
//             text_input(input)
//                 .style(|s| s.flex_grow(1.0)),
//             button("Send").action(move || {
//                 let text = input.get();
//                 if !text.is_empty() && !thinking.get() {
//                     let id = next_id.get();
//                     next_id.update(|n| *n += 1);
//                     messages.update(|m| m.push((id, false, text.clone())));
//                     tx.send(text).unwrap();
//                     input.set(String::new());
//                     thinking.set(true);
//                 }
//             }),
//         )).style(|s| s.gap(8).padding(8)),

//         // Status
//         label(move || if thinking.get() {
//             "Yumon is thinking"
//         } else {
//             "".into()
//         }),
//     ))
//     .style(|s| s.size_full())
// }

pub fn main() {}