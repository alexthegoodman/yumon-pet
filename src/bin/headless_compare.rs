// #![recursion_limit = "256"]
// #![allow(warnings)]

// use anyhow::Result;
// use burn::backend::Wgpu;
// use cubecl::wgpu::WgpuRuntime;
// use std::env;

// use yumon_pet::brain::{
//     decoder_model::YumonDecBrain,
//     model::YumonBrain,
//     samples::TrainingStage,
// };

// fn wrap_prompt(stage: TrainingStage, prompt: &str) -> String {
//     if stage == TrainingStage::Structured {
//         serde_json::to_string_pretty(&serde_json::json!({
//             "memories": Vec::<serde_json::Value>::new(),
//             "message": prompt,
//         }))
//         .unwrap()
//     } else {
//         prompt.to_string()
//     }
// }

// fn main() -> Result<()> {
//     let args: Vec<String> = env::args().collect();
//     if args.len() < 4 {
//         eprintln!("usage: headless_compare <dec|enc> <checkpoint_dir> <prompt>");
//         std::process::exit(1);
//     }
//     let arch = &args[1];
//     let checkpoint = &args[2];
//     let prompt = &args[3];

//     type B = Wgpu;
//     let device = Default::default();

//     match arch.as_str() {
//         "dec" => {
//             let (model, tokenizer, config) = YumonDecBrain::<B>::load(checkpoint, &device)?;
//             let wrapped = wrap_prompt(config.training_stage, prompt);
//             let result = model.generate_unmasked_parsed::<WgpuRuntime>(
//                 &tokenizer,
//                 &wrapped,
//                 config.max_seq_len,
//                 &device,
//             );
//             println!("REPLY: {}", result.reply);
//             println!("RAW: {}", result.raw_output);
//         }
//         "enc" => {
//             let (model, tokenizer, config) = YumonBrain::<B>::load(checkpoint, &device)?;
//             let wrapped = wrap_prompt(config.training_stage, prompt);
//             let result = model.generate_unmasked_parsed::<WgpuRuntime>(
//                 &tokenizer,
//                 &wrapped,
//                 config.max_seq_len,
//                 &device,
//             );
//             println!("REPLY: {}", result.reply);
//             println!("RAW: {}", result.raw_output);
//         }
//         other => {
//             eprintln!("unknown arch: {other}, expected dec|enc");
//             std::process::exit(1);
//         }
//     }

//     Ok(())
// }

pub fn main() {
    println!("hi");
}