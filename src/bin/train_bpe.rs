use yumon_pet::{brain::{PAD_TOKEN, bpe::{self, BpeTokenizer, TokenizerKind}, loader::{DataLoader, FileKind}, mdx::{load_arena_chats, load_dictionary_sentences, load_handcrafted_chats, load_qa_pairs, load_specific_dict_sentences, load_txt_sentences}, samples::TrainingStage, train::{build_keyword_index, build_label_keywords}}, vision::CIFAR_CLASSES};


#[cfg(target_os = "windows")]
use yumon_pet::{brain::{mdx::{load_handcrafted_sentences, load_mdx_sentences}}};

pub fn main() {
    // let all_words = load_csv_words("archive/word_counts.csv");
    // let all_words =  all_words.as_ref().expect("Couldn't get words");
    
    // let dict_sentences = load_specific_dict_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt", all_words);
    // let dict_sentences = dict_sentences.as_ref().expect("Couldn't get dict_sentences");

    // let wiki_xml = "data/simplewiki-latest-pages-articles.xml";
    
    // let sentences = load_wiki_sentences(wiki_xml, 25_000);

    // ── Load + tokenize wiki corpus ───────────────────────────────────────────
    // let mut sentences = Vec::new();

    // let wiki_sentences: Result<Vec<String>, anyhow::Error> = load_wiki_sentences(wiki_xml, 50_000, 1);
    // let wiki_sentences = wiki_sentences.as_ref().expect("Couldn't get wiki_sentences");

    // for (i, sent) in sentences.iter().enumerate() {
    //     if (i < 12) {
    //         println!("WIKI: {:?}", sent);
    //     }
    // }

    // let wiki_sentences: Result<Vec<String>, anyhow::Error> = load_txt_sentences("data/wiki_extract.txt");
    // let wiki_sentences = wiki_sentences.as_ref().expect("Couldn't get wiki_sentences");

    // for (i, sent) in sentences.iter().enumerate() {
    //     if (i < 12) {
    //         println!("WIKI: {:?}", sent);
    //     }
    // }

    // let mdx_sentences = load_mdx_sentences("data/(poems)/");
    // let mdx_sentences = mdx_sentences.as_ref().expect("Couldn't get mdx_sentences");

    // for (i, sent) in mdx_sentences.iter().enumerate() {
    //     if (i < 12) {
    //         println!("MDX: {:?}", sent);
    //     }
    // }

    // // let quote_sentences = load_csv_quotes("data/quotes.csv");
    // // let quote_sentences = quote_sentences.as_ref().expect("Couldn't get quote_sentences");

    // // for (i, sent) in quote_sentences.iter().enumerate() {
    // //     if (i < 12) {
    // //         println!("QUOTE: {:?}", sent);
    // //     }
    // // }

    // // let dict_sentences = load_dictionary_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt");
    // // let dict_sentences = dict_sentences.as_ref().expect("Couldn't get dict_sentences");

    // // for (i, sent) in dict_sentences.iter().enumerate() {
    // //     if (i < 12) {
    // //         println!("DICT: {:?}", sent);
    // //     }
    // // }

    // // let qna_sentences = load_csv_qna("data/AI.csv");
    // // let qna_sentences = qna_sentences.as_ref().expect("Couldn't get qna_sentences");

    // // for (i, sent) in qna_sentences.iter().enumerate() {
    // //     if (i < 12) {
    // //         println!("Q&A: {:?}", sent);
    // //     }
    // // }

    // let bible_verses = load_csv_bible("data/bible_bbe.csv");
    // let bible_verses = bible_verses.as_ref().expect("Couldn't get bible_verses");

    // for (i, sent) in bible_verses.iter().enumerate() {
    //     if (i < 12) {
    //         println!("Verse: {:?}", sent);
    //     }
    // }

    // let handcrafted = load_handcrafted_sentences("archive/handcrafted.txt");
    // let handcrafted = handcrafted.as_ref().expect("Couldn't get handcrafted");

    // for (i, sent) in handcrafted.iter().enumerate() {
    //     if (i < 12) {
    //         println!("handcrafted: {:?}", sent);
    //     }
    // }

    // // let notions = load_notion_sentences("data/notion/");
    // // let notions = notions.as_ref().expect("Couldn't get handcrafted");

    // // for (i, sent) in notions.iter().enumerate() {
    // //     if (i < 12) {
    // //         println!("notion: {:?}", sent);
    // //     }
    // // }

    // // let txt = load_txt_sentences("data/creative_stories.txt");
    // // let txt = txt.as_ref().expect("Couldn't get txt");

    // // for (i, sent) in txt.iter().enumerate() {
    // //     if (i < 12) {
    // //         println!("txt: {:?}", sent);
    // //     }
    // // }

    // let mut my_qna = load_qa_pairs("archive/handcrafted_pairs.txt");
    // let my_qna = my_qna.as_ref().expect("Couldn't get handcrafted");

    // let mut handcrafted_qa = Vec::new();

    // for (i, sent) in my_qna.iter().enumerate() {
    //     handcrafted_qa.push(sent.0.clone() + " " + &sent.1.clone());

    //     if (i < 12) {
    //         println!("qa: {:?}", sent);
    //     }
    // }

    // let mut my_qna2 = load_qa_pairs("data/qa_journal.txt");
    // let my_qna2 = my_qna2.as_ref().expect("Couldn't get handcrafted");

    // let mut handcrafted_qa2 = Vec::new();

    // for (i, sent) in my_qna2.iter().enumerate() {
    //     handcrafted_qa2.push(sent.0.clone() + " " + &sent.1.clone());

    //     if (i < 12) {
    //         println!("qa: {:?}", sent);
    //     }
    // }

    // let ebooks = load_pdf_ebook_sentences("data/ebooks/survival_handbook.pdf");
    // let ebooks = ebooks.as_ref().expect("Couldn't get handcrafted");

    // for (i, sent) in ebooks.iter().enumerate() {
    //     if (i < 12) {
    //         println!("ebook: {:?}", sent);
    //     }
    // }

    // let ebooks = load_pdf_ebook_sentences("data/stephen_hawking_a_brief_history_of_time.pdf");
    // let ebooks = ebooks.as_ref().expect("Couldn't get handcrafted");

    // for (i, sent) in ebooks.iter().enumerate() {
    //     if (i < 12) {
    //         println!("ebook: {:?}", sent);
    //     }
    // }

    // let mut chats = load_handcrafted_chats("archive/handcrafted_pairs.txt");
    // let chats = chats.as_ref().expect("Couldn't get handcrafted");

    // let mut chats_combined = Vec::new();

    // for (i, block) in chats.blocks.iter().enumerate() {
    //     for (x, memory) in block.memories.iter().enumerate() {
    //         chats_combined.push(&memory.human);
    //         chats_combined.push(&memory.bot);

    //         if (i < 12) {
    //             println!("qa human: {:?}", memory.human);
    //             println!("qa bot: {:?}", memory.bot);
    //         }
    //     }
    // }

    // let mut chats2 = load_handcrafted_chats("archive/ov_chats.txt");
    // let chats2 = chats2.as_ref().expect("Couldn't get ov chats");

    // let mut chats_combined2 = Vec::new();

    // for (i, block) in chats2.blocks.iter().enumerate() {
    //     for (x, memory) in block.memories.iter().enumerate() {
    //         chats_combined2.push(&memory.human);
    //         chats_combined2.push(&memory.bot);

    //         if (i < 12) {
    //             println!("qa human: {:?}", memory.human);
    //             println!("qa bot: {:?}", memory.bot);
    //         }
    //     }
    // }

    // let mut chats3 = load_arena_chats("data/chatbot_arena_conversations.json");
    // let chats3 = chats3.as_ref().expect("Couldn't get arena chats");

    // let mut chats_combined3 = Vec::new();

    // for (i, block) in chats3.blocks.iter().enumerate() {
    //     for (x, memory) in block.memories.iter().enumerate() {
    //         chats_combined3.push(&memory.human);
    //         chats_combined3.push(&memory.bot);

    //         if (i < 12) {
    //             println!("arena human: {:?}", memory.human);
    //             println!("arena bot: {:?}", memory.bot);
    //         }
    //     }
    // }

    let mut loader = DataLoader::new(TrainingStage::Language);
    // match stage {
    //     TrainingStage::Language => {
    //         loader = loader
    //             .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
    //     }
    //     TrainingStage::Structured => {
    //         loader = loader
    //             .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
    //     }
    // }

    loader = loader
        // // .add("data/chatbot_arena_conversations.json",   FileKind::JsonChats, None)
        .add("data/ideas.txt",   FileKind::TxtLines, Some(5_000))
        // .add("archive/arena_extract.txt",   FileKind::Chats, Some(25_000))
        // .add("data/distillchatv1.csv",   FileKind::DistillChat, Some(25_000))
        // .add("data/wiki_extract.txt",   FileKind::Txt, Some(250_000))
        // .add("data/bible_bbe.csv", FileKind::BibleCsv, None)
        // .add("data/bible_asv.csv", FileKind::BibleCsv, None)
        // LLM-generated Q&A pairs from src/bin/gen_synthetic_data.rs — proper
        // message/reply splits instead of BibleCsv's arbitrary mid-sentence cuts.
        .add("data/synthetic/bible.txt", FileKind::Chats, None)
        .add("data/synthetic/business.txt", FileKind::Chats, None)
        .add("data/synthetic/universe.txt", FileKind::Chats, None)
        // .add("data/creative_stories.txt", FileKind::Txt, Some(50_000)) // good but gets split
        // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, Some(50_000))
        // .add("archive/handcrafted_pairs.txt", FileKind::Chats, None);
        // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("data/The-Office-Lines-V4.csv",   FileKind::DialogueCsv, Some(25_000))
        // .add("data/friends_all_episodes_clean.csv",   FileKind::FriendsCsv, Some(25_000))
        // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("archive/you_chats.txt", FileKind::Chats, None)
        // .add("archive/you_chats.txt", FileKind::Chats, None)
        // .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        // .add("archive/clean_chats.txt", FileKind::Chats, None)
        // .add("archive/clean_chats.txt", FileKind::Chats, None)
        // .add("archive/clean_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None);

        // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("data/chatbot_arena_conversations.json",   FileKind::JsonChats, None)
        // .add("data/wiki_extract.txt",   FileKind::Txt, None)
        // .add("data/creative_stories.txt", FileKind::Txt, None)
        // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, None)
        // .add("archive/handcrafted_pairs.txt", FileKind::Chats, None)
        // .add(vec![
        //         "data/ebooks/faa-h-8083-25c.pdf".to_string(),
        //         "data/ebooks/algor_intro.pdf".to_string(),
        //         "data/ebooks/intro_engineer.pdf".to_string(),
        //         "data/ebooks/meap.pdf".to_string(),
        //         "data/ebooks/missiles.pdf".to_string(),
        //         "data/ebooks/os_concepts.pdf".to_string(),
        //         "data/ebooks/real-time-embedded.pdf".to_string(),
        //         "data/ebooks/riscv.pdf".to_string(),
        //         "data/ebooks/rtos.pdf".to_string(),
        //         "data/ebooks/stephen_hawking_a_brief_history_of_time.pdf".to_string(),
        //     ].join(", "), 
        //     FileKind::PDF, 
        //     None
        // );

    let sentences: Vec<String> = loader.total_limit(400_000).seed(4815162342).load_sentences().expect("Couldn't get sentences");
    let sentences: Vec<&String> = sentences.iter().collect();

    let mut x = 0;
    for sent in &sentences {
        println!("Sentence: {:?}", sent);
        x = x + 1;
        if x > 25 {
            break;
        }
    }
    
    // sentences.extend(wiki_sentences);
    // sentences.extend(mdx_sentences);
    // // sentences.extend(qna_sentences);
    // sentences.extend(bible_verses);
    // sentences.extend(handcrafted);
    // sentences.extend(&handcrafted_qa);
    // sentences.extend(&handcrafted_qa2);
    // sentences.extend(quote_sentences);
    // sentences.extend(dict_sentences);
    // sentences.extend(notions);
    // // sentences.extend(ebooks);
    // sentences.extend(txt);
    // sentences.extend(ebooks);
    // sentences.extend(chats_combined);
    // // sentences.extend(dict_sentences);
    // sentences.extend(chats_combined2);
    // sentences.extend(chats_combined3);
    // sentences.extend(wiki_sentences);
        
    let bpe = BpeTokenizer::train(
        sentences.clone(),
        // 1024 // puts more effort into spelling
        // 4096 // max size on my igpu at 128 batch size
        // 8192
        16384 // doesnt seem to help at all (can do at 16 batch size)
    );
    let bpe = bpe.as_ref().expect("Couldn't train bpe");

    bpe.save("yumon_bpe_16k").as_ref().expect("Couldn't save bpe");

    // // ── Diagnostic: avg sample length in chars/tokens ───────────────────────
    // // Helps size max_seq_len. These are raw pre-JSON sentences (individual
    // // human/bot turns) — budget extra on top for the Structured stage's
    // // "{memories, message}" / "{action, emotion, reply}" JSON wrapper.
    // {
    //     let n = sentences.len();
    //     let mut total_chars  = 0usize;
    //     let mut total_tokens = 0usize;
    //     let mut max_chars    = 0usize;
    //     let mut max_tokens   = 0usize;

    //     for sent in &sentences {
    //         let chars  = sent.chars().count();
    //         let tokens = bpe.encode_raw(sent).map(|t| t.len()).unwrap_or(0);
    //         total_chars  += chars;
    //         total_tokens += tokens;
    //         max_chars  = max_chars.max(chars);
    //         max_tokens = max_tokens.max(tokens);
    //     }

    //     let avg_chars  = total_chars as f64 / n.max(1) as f64;
    //     let avg_tokens = total_tokens as f64 / n.max(1) as f64;

    //     println!("\n📏 Corpus length diagnostic ({n} samples, per human/bot turn)");
    //     println!("   avg chars:   {avg_chars:.1}  (max {max_chars})");
    //     println!("   avg tokens:  {avg_tokens:.1}  (max {max_tokens})");
    //     println!("   chars/token: {:.2}", avg_chars / avg_tokens.max(1.0));
    //     println!("   +20% JSON overhead estimate: ~{:.0} tokens/turn", avg_tokens * 1.2);
    //     println!("   packed pair estimate (input+sep+reply, +20%): ~{:.0} tokens", avg_tokens * 2.0 * 1.2);
    // }

    // // ── Diagnostic: Structured-stage packed sequence length, growing memories ──
    // // The `memories` array in the input JSON accumulates every prior turn in a
    // // chat block, so later turns carry far more input than a lone human/bot
    // // pair does. This walks the same chat corpora with the real memory-growth
    // // logic from samples.rs to measure the actual packed (input+sep+target)
    // // sequence the DecoderOnly path sees, worst case included.
    // {
    //     use rand::thread_rng;
    //     use yumon_pet::brain::samples::generate_training_sample;
    //     use yumon_pet::brain::sentiment::EmotionAnalyzer;

    //     let chat_files = [
    //         "data/synthetic/bible.txt",
    //         "data/synthetic/business.txt",
    //         "data/synthetic/universe.txt",
    //         "archive/ov_chats.txt",
    //         "archive/you_chats.txt",
    //         "archive/clean_chats.txt",
    //     ];

    //     let analyzer = EmotionAnalyzer::new();
    //     let mut rng = thread_rng();
    //     let sep_tokens_len = bpe.encode_raw("\n---\n").map(|t| t.len()).unwrap_or(0);

    //     let mut n_blocks = 0usize;
    //     let mut n_turns = 0usize;
    //     let mut total_memories = 0usize;
    //     let mut max_memories = 0usize;

    //     let mut in_total_tokens = 0usize;
    //     let mut in_max_tokens = 0usize;
    //     let mut tgt_total_tokens = 0usize;
    //     let mut tgt_max_tokens = 0usize;
    //     let mut combined_total_tokens = 0usize;
    //     let mut combined_max_tokens = 0usize;

    //     // Bucketed by exact prior-memory count `i` (0, 1, 2, …) so we can read
    //     // off a marginal per-memory token cost instead of one blended average.
    //     let mut by_memcount: std::collections::BTreeMap<usize, (usize, usize, usize)> = std::collections::BTreeMap::new();
    //     // (count, sum_in_tokens, max_in_tokens)

    //     for path in chat_files {
    //         let chats = match load_handcrafted_chats(path) {
    //             Ok(c) => c,
    //             Err(e) => { println!("   ⚠️  skipping {path}: {e}"); continue; }
    //         };
    //         for block in &chats.blocks {
    //             n_blocks += 1;
    //             max_memories = max_memories.max(block.memories.len());

    //             for (i, memory) in block.memories.iter().enumerate() {
    //                 n_turns += 1;
    //                 total_memories += i;

    //                 let prior_memories: Vec<serde_json::Value> = block.memories[..i]
    //                     .iter()
    //                     .map(|m| serde_json::json!({ "human": m.human, "yumon": m.bot }))
    //                     .collect();

    //                 let input_json = serde_json::to_string_pretty(&serde_json::json!({
    //                     "memories": prior_memories,
    //                     "message":  memory.human,
    //                 })).unwrap();

    //                 let tsample = generate_training_sample(&mut rng);
    //                 let sentiment = sentiment::analyze(memory.bot.to_string());
    //                 let emotion = analyzer.analyze(&memory.bot, &sentiment);

    //                 let target_json = serde_json::to_string_pretty(&serde_json::json!({
    //                     "action":  tsample.action.as_str(),
    //                     "emotion": emotion,
    //                     "reply":   memory.bot,
    //                 })).unwrap();

    //                 let in_tokens  = bpe.encode_raw(&input_json).map(|t| t.len()).unwrap_or(0);
    //                 let tgt_tokens = bpe.encode_raw(&target_json).map(|t| t.len()).unwrap_or(0);
    //                 let combined   = in_tokens + sep_tokens_len + tgt_tokens;

    //                 in_total_tokens += in_tokens;
    //                 in_max_tokens = in_max_tokens.max(in_tokens);
    //                 tgt_total_tokens += tgt_tokens;
    //                 tgt_max_tokens = tgt_max_tokens.max(tgt_tokens);
    //                 combined_total_tokens += combined;
    //                 combined_max_tokens = combined_max_tokens.max(combined);

    //                 let bucket = by_memcount.entry(i).or_insert((0, 0, 0));
    //                 bucket.0 += 1;
    //                 bucket.1 += in_tokens;
    //                 bucket.2 = bucket.2.max(in_tokens);
    //             }
    //         }
    //     }

    //     let avg = |t: usize| t as f64 / n_turns.max(1) as f64;

    //     println!("\n📏 Structured packed-sequence diagnostic ({n_blocks} chat blocks, {n_turns} turns, incl. growing memories)");
    //     println!("   avg memories/turn: {:.1}  (max chain depth: {max_memories})", total_memories as f64 / n_turns.max(1) as f64);
    //     println!("   input  (memories+message)     — avg tokens: {:.1}  (max {in_max_tokens})", avg(in_total_tokens));
    //     println!("   target (action+emotion+reply) — avg tokens: {:.1}  (max {tgt_max_tokens})", avg(tgt_total_tokens));
    //     println!("   combined packed (input+sep+target) — avg tokens: {:.1}  (max {combined_max_tokens})", avg(combined_total_tokens));
    //     println!("   +20% headroom on worst case: ~{:.0} tokens  <- size max_seq_len against this", combined_max_tokens as f64 * 1.2);

    //     println!("\n📏 Input tokens by prior-memory count (marginal cost per memory)");
    //     println!("   {:>8}  {:>7}  {:>12}  {:>12}", "memories", "turns", "avg_in_tok", "max_in_tok");
    //     for (mem_count, (count, sum_in, max_in)) in &by_memcount {
    //         println!("   {:>8}  {:>7}  {:>12.1}  {:>12}", mem_count, count, *sum_in as f64 / *count as f64, max_in);
    //     }

    //     // Linear fit (avg_in_tokens ~ base + slope * mem_count) over the observed
    //     // buckets, so we can extrapolate to memory counts deeper than anything
    //     // actually present in this corpus (max chain depth here: {max_memories}).
    //     if by_memcount.len() >= 2 {
    //         let points: Vec<(f64, f64)> = by_memcount.iter()
    //             .map(|(&k, &(count, sum, _))| (k as f64, sum as f64 / count as f64))
    //             .collect();
    //         let n = points.len() as f64;
    //         let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    //         let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    //         let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    //         let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();
    //         let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    //         let intercept = (sum_y - slope * sum_x) / n;

    //         let avg_target = tgt_total_tokens as f64 / n_turns.max(1) as f64;

    //         println!("\n📏 Extrapolated packed-sequence size for a chosen memory budget");
    //         println!("   per-memory marginal cost: ~{slope:.1} tokens  (base message w/ 0 memories: ~{intercept:.1} tokens)");
    //         for want in [1usize, 2, 3, 5, 8, 10, 15, 20] {
    //             let est_in = intercept + slope * want as f64;
    //             let est_combined = est_in + sep_tokens_len as f64 + avg_target;
    //             println!(
    //                 "   {:>2} memories → est. input ~{:.0} tok, combined ~{:.0} tok, +20% headroom → max_seq_len ~{:.0}",
    //                 want, est_in, est_combined, est_combined * 1.2
    //             );
    //         }
    //     }
    // }

    // // ----------------------
    // // ── Check samples sizes ─────────────────────────────────────────────
    // // ----------------------
    // let label_keywords   = build_label_keywords();
    // let keyword_index    = build_keyword_index(&label_keywords);
    // println!("🏷  CIFAR-100 keyword index: {} unique keywords across {} classes",
    //          keyword_index.len(), CIFAR_CLASSES);

    // let tokenizer = TokenizerKind::Bpe(BpeTokenizer::load("yumon_bpe").expect("Couldn't get bpe"));

    // let training_stage = TrainingStage::Language;
 
    // let training_samples = DataLoader::new(training_stage)
    //     .add("archive/ov_chats.txt", FileKind::Chats, None)
    //     .add("archive/handcrafted_pairs.txt",   FileKind::Chats, None)
    //     // .add("archive/ov_chats.txt",   FileKind::Chats, None)
    //     // .add("data/chatbot_arena_conversations.json",   FileKind::JsonChats, None)
    //     // .add("data/wiki_extract.txt",   FileKind::Txt, Some(200_000))
    //     // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, None)
    //     .total_limit(4096)
    //     .seed(4815162342)
    //     .load(&tokenizer, &keyword_index, 320).expect("Couldn't get samples");
    
    // println!("language training samples: {}", training_samples.len());
    
    // // debug print — first samples
    // for (i, sample) in training_samples.iter().enumerate() {
    //     if i >= 50 { break; }
    //     println!("language input_len:     {}", sample.input_ids.iter().filter(|&&t| t != PAD_TOKEN).count());
    //     println!("language target_active: {}", sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count());
    // }

    // let training_stage = TrainingStage::Structured;
 
    // let training_samples = DataLoader::new(training_stage)
    //     .add("archive/ov_chats.txt", FileKind::Chats, None)
    //     .add("archive/handcrafted_pairs.txt",   FileKind::Chats, None)
    //     // .add("archive/ov_chats.txt",   FileKind::Chats, None)
    //     // .add("data/chatbot_arena_conversations.json",   FileKind::JsonChats, None)
    //     // .add("data/wiki_extract.txt",   FileKind::Txt, Some(200_000))
    //     // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, None)
    //     .total_limit(4096)
    //     .seed(4815162342)
    //     .load(&tokenizer, &keyword_index, 320).expect("Couldn't get samples");
    
    // println!("structured training samples: {}", training_samples.len());
    
    // // debug print — first samples
    // for (i, sample) in training_samples.iter().enumerate() {
    //     if i >= 50 { break; }
    //     println!("structured input_len:     {}", sample.input_ids.iter().filter(|&&t| t != PAD_TOKEN).count());
    //     println!("structured target_active: {}", sample.target_labels.iter().filter(|&&t| t != PAD_TOKEN).count());
    // }
}