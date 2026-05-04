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
        // .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/arena_extract.txt",   FileKind::Chats, Some(200_000))
        // .add("data/Dictionary/Oxford/Oxford_English_Dictionary.txt",   FileKind::SpecificDict, Some(50_000))
        .add("data/bible_bbe.csv", FileKind::BibleCsv, Some(200_000))
        .add("data/creative_stories.txt", FileKind::Txt, Some(200_000))
        .add("data/The-Office-Lines-V4.csv",   FileKind::DialogueCsv, Some(200_000))
        .add("data/friends_all_episodes_clean.csv",   FileKind::FriendsCsv, Some(200_000))
        // .add("data/distillchatv1.csv",   FileKind::DistillChat, Some(50_000))
        // .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        .add("archive/ov_chats.txt", FileKind::Chats, None)
        // .add("archive/you_chats.txt", FileKind::Chats, None)
        // .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        .add("archive/you_chats.txt", FileKind::Chats, None)
        // .add("archive/clean_chats.txt", FileKind::Chats, None)
        // .add("archive/clean_chats.txt", FileKind::Chats, None)
        .add("archive/clean_chats.txt", FileKind::Chats, None)
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
        sentences, 
        4096 // max size on my igpu at 128 batch size
        // 8192
        // 16384 // doesnt seem to help at all (can do at 16 batch size)
    );
    let bpe = bpe.as_ref().expect("Couldn't train bpe");

    bpe.save("yumon_bpe").as_ref().expect("Couldn't save bpe");

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