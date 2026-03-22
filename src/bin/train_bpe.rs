use yumon_pet::brain::{bpe::{self, BpeTokenizer}, mdx::{load_csv_bible, load_csv_qna, load_csv_quotes, load_dictionary_sentences, load_handcrafted_sentences, load_mdx_sentences, load_notion_sentences}, pdf::load_pdf_ebook_sentences, wiki::load_wiki_sentences};

pub fn main() {
    let wiki_xml = "data/simplewiki-latest-pages-articles.xml";
    
    // let sentences = load_wiki_sentences(wiki_xml, 25_000);

    // ── Load + tokenize wiki corpus ───────────────────────────────────────────
    let mut sentences = Vec::new();

    // let wiki_sentences: Result<Vec<String>, anyhow::Error> = load_wiki_sentences(wiki_xml, 5_000);
    // let wiki_sentences = wiki_sentences.as_ref().expect("Couldn't get wiki_sentences");

    // for (i, sent) in sentences.iter().enumerate() {
    //     if (i < 12) {
    //         println!("WIKI: {:?}", sent);
    //     }
    // }

    let mdx_sentences = load_mdx_sentences("data/(poems)/");
    let mdx_sentences = mdx_sentences.as_ref().expect("Couldn't get mdx_sentences");

    for (i, sent) in mdx_sentences.iter().enumerate() {
        if (i < 12) {
            println!("MDX: {:?}", sent);
        }
    }

    // let quote_sentences = load_csv_quotes("data/quotes.csv");
    // let quote_sentences = quote_sentences.as_ref().expect("Couldn't get quote_sentences");

    // for (i, sent) in quote_sentences.iter().enumerate() {
    //     if (i < 12) {
    //         println!("QUOTE: {:?}", sent);
    //     }
    // }

    // let dict_sentences = load_dictionary_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt");
    // let dict_sentences = dict_sentences.as_ref().expect("Couldn't get dict_sentences");

    // for (i, sent) in dict_sentences.iter().enumerate() {
    //     if (i < 12) {
    //         println!("DICT: {:?}", sent);
    //     }
    // }

    // let qna_sentences = load_csv_qna("data/AI.csv");
    // let qna_sentences = qna_sentences.as_ref().expect("Couldn't get qna_sentences");

    // for (i, sent) in qna_sentences.iter().enumerate() {
    //     if (i < 12) {
    //         println!("Q&A: {:?}", sent);
    //     }
    // }

    let bible_verses = load_csv_bible("data/bible_bbe.csv");
    let bible_verses = bible_verses.as_ref().expect("Couldn't get bible_verses");

    for (i, sent) in bible_verses.iter().enumerate() {
        if (i < 12) {
            println!("Verse: {:?}", sent);
        }
    }

    let handcrafted = load_handcrafted_sentences("data/handcrafted.txt");
    let handcrafted = handcrafted.as_ref().expect("Couldn't get handcrafted");

    for (i, sent) in handcrafted.iter().enumerate() {
        if (i < 12) {
            println!("handcrafted: {:?}", sent);
        }
    }

    let notions = load_notion_sentences("data/notion/");
    let notions = notions.as_ref().expect("Couldn't get handcrafted");

    for (i, sent) in notions.iter().enumerate() {
        if (i < 12) {
            println!("notion: {:?}", sent);
        }
    }

    // let ebooks = load_pdf_ebook_sentences("data/algorithms_ebook.pdf");
    // let ebooks = ebooks.as_ref().expect("Couldn't get handcrafted");

    // for (i, sent) in ebooks.iter().enumerate() {
    //     if (i < 12) {
    //         println!("ebook: {:?}", sent);
    //     }
    // }

    let ebooks = load_pdf_ebook_sentences("data/stephen_hawking_a_brief_history_of_time.pdf");
    let ebooks = ebooks.as_ref().expect("Couldn't get handcrafted");

    for (i, sent) in ebooks.iter().enumerate() {
        if (i < 12) {
            println!("ebook: {:?}", sent);
        }
    }
    
    // sentences.extend(wiki_sentences);
    sentences.extend(mdx_sentences);
    // sentences.extend(qna_sentences);
    sentences.extend(bible_verses);
    sentences.extend(handcrafted);
    // sentences.extend(quote_sentences);
    // sentences.extend(dict_sentences);
    sentences.extend(notions);
    sentences.extend(ebooks);
        
    let bpe = BpeTokenizer::train(
        sentences, 
        4096 // max size on my igpu at 128 batch size
        // 8192
        // 16384 // doesnt seem to help at all (can do at 16 batch size)
    );
    let bpe = bpe.as_ref().expect("Couldn't train bpe");

    bpe.save("yumon_bpe").as_ref().expect("Couldn't save bpe");
}