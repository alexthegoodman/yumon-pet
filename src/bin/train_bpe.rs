use yumon_pet::brain::{bpe::{self, BpeTokenizer}, mdx::{load_csv_quotes, load_dictionary_sentences, load_mdx_sentences}, wiki::load_wiki_sentences};

pub fn main() {
    let wiki_xml = "data/simplewiki-latest-pages-articles.xml";
    
    // let sentences = load_wiki_sentences(wiki_xml, 25_000);

    // ── Load + tokenize wiki corpus ───────────────────────────────────────────
    let mut sentences = Vec::new();

    let wiki_sentences: Result<Vec<String>, anyhow::Error> = load_wiki_sentences(wiki_xml, 5_000);
    let wiki_sentences = wiki_sentences.as_ref().expect("Couldn't get wiki_sentences");

    for (i, sent) in sentences.iter().enumerate() {
        if (i < 12) {
            println!("WIKI: {:?}", sent);
        }
    }

    let mdx_sentences = load_mdx_sentences("data/(poems)/");
    let mdx_sentences = mdx_sentences.as_ref().expect("Couldn't get mdx_sentences");

    for (i, sent) in mdx_sentences.iter().enumerate() {
        if (i < 12) {
            println!("MDX: {:?}", sent);
        }
    }

    let quote_sentences = load_csv_quotes("data/quotes.csv");
    let quote_sentences = quote_sentences.as_ref().expect("Couldn't get quote_sentences");

    for (i, sent) in quote_sentences.iter().enumerate() {
        if (i < 12) {
            println!("QUOTE: {:?}", sent);
        }
    }

    let dict_sentences = load_dictionary_sentences("data/Dictionary/Oxford/Oxford_English_Dictionary.txt");
    let dict_sentences = dict_sentences.as_ref().expect("Couldn't get dict_sentences");

    for (i, sent) in dict_sentences.iter().enumerate() {
        if (i < 12) {
            println!("DICT: {:?}", sent);
        }
    }
    
    sentences.extend(wiki_sentences);
    sentences.extend(mdx_sentences);
    sentences.extend(quote_sentences);
    sentences.extend(dict_sentences);
        
    let bpe = BpeTokenizer::train(
        sentences, 
        4096 // max size on my igpu
        // 8192
        // 16384
    );
    let bpe = bpe.as_ref().expect("Couldn't train bpe");

    bpe.save("yumon_bpe").as_ref().expect("Couldn't save bpe");
}