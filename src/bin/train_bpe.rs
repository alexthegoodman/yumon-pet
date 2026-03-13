use yumon_pet::brain::{bpe::{self, BpeTokenizer}, wiki::load_wiki_sentences};

pub fn main() {
    let wiki_xml = "data/simplewiki-latest-pages-articles.xml";
    let sentences = load_wiki_sentences(wiki_xml, 25_000);
    let sentences = sentences.as_ref().expect("Couldn't get sentences");
    let bpe = BpeTokenizer::train(&sentences, 4096);
    let bpe = bpe.as_ref().expect("Couldn't train bpe");
    bpe.save("yumon_bpe").as_ref().expect("Couldn't save bpe");
}