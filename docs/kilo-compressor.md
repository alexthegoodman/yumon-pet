# Architecture & Specification: Kilo Tokenizer / Vocab Mapping Model

**Epic:** Kilo — Statistical Intent Encoding & Bounded Vocabulary Mapping  
**Component:** `src/brain/kilo_vocab.rs` (Tokenizer / Vocab Mapping Model)  
**Pipeline Driver:** `src/bin/cache_yumon_talk.rs` (Dataset Compressor & Sample Cache)  
**Reconstruction Decoder:** `src/brain/decompressor.rs` (`NGramDecompressor`)  
**Priority:** High  
**Status:** Approved / In Progress  

---

## 1. Executive Summary & Architectural Roles

The Kilo language system operates on a strictly bounded **4,096-token vocabulary**. To achieve compact representation and prevent vocabulary explosion without compromising semantic fidelity, the system divides responsibilities across three dedicated components:

```
+-----------------------------------------------------------------------------------+
| 1. Kilo Tokenizer / Vocab Mapping Model (`src/brain/kilo_vocab.rs`)              |
|    - Learns empirical vocabulary (words + phrases) from corpora without thesaurus |
|    - Maps natural English into 4,096 bounded tokens with contextual awareness     |
|    - Preserves distinct meanings (e.g. 'world' != 'universe')                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                    Tokens (4,096 IDs)
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. Dataset Compressor & Pipeline (`src/bin/cache_yumon_talk.rs`)                 |
|    - Streams raw English dataset samples (dialogue pairs, articles, stories)      |
|    - Drives tokenization, entity context tracking, and roundtrip filtering       |
|    - Serializes into `yumon_talk_samples.bin` for MoE / Transformer training     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                    Trained Model
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. Statistical N-Gram Decompressor (`src/brain/decompressor.rs`)                  |
|    - Contextual N-gram LM (trigrams/4-grams with Stupid Backoff)                  |
|    - Beam search decoding over canonical token allowable word sets                |
|    - Expands compact representations back into fluent, natural English            |
+-----------------------------------------------------------------------------------+
```

- **The Compressor:** `src/bin/cache_yumon_talk.rs` acts as the overall dataset compressor and pipeline runner. It processes gigabytes of text, manages concurrency with Rayon, extracts contextual entities, and serializes training tensors.
- **The Tokenizer / Vocab Mapping Model:** `src/brain/kilo_vocab.rs` is the core semantic mapping model. It is responsible for selecting the bounded vocabulary from data and encoding words/phrases into canonical token IDs such that the decompressor can reconstruct the intended meaning in context.
- **The Decompressor:** `src/brain/decompressor.rs` (`NGramDecompressor`) reconstructs rich, uncompressed English from token sequences using contextual n-gram probabilities.

---

## 2. Core Philosophy: Meaning in Context vs. Hardcoded Thesaurus

Prior vocabulary generation relied on static thesaurus tables (`curated_synonyms.json` and WordNet via the `thesaurus` crate). While this provided initial deduplication, it suffered from fundamental limitations:

1. **Semantic Erasure:** Words with distinct connotations or domain meanings were conflated (e.g., merging `world` into `universe`, or `huge` into `big`).
2. **Context Blindness:** A synonym that works in one idiom often produces awkward or ungrammatical collocations in another.
3. **Decompression Drift:** Naive substitution without regard to decompressor n-gram priors causes the decompressor to reconstruct arbitrary synonyms.

**The Kilo Tokenizer / Vocab Mapping Model replaces static thesaurus collapsing with two foundational principles:**

- **Empirical Distinctness:** High-frequency, high-information words retain distinct vocabulary entries in the 4,096 budget. Words are never merged simply because a thesaurus lists them as technical synonyms.
- **Decompressor-Aware Contextual Mapping (Invertibility):** When mapping an input expression into a canonical representation, the mapping model selects tokens that the decompressor can reliably expand back to the intended meaning in that exact n-gram context:
  $$\text{token}^* = \arg\max_c P(\text{target\_meaning} \mid c, \text{context})$$

---

## 3. Token Budget Breakdown (Exactly 4,096 Discrete Tokens)

| Category | Token Count | Description & Examples |
| :--- | :---: | :--- |
| **Special Tokens** | 4 | `<pad>` (0), `<s>` (1), `</s>` (2), `<unk>` (3) |
| **Punctuation & Layout** | 24 | `.`, `,`, `!`, `?`, `;`, `:`, `(`, `)`, `[`, `]`, `"`, `'`, `-`, `/`, `\n`, etc. |
| **Entity & Slot Placeholders** | 84 | Contextually preserved literals: |
| ↳ *Named Entities* | 48 | `<NAME_0>`..`<NAME_47>` (capitalized proper nouns, entities, brands) |
| ↳ *Numerics* | 20 | `<NUM_0>`..`<NUM_19>` (integers, floats, quantities) |
| ↳ *Template Slots* | 16 | `<SLOT_0>`..`<SLOT_15>` (rare open-class terms, dynamic variables) |
| **Learned Vocabulary** | 3,984 | Top empirical unigrams and mined multi-word phrases |
| ↳ *Core Closed-Class Seeds* | ~160 | Pronouns, prepositions, conjunctions, primary auxiliaries (`SEALED_WORDS`) |
| ↳ *High-Frequency Words* | ~3,500 | Open-class vocabulary mined directly from training corpora by frequency |
| ↳ *Multi-word Phrases* | ~324 | High-mutual-information collocations (e.g., `as well as`, `for example`) |
| **Total Vocabulary** | **4,096** | **Strictly bounded, indexable in `0..4095` (`u16`)** |

---

## 4. Functional Requirements

### FR-1: Data-Driven Vocabulary Induction
- The model shall build its vocabulary from a configurable training cache (`training-cache/samples.bin` or raw text corpora).
- Frequencies of single words and multi-word n-grams shall be computed statistically.
- Words with high corpus frequency and distinct usage shall be prioritized for individual canonical token slots.
- Static thesaurus collapsing is eliminated for vocabulary induction. Distinct concepts (such as `world` vs. `universe`, `code` vs. `script`, `begin` vs. `start`) retain separate IDs whenever their frequency warrants it.

### FR-2: Greedy Longest-Match Phrase Tokenization
- The tokenizer shall support multi-word phrase tokens for high-frequency collocations (e.g. `at the same time`, `in order to`, `as well as`).
- During tokenization, input token streams are matched using greedy longest-match prefix search:
  1. Multi-word phrase matches are emitted first as single token IDs, reducing sequence length and preserving exact idioms.
  2. Individual vocabulary words are emitted for direct matches.
  3. Morphological lemmatization handles regular inflections (`-s`, `-ed`, `-ing`) if the base lemma is in the vocabulary.

### FR-3: Context-Aware Fallback & Entity Preservation
- Capitalized entities not in the 4,096 vocabulary map to `<NAME_0>`..`<NAME_47>`, with their raw strings saved in `TranslationContext::entities`.
- Numbers map to `<NUM_0>`..`<NUM_19>`, with numeric strings stored in `TranslationContext::numbers`.
- Rare out-of-vocabulary open-class words map to template slots `<SLOT_0>`..`<SLOT_15>`, preserving their exact strings in `TranslationContext::slots`.
- The sequence remains 100% reconstructible without data corruption.

### FR-4: Decompressor Compatibility & Invertibility
- The generated vocabulary format and token ID layout shall remain directly compatible with `NGramDecompressor` and `cache_yumon_talk.rs`.
- The model provides serialization and deserialization (`kilo_vocab.json` / binary `.bin`).
- An evaluation interface measures:
  - Token compression ratio (original word count vs. emitted Kilo token count).
  - Out-of-vocabulary slot rate (`<SLOT_x>` frequency).
  - Reconstruction BLEU / exact-match fidelity when piped through `NGramDecompressor`.

---

## 5. Implementation Architecture & Interfaces

### Data Structure (`src/brain/kilo_vocab.rs`)

```rust
pub struct KiloVocab {
    pub version: usize,
    pub vocab_size: usize,                    // Exactly 4096
    pub special_tokens: Vec<String>,          // 4
    pub punctuation: Vec<String>,             // 24
    pub placeholders: Vec<String>,            // 84
    pub vocabulary: Vec<String>,              // 3984 (words + phrases)
    
    // Fast lookup tables
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub phrase_trie: PhraseTrie,              // Prefix trie for multi-word token matching
}

impl KiloVocab {
    /// Learn vocabulary from word/phrase statistics.
    pub fn learn_from_corpus(
        word_counts: &HashMap<String, usize>,
        phrase_counts: &HashMap<String, usize>,
        target_vocab_size: usize,
    ) -> Self;

    /// Tokenize text into Kilo token IDs with entity tracking.
    pub fn encode(&self, text: &str, ctx: &mut TranslationContext) -> Result<Vec<usize>>;

    /// Detokenize Kilo token IDs back into canonical token text.
    pub fn decode_canonical(&self, tokens: &[usize]) -> String;

    /// Save to JSON or binary.
    pub fn save_to_file(&self, path: &Path) -> Result<()>;
    pub fn load_from_file(path: &Path) -> Result<Self>;
}
```

### Integration Workflow

1. **Vocabulary Training:** `build_yumon_talk_vocab` is updated to use `KiloVocab::learn_from_corpus`, extracting top unigrams and recurring bigrams/trigrams, saving `training-cache/kilo_vocab.json`.
2. **Dataset Compression:** `cache_yumon_talk` loads `KiloVocab` and invokes `vocab.encode(prompt, ctx)` and `vocab.encode(reply, ctx)`, writing the cached samples to `training-cache/yumon_talk_samples.bin`.
3. **Decompression:** `NGramDecompressor` accepts the token sequence and context, expanding phrase and word tokens into natural sentences with beam search.
