//! Easy-to-hard ordering of a prompt/reply cache for an inner-epoch curriculum.
//!
//! Each sample gets three features: length (active prompt + reply tokens),
//! rarity (mean unigram surprisal of its tokens, in bits, against the whole
//! cache: the word-choice measure) and complexity (clause punctuation in the
//! text plus placeholder tokens). Each feature is turned into a percentile rank
//! so the units are comparable, and the weighted mean of the ranks is the
//! difficulty. Samples are stably sorted by it, easiest first. DPO caches are
//! ordered by pair (mean of chosen and rejected) so pairs stay adjacent.
use super::{BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, samples::Sample};
use anyhow::{Result, ensure};

/// Cache metadata `order` tag written for difficulty-sorted caches.
pub const DIFFICULTY_ORDER: &str = "difficulty-v1";

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub length: f64,
    pub rarity: f64,
    pub complexity: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self { length: 1.0, rarity: 1.0, complexity: 1.0 }
    }
}

impl std::str::FromStr for Weights {
    type Err = anyhow::Error;
    /// "length,rarity,complexity", e.g. "1,1,1" or "2,1,0".
    fn from_str(s: &str) -> Result<Self> {
        let values = s.split(',').map(|v| v.trim().parse::<f64>()).collect::<Result<Vec<_>, _>>()?;
        ensure!(values.len() == 3, "weights are length,rarity,complexity");
        ensure!(values.iter().all(|&v| v >= 0.0) && values.iter().sum::<f64>() > 0.0,
            "weights must be non-negative and not all zero");
        Ok(Self { length: values[0], rarity: values[1], complexity: values[2] })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Features {
    pub length: usize,
    pub rarity: f64,
    pub complexity: usize,
    pub difficulty: f64,
}

/// Active tokens of a prompt/reply sample: prompt after BOS, reply before EOS.
fn active_tokens(sample: &Sample) -> impl Iterator<Item = usize> + '_ {
    let prompt = sample.input_ids.iter().take_while(|&&t| t != PAD_TOKEN).filter(|&&t| t != BOS_TOKEN);
    let reply = sample.target_labels.iter().take_while(|&&t| t != PAD_TOKEN && t != EOS_TOKEN);
    prompt.chain(reply).copied()
}

fn clause_marks(text: &str) -> usize {
    text.chars().filter(|c| matches!(c, ',' | ';' | ':' | '.' | '!' | '?' | '\n')).count()
}

/// Share of repeated token 4-grams in a sample (0 = all distinct, near 1 = one
/// phrase over and over). Chant-like samples ("da na na na na . . .") would
/// otherwise rank as hard because they are long and punctuation-heavy.
pub fn repetition(sample: &Sample) -> f64 {
    let tokens: Vec<usize> = active_tokens(sample).collect();
    if tokens.len() < 8 { return 0.0; }
    let grams: Vec<&[usize]> = tokens.windows(4).collect();
    let distinct: std::collections::HashSet<&[usize]> = grams.iter().copied().collect();
    1.0 - distinct.len() as f64 / grams.len() as f64
}

/// Drops samples whose `repetition` exceeds `max` (pairs drop together when either
/// is over). Returns the number of samples removed.
pub fn drop_repetitive(samples: &mut Vec<Sample>, max: f64, pairs: bool) -> usize {
    let group = if pairs { 2 } else { 1 };
    let before = samples.len();
    let keep: Vec<bool> = samples.chunks(group)
        .map(|g| g.iter().all(|s| repetition(s) <= max)).collect();
    let mut index = 0;
    samples.retain(|_| { let k = keep[index / group]; index += 1; k });
    before - samples.len()
}

/// Percentile rank in [0, 1]; ties share their mean rank.
fn percentile_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; n];
    let scale = (n.max(2) - 1) as f64;
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && values[order[end]] == values[order[start]] { end += 1; }
        let rank = (start + end - 1) as f64 / 2.0 / scale;
        for &i in &order[start..end] { ranks[i] = rank; }
        start = end;
    }
    ranks
}

/// Features of every sample. `is_placeholder` marks slot/name/number tokens.
pub fn features(samples: &[Sample], weights: Weights, is_placeholder: impl Fn(usize) -> bool) -> Vec<Features> {
    let mut counts = std::collections::HashMap::<usize, u64>::new();
    let mut total = 0u64;
    for sample in samples {
        for token in active_tokens(sample) {
            *counts.entry(token).or_default() += 1;
            total += 1;
        }
    }
    let mut out: Vec<Features> = samples.iter().map(|sample| {
        let (mut length, mut bits, mut placeholders) = (0usize, 0.0f64, 0usize);
        for token in active_tokens(sample) {
            length += 1;
            bits -= (counts[&token] as f64 / total as f64).log2();
            placeholders += is_placeholder(token) as usize;
        }
        Features {
            length,
            rarity: if length == 0 { 0.0 } else { bits / length as f64 },
            complexity: clause_marks(&sample.pair.0) + clause_marks(&sample.pair.1) + placeholders,
            difficulty: 0.0,
        }
    }).collect();
    let length = percentile_ranks(&out.iter().map(|f| f.length as f64).collect::<Vec<_>>());
    let rarity = percentile_ranks(&out.iter().map(|f| f.rarity).collect::<Vec<_>>());
    let complexity = percentile_ranks(&out.iter().map(|f| f.complexity as f64).collect::<Vec<_>>());
    let sum = weights.length + weights.rarity + weights.complexity;
    for (i, f) in out.iter_mut().enumerate() {
        f.difficulty = (weights.length * length[i] + weights.rarity * rarity[i]
            + weights.complexity * complexity[i]) / sum;
    }
    out
}

/// Sorts samples easiest first (stable) and returns the features in the new order.
/// With `pairs`, consecutive (chosen, rejected) samples move together.
pub fn sort_by_difficulty(samples: &mut Vec<Sample>, weights: Weights, pairs: bool,
    is_placeholder: impl Fn(usize) -> bool) -> Result<Vec<Features>> {
    let group = if pairs { 2 } else { 1 };
    ensure!(samples.len() % group == 0, "pair ordering needs an even number of samples");
    let features = features(samples, weights, is_placeholder);
    let key = |g: usize| (0..group).map(|j| features[g * group + j].difficulty).sum::<f64>() / group as f64;
    let mut groups: Vec<usize> = (0..samples.len() / group).collect();
    groups.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
    let mut slots: Vec<Option<Sample>> = std::mem::take(samples).into_iter().map(Some).collect();
    let mut sorted_features = Vec::with_capacity(slots.len());
    for g in groups {
        for j in 0..group {
            let i = g * group + j;
            samples.push(slots[i].take().unwrap());
            sorted_features.push(features[i]);
        }
    }
    Ok(sorted_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::samples::{Action, CardinalDir, WorldContext};

    fn sample(prompt: &[usize], reply: &[usize], text: &str) -> Sample {
        let mut input_ids = vec![BOS_TOKEN];
        input_ids.extend(prompt);
        input_ids.resize(16, PAD_TOKEN);
        let mut target_labels = reply.to_vec();
        target_labels.push(EOS_TOKEN);
        target_labels.resize(16, PAD_TOKEN);
        Sample { input_ids, target_labels, action: Action::Sit, motion_dir: CardinalDir::None,
            world: WorldContext::default(), target_json: String::new(), pair: (String::new(), text.into()) }
    }

    #[test]
    fn short_common_samples_come_first() {
        let mut samples = vec![
            sample(&[10, 11, 12, 13, 14, 15], &[16, 17, 18, 19, 20], "a, b; c. d!"),
            sample(&[10, 10], &[10, 11], "a"),
            sample(&[10, 11, 12], &[10, 11, 30], "a, b"),
        ];
        let features = sort_by_difficulty(&mut samples, Weights::default(), false, |_| false).unwrap();
        assert_eq!(samples[0].pair.1, "a");
        assert_eq!(samples[2].pair.1, "a, b; c. d!");
        assert!(features.windows(2).all(|w| w[0].difficulty <= w[1].difficulty));
        assert_eq!(features[0].length, 4);
    }

    #[test]
    fn pairs_stay_together() {
        let mut samples = vec![
            sample(&[10, 11, 12, 13], &[20, 21, 22, 23], "hard, hard"),
            sample(&[10, 11, 12, 13], &[24, 25, 26, 27], "hard, hard"),
            sample(&[10], &[10], "easy"),
            sample(&[10], &[11], "easy"),
        ];
        sort_by_difficulty(&mut samples, Weights::default(), true, |_| false).unwrap();
        assert_eq!(samples[0].pair.1, "easy");
        assert_eq!(samples[1].pair.1, "easy");
        assert_eq!(samples[1].target_labels[0], 11);
        assert!(sort_by_difficulty(&mut samples[..3].to_vec(), Weights::default(), true, |_| false).is_err());
    }

    #[test]
    fn repetitive_samples_are_dropped() {
        let chant: Vec<usize> = [20, 21, 21, 21, 21, 22].repeat(5);
        let varied: Vec<usize> = (20..50).collect();
        let mut samples = vec![sample(&varied[..12], &varied[12..], "varied"), sample(&chant[..12], &chant[12..], "chant")];
        assert!(repetition(&samples[0]) == 0.0 && repetition(&samples[1]) > 0.5);
        assert_eq!(drop_repetitive(&mut samples, 0.5, false), 1);
        assert_eq!(samples[0].pair.1, "varied");
    }

    #[test]
    fn weights_parse() {
        let w: Weights = "2,0,1".parse().unwrap();
        assert_eq!((w.length, w.rarity, w.complexity), (2.0, 0.0, 1.0));
        assert!("0,0,0".parse::<Weights>().is_err());
        assert!("1,1".parse::<Weights>().is_err());
    }
}
