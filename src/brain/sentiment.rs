use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum Emotion {
    Happy,
    Sad,
    Excited,
    Fearful,
}

#[derive(Default, Debug)]
struct Scores {
    happy: f32,
    excited: f32,
    sad: f32,
    fearful: f32,
}

impl Scores {
    fn max_emotion(&self) -> Emotion {
        let mut best = (Emotion::Happy, self.happy);

        let candidates = [
            (Emotion::Excited, self.excited),
            (Emotion::Sad, self.sad),
            (Emotion::Fearful, self.fearful),
        ];

        for (emotion, score) in candidates {
            if score > best.1 {
                best = (emotion, score);
            }
        }

        best.0
    }
}

pub struct EmotionAnalyzer {
    lexicon: HashMap<Emotion, Vec<&'static str>>,
}

impl EmotionAnalyzer {
    pub fn new() -> Self {
        let mut lexicon = HashMap::new();

        lexicon.insert(Emotion::Happy, vec![
            "happy", "glad", "content", "pleased", "good", "nice", "relieved"
        ]);

        lexicon.insert(Emotion::Excited, vec![
            "excited", "thrilled", "hyped", "pumped", "ecstatic",
            "amazing", "awesome", "incredible", "lets go"
        ]);

        lexicon.insert(Emotion::Sad, vec![
            "sad", "down", "depressed", "unhappy", "miserable",
            "crying", "heartbroken", "lonely", "tired"
        ]);

        lexicon.insert(Emotion::Fearful, vec![
            "scared", "afraid", "terrified", "nervous",
            "anxious", "worried", "panic", "panicking", "fear"
        ]);

        Self { lexicon }
    }

    pub fn analyze(&self, text: &str, sentiment: &sentiment::Analysis) -> String {
        let normalized = text.to_lowercase();

        let mut scores = Scores::default();

        // --- INTENSITY SIGNALS ---
        let exclamation_boost = text.matches('!').count() as f32 * 0.1;
        let caps_boost = if text.chars().any(|c| c.is_uppercase()) {
            0.2
        } else {
            0.0
        };

        let intensity = sentiment.comparative.abs() + exclamation_boost + caps_boost;
        let valence = sentiment.score;

        // --- KEYWORD MATCHING ---
        for (emotion, words) in &self.lexicon {
            for word in words {
                if normalized.contains(word) {
                    let base = 1.0 + intensity;

                    match emotion {
                        Emotion::Happy => scores.happy += base,
                        Emotion::Excited => scores.excited += base * 1.2,
                        Emotion::Sad => scores.sad += base,
                        Emotion::Fearful => scores.fearful += base * 1.2,
                    }
                }
            }
        }

        // --- SENTIMENT INJECTION ---
        if valence > 0.0 {
            scores.happy += valence;
            scores.excited += intensity * 1.5;
        } else {
            scores.sad += valence.abs();
            scores.fearful += intensity * 1.5;
        }

        // --- INTENSITY SHAPING ---
        if intensity > 0.5 {
            scores.excited *= 1.3;
            scores.fearful *= 1.3;
        } else {
            scores.happy *= 1.1;
            scores.sad *= 1.1;
        }

        let max_emotion = scores.max_emotion();

        match max_emotion {
            Emotion::Excited => {
                "excited".to_string()
            },
            Emotion::Fearful => {
                "fearful".to_string()
            },
            Emotion::Happy => {
                "happy".to_string()
            },
            Emotion::Sad => {
                "sad".to_string()
            },
            _ => {
                "neutral".to_string()
            }
        }

    }
}