// CREDIT to: json-fix = "0.1.1"

use fancy_regex::{Captures, Regex};
use serde_json;

/// Robust JSON fixer for malformed AI or chatbot output.
/// Designed to recover from common formatting issues in JSON-like text.
#[derive(Debug)]
pub struct FixReport {
    pub original: String,
    pub fixed: String,
    pub steps: Vec<String>,
    pub success: bool,
}

#[derive(Debug)]
enum FixStep {
    RemoveEscapedQuoteComma,
    // ... (other steps could be added here)
}

fn apply_step<F: FnOnce(String) -> String>(input: String, _step: FixStep, f: F) -> String {
    f(input)
}

pub fn fix_json_syntax(input: &str) -> FixReport {
    let mut steps = Vec::new();
    let mut fixed = input.trim().to_string();

    // 1. Strip markdown wrappers or triple quotes
    if fixed.starts_with("```json") || fixed.starts_with("```") {
        fixed = fixed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string();
        steps.push("Stripped markdown wrappers or triple quotes".to_string());
    }

    // 2. Fix missing commas between fields (e.g. "title": "x" "body": "y")
    let re_missing_commas = Regex::new(r#"(\"[^\"]+\"\s*:\s*\"[^\"]+\")\s+\""#).unwrap();
    let new_fixed = re_missing_commas.replace_all(&fixed, "$1,\n\"").to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed missing commas between fields".to_string());
    }

    // 3. Fix adjacent object blocks
    let new_fixed = fixed.replace("}\n{", "},\n{");
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed adjacent object blocks".to_string());
    }

    // 4. Fix generic missing commas between quoted values
    let re_adjacent_quoted = Regex::new(r#""\s+""#).unwrap();
    let new_fixed = re_adjacent_quoted
        .replace_all(&fixed, "\",\n\"")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed generic missing commas between quoted values".to_string());
    }

    // 5. Remove trailing commas in arrays or objects
    let re_trailing_commas = Regex::new(r",\s*([\]}])").unwrap();
    let new_fixed = re_trailing_commas.replace_all(&fixed, "$1").to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Removed trailing commas".to_string());
    }

    // 6. Convert single quotes to double quotes (if outside word boundaries)
    let re_single_quotes = Regex::new(r"'([^']*)'").unwrap();
    let new_fixed = re_single_quotes.replace_all(&fixed, "\"$1\"").to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Converted single quotes to double quotes".to_string());
    }

    // 7. Convert curly quotes and weird symbols
    let new_fixed = fixed
        .replace('“', "\"")
        .replace('”', "\"")
        .replace('‘', "'")
        .replace('’', "'");
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Converted curly quotes and weird symbols".to_string());
    }

    // 9 and 8. Fix broken contractions and apostrophes.
    // Note: Step 9 must come before Step 8 to avoid conflicts.
    // Step 9 fixes contractions written with double quotes instead of apostrophes (e.g. I"m → I'm)
    let re_broken_contractions = Regex::new(r#"(\b\w+)"(\w+)"#).unwrap();
    let new_fixed = re_broken_contractions
        .replace_all(&fixed, "$1'$2")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed broken contractions written with double quotes".to_string());
    }

    // Step 8 fixes broken apostrophes written as quotes (e.g., it"s → it's)
    let re_broken_apostrophes = Regex::new(r#"(\w)"([sdmt])\b"#).unwrap();
    let new_fixed = re_broken_apostrophes
        .replace_all(&fixed, "$1'$2")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed broken apostrophes written as quotes".to_string());
    }

    // 10. Handle escaped stringified JSON
    if fixed.starts_with('\"') && fixed.ends_with('\"') {
        if let Ok(unescaped) = serde_json::from_str::<String>(&fixed) {
            if unescaped != fixed {
                fixed = unescaped;
                steps.push("Handled escaped stringified JSON".to_string());
            }
        }
    }

    // 11. Quote unquoted keys (e.g. name: "John" → "name": "John")
    let re_unquoted_keys = Regex::new(r#"(?m)(^|[{,\s])(\w+)(\s*:\s*)""#).unwrap();
    let new_fixed = re_unquoted_keys
        .replace_all(&fixed, "$1\"$2\"$3")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Quoted unquoted keys".to_string());
    }

    // 12. Escape unescaped inner double quotes within values
    // Note: This step escapes inner quotes inside string values.
    // Step 14.6 also escapes inner quotes, so order and pattern safeguards are important to avoid double escaping.
    let re_unescaped_inner_quotes = Regex::new(r#":\s*"([^"]*?)"([^\\"][^"]*?)""#).unwrap();
    let new_fixed = re_unescaped_inner_quotes
        .replace_all(&fixed, r#": "$1\"$2""#)
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Escaped unescaped inner double quotes within values".to_string());
    }

    // 13. Remove invalid escape sequences
    let re_invalid_escapes = Regex::new(r#"\\[^"\\/bfnrt]"#).unwrap();
    let new_fixed = re_invalid_escapes.replace_all(&fixed, "").to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Removed invalid escape sequences".to_string());
    }

    // 14. Convert raw newlines in strings to \n
    // Fixed closure signature to use fancy_regex::Captures as required by fancy_regex crate.
    let re_multiline_strings = Regex::new(r#""([^"]*?)\n([^"]*?)""#).unwrap();
    let new_fixed = re_multiline_strings
        .replace_all(&fixed, |caps: &Captures| {
            let first = &caps[1].replace('\n', "\\n");
            let second = &caps[2].replace('\n', "\\n");
            format!("\"{}\\n{}\"", first, second)
        })
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Converted raw newlines in strings to \\n".to_string());
    }

    // 14.5: Generic embedded key fixer – fixes when a second key gets trapped inside a string value
    let re_embedded_key_start = Regex::new(
        r#""(?P<key1>\w+)"\s*:\s*"(?P<val>[^"]*?),\s*\\?"(?P<key2>\w+)"\s*:\s*(?P<val2>[^"{}\[\],]+)"#
    ).unwrap();

    let new_fixed = re_embedded_key_start
        .replace_all(&fixed, |caps: &Captures| {
            let key1 = &caps["key1"];
            let val = &caps["val"];
            let key2 = &caps["key2"];
            let val2 = &caps["val2"];
            format!(r#""{}": "{}", "{}": {}"#, key1, val.trim(), key2, val2)
        })
        .to_string();

    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed embedded key-value pair trapped inside string".to_string());
    }

    // 14.45. Fix embedded key start inside unescaped value (e.g., "emotion": "hopeful, "score": 80)
    let re_embedded_key_start = Regex::new(
        r#""(?P<key1>\w+)"\s*:\s*"(?P<val>[^"]*?),\s*"(?P<key2>\w+)"\s*:\s*(?P<val2>[^"{}\[\],]+)"#,
    )
    .unwrap();

    let new_fixed = re_embedded_key_start
        .replace_all(&fixed, |caps: &Captures| {
            let key1 = &caps["key1"];
            let val = &caps["val"];
            let key2 = &caps["key2"];
            let val2 = &caps["val2"];
            format!(
                r#""{}": "{}", "{}": {}"#,
                key1,
                val.trim(),
                key2,
                val2.trim()
            )
        })
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed embedded key start inside unescaped value".to_string());
    }

    // 14.55. Fix misescaped internal quote sequences (e.g., "text with \"some quote" → "text with some quote")
    let re_misescaped = Regex::new(r#"(?P<key>:\s*")(?P<val>[^"]*?)\\",\s*(?P<rest>")"#).unwrap();
    let new_fixed = re_misescaped
        .replace_all(&fixed, "${key}${val}, ${rest}")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Fixed misescaped internal quote sequences".to_string());
    }

    // 14.6. Escape unescaped double quotes inside string values (e.g., "... "word" ...")
    // This step is after step 12 to avoid double escaping.
    // The regex is designed to avoid matching already escaped quotes.
    let re_inner_unescaped_quotes =
        Regex::new(r#"(".*?:\s*")((?:[^"\\]|\\.)*?)"((?:[^"\\]|\\.)*?)""#).unwrap();
    let new_fixed = re_inner_unescaped_quotes
        .replace_all(&fixed, "$1$2\\\"$3\"")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Escaped unescaped double quotes inside string values".to_string());
    }

    // 15. Auto-fix dangling or mismatched brackets/braces using a shallow stack matcher
    let mut stack = vec![];
    let mut cleaned = String::new();
    for c in fixed.chars() {
        match c {
            '{' | '[' => {
                stack.push(c);
                cleaned.push(c);
            }
            '}' => {
                if stack.last() == Some(&'{') {
                    stack.pop();
                    cleaned.push('}');
                }
                // else skip unmatched }
            }
            ']' => {
                if stack.last() == Some(&'[') {
                    stack.pop();
                    cleaned.push(']');
                }
                // else skip unmatched ]
            }
            _ => cleaned.push(c),
        }
    }
    // Auto-close any remaining opened delimiters
    while let Some(c) = stack.pop() {
        match c {
            '{' => cleaned.push('}'),
            '[' => cleaned.push(']'),
            _ => {}
        }
    }
    if cleaned != fixed {
        fixed = cleaned;
        steps.push("Auto-fixed dangling or mismatched brackets/braces".to_string());
    }

    // Normalize line breaks between quoted array elements (e.g. from GPT)
    let re_stray_array_linebreaks = Regex::new(r#"(\"\s*),\s*\\n\s*(\")"#).unwrap();
    let new_fixed = re_stray_array_linebreaks
        .replace_all(&fixed, "$1, $2")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Normalized line breaks between quoted array elements".to_string());
    }

    // Also normalize overly escaped array strings with embedded linebreaks
    let re_array_line_merger = Regex::new(r#"\",\s*\\n\s*\""#).unwrap();
    let new_fixed = re_array_line_merger
        .replace_all(&fixed, "\", \"")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Normalized overly escaped array strings with embedded linebreaks".to_string());
    }

    // Normalize line breaks between array items
    let re_linebreaks_between_items = Regex::new(r#"\",\s*\n\s*\""#).unwrap();
    let new_fixed = re_linebreaks_between_items
        .replace_all(&fixed, "\", \"")
        .to_string();
    if new_fixed != fixed {
        fixed = new_fixed;
        steps.push("Normalized line breaks between array items".to_string());
    }

    // 15.9. Remove stray escaped quote-comma sequences (e.g. \"\, → \",)
    fixed = apply_step(fixed, FixStep::RemoveEscapedQuoteComma, |s| {
        s.replace("\\\",", "\",")
    });

    // 16. Final clean-up: attempt full JSON parse and re-serialize if possible
    // Moved to the end after all mutations, including bracket fixing, for best recovery chance.
    // let success = if let Ok(val) = serde_json::from_str::<serde_json::Value>(&fixed) {
    //     if let Ok(re) = serde_json::to_string_pretty(&val) {
    //         fixed = re;
    //         true
    //     } else {
    //         false
    //     }
    // } else {
    //     if let Err(e) = serde_json::from_str::<serde_json::Value>(&fixed) {
    //         println!("❌ Final JSON parse error: {}", e);
    //         let line = e.line();
    //         let column = e.column();
    //         println!("📍 Error occurred at line {}, column {}", line, column);
    //         println!("📍 Faulty fixed JSON:\n{:#?}", fixed);
    //     }
    //     false
    // };

    FixReport {
        original: input.to_string(),
        fixed,
        steps,
        success: true,
    }
}
