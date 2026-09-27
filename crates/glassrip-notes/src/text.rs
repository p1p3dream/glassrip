//! Text helpers: normalization for matching, house-style sanitizing, times.

/// Lowercase alphanumeric tokens of `s`, joined by single spaces.
///
/// Used for quote checks: case, whitespace and punctuation are ignored, words are not.
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.extend(c.to_lowercase());
        } else if c == '\'' || c == '\u{2019}' {
            // "don't" and "dont" compare equal
        } else {
            pending_space = true;
        }
    }
    out
}

/// Normalized tokens of `s`.
pub fn tokens(s: &str) -> Vec<String> {
    normalize(s)
        .split(' ')
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "we", "you", "i", "they", "he", "she", "our", "your", "their", "as", "so", "do",
    "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "what", "how",
    "which", "who", "if", "then", "than", "into", "about", "just", "like", "any", "anything",
    "there", "here", "some", "maybe", "also", "not", "no", "yes", "um", "uh",
];

/// True for common function words that carry no topic.
pub fn is_stopword(t: &str) -> bool {
    STOPWORDS.contains(&t)
}

/// Normalized tokens of `s` without stopwords.
pub fn content_tokens(s: &str) -> Vec<String> {
    tokens(s).into_iter().filter(|t| !is_stopword(t)).collect()
}

/// Words too generic to identify a board target on their own ("Kiosk App" is
/// identified by "kiosk", not "app").
const GENERIC_WORDS: &[&str] = &[
    "api",
    "apis",
    "app",
    "apps",
    "application",
    "backend",
    "box",
    "client",
    "code",
    "component",
    "components",
    "config",
    "core",
    "data",
    "database",
    "db",
    "design",
    "doc",
    "docs",
    "feature",
    "features",
    "flow",
    "frontend",
    "item",
    "items",
    "kit",
    "layer",
    "main",
    "model",
    "module",
    "new",
    "old",
    "page",
    "part",
    "piece",
    "platform",
    "project",
    "screen",
    "server",
    "service",
    "services",
    "setup",
    "side",
    "store",
    "stuff",
    "system",
    "systems",
    "team",
    "thing",
    "things",
    "tool",
    "tools",
    "ui",
    "ux",
    "view",
    "work",
];

/// The tokens that are not generic words.
pub fn distinctive_tokens(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .filter(|t| !GENERIC_WORDS.contains(&t.as_str()))
        .cloned()
        .collect()
}

/// True when `words` name a target whose content tokens are `target`: one of
/// its distinctive tokens, or, for a target made only of generic words, all of
/// them. One generic word ("app") never names a target.
pub fn names_target(words: &std::collections::BTreeSet<String>, target: &[String]) -> bool {
    let distinctive: Vec<&String> = target
        .iter()
        .filter(|t| !GENERIC_WORDS.contains(&t.as_str()))
        .collect();
    if distinctive.is_empty() {
        !target.is_empty() && target.iter().all(|t| words.contains(t))
    } else {
        distinctive.into_iter().any(|t| words.contains(t))
    }
}

/// Jaccard similarity of two token sets.
pub fn jaccard(a: &[String], b: &[String]) -> f64 {
    use std::collections::BTreeSet;
    let sa: BTreeSet<&str> = a.iter().map(String::as_str).collect();
    let sb: BTreeSet<&str> = b.iter().map(String::as_str).collect();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    sa.intersection(&sb).count() as f64 / union as f64
}

/// Replaces em dashes, en dashes and double hyphens with house-style punctuation.
///
/// A dash between two numbers becomes " to "; any other dash becomes ", ".
pub fn sanitize_dashes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let is_dash = c == '\u{2014}' || c == '\u{2013}' || c == '\u{2015}';
        let is_double = c == '-' && chars.get(i + 1) == Some(&'-');
        if is_dash || is_double {
            let width = if is_double {
                let mut n = 0;
                while chars.get(i + n) == Some(&'-') {
                    n += 1;
                }
                n
            } else {
                1
            };
            let prev = out.trim_end().chars().last();
            let next = chars[i + width..]
                .iter()
                .find(|c| !c.is_whitespace())
                .copied();
            while out.ends_with(' ') {
                out.pop();
            }
            let numeric = prev.is_some_and(|p| p.is_ascii_digit())
                && next.is_some_and(|n| n.is_ascii_digit());
            if numeric {
                out.push_str(" to ");
            } else if prev.is_some() && next.is_some() {
                if !out.ends_with(',') && !out.ends_with(':') && !out.ends_with(';') {
                    out.push(',');
                }
                out.push(' ');
            }
            i += width;
            while chars.get(i).is_some_and(|c| c.is_whitespace()) {
                i += 1;
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out.trim().to_string()
}

/// Formats seconds as `mm:ss` (or `h:mm:ss` from one hour).
pub fn mmss(t_s: f64) -> String {
    let total = if t_s.is_finite() && t_s > 0.0 {
        t_s.floor() as u64
    } else {
        0
    };
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Rough token estimate for prompt budgeting (bytes / 3.5, rounded up).
pub fn estimate_tokens(s: &str) -> usize {
    (s.len() * 2).div_ceil(7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_ignores_case_punctuation_and_apostrophes() {
        assert_eq!(normalize("Let's  skip, the STEP!"), "lets skip the step");
        assert_eq!(normalize("  "), "");
        assert_eq!(normalize("a...b"), "a b");
    }

    #[test]
    fn dashes_become_commas_or_ranges() {
        assert_eq!(sanitize_dashes("ship it \u{2014} later"), "ship it, later");
        assert_eq!(sanitize_dashes("ship it\u{2014}later"), "ship it, later");
        assert_eq!(sanitize_dashes("pages 3\u{2013}5"), "pages 3 to 5");
        assert_eq!(sanitize_dashes("a -- b"), "a, b");
        assert_eq!(sanitize_dashes("well-known"), "well-known");
        assert_eq!(sanitize_dashes("\u{2014} lead"), "lead");
    }

    #[test]
    fn mmss_formats() {
        assert_eq!(mmss(0.0), "00:00");
        assert_eq!(mmss(28.74), "00:28");
        assert_eq!(mmss(1851.0), "30:51");
        assert_eq!(mmss(3725.0), "1:02:05");
        assert_eq!(mmss(f64::NAN), "00:00");
    }

    #[test]
    fn jaccard_of_content_tokens() {
        let a = content_tokens("Which fonts do we pick for the menu?");
        let b = content_tokens("What fonts does the menu pick?");
        assert!(jaccard(&a, &b) > 0.4);
        assert_eq!(jaccard(&[], &[]), 0.0);
    }
}
