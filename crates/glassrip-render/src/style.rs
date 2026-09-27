//! House style: palette, semantic roles, text metrics.

use serde::Serialize;

/// Semantic role of a component, from generic words in its label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Apps, screens and UI kits.
    Client,
    /// APIs and servers.
    Api,
    /// Services and workers.
    Processing,
    /// Databases and stores.
    Storage,
    /// Anything else (often a third-party product).
    External,
}

impl Role {
    /// Card and zone color.
    pub fn color(self) -> &'static str {
        match self {
            // Appendix B has no client color and reserves indigo for analytics;
            // client apps use the external teal
            Role::Client => "#0891b2",
            Role::Api => "#2563eb",
            Role::Processing => "#10b981",
            Role::Storage => "#7c3aed",
            Role::External => "#0891b2",
        }
    }

    /// Id suffix for gradients and filters.
    pub fn key(self) -> &'static str {
        match self {
            Role::Client => "client",
            Role::Api => "api",
            Role::Processing => "processing",
            Role::Storage => "storage",
            Role::External => "external",
        }
    }

    /// Zone title.
    pub fn zone_title(self) -> &'static str {
        match self {
            Role::Client => "Client",
            Role::Api => "API",
            Role::Processing => "Services",
            Role::Storage => "Storage",
            Role::External => "Content and external",
        }
    }

    /// Short description on the card body.
    pub fn describe(self) -> &'static str {
        match self {
            Role::Client => "Client",
            Role::Api => "API layer",
            Role::Processing => "Service",
            Role::Storage => "Data store",
            Role::External => "External system",
        }
    }
}

const CLIENT: &[&str] = &[
    "app",
    "apps",
    "mobile",
    "web",
    "ui",
    "frontend",
    "front-end",
    "design",
    "kit",
    "screen",
    "kiosk",
    "client",
    "renderer",
    "browser",
    "dashboard",
];
const API: &[&str] = &[
    "api", "server", "graphql", "grpc", "gateway", "backend", "back-end", "endpoint", "bff",
    "relay",
];
const PROCESSING: &[&str] = &[
    "service",
    "manager",
    "worker",
    "job",
    "pipeline",
    "processor",
    "queue",
    "scheduler",
    "engine",
];
const STORAGE: &[&str] = &[
    "db",
    "database",
    "store",
    "storage",
    "cache",
    "bucket",
    "warehouse",
    "table",
    "lake",
];

/// Role of a component label.
pub fn role_of(label: &str) -> Role {
    let words: Vec<String> = label
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    let has = |list: &[&str]| words.iter().any(|w| list.contains(&w.as_str()));
    if has(PROCESSING) {
        Role::Processing
    } else if has(API) {
        Role::Api
    } else if has(STORAGE) {
        Role::Storage
    } else if has(CLIENT) {
        Role::Client
    } else {
        Role::External
    }
}

/// Approximate rendered width of `s` at `size` px (Inter-like proportions).
pub fn text_width(s: &str, size: f64, bold: bool) -> f64 {
    let per = if bold { 0.6 } else { 0.55 };
    // full-width scripts take about a full em
    s.chars()
        .map(|c| if c.is_ascii() { size * per } else { size })
        .sum()
}

/// Greedy word wrap to at most `max_chars` per line.
pub fn wrap(text: &str, max_chars: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for w in text.split_whitespace() {
        if !cur.is_empty() && cur.chars().count() + 1 + w.chars().count() > max_chars {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Wraps and truncates to `max_lines`, ending the last line with `...` when cut.
/// A word longer than a line (a URL, an identifier) is split across lines.
pub fn wrap_lines(text: &str, max_chars: usize, max_lines: usize) -> Vec<String> {
    let max = max_chars.max(1);
    let mut lines: Vec<String> = wrap(text, max)
        .into_iter()
        .flat_map(|l| {
            let chars: Vec<char> = l.chars().collect();
            chars
                .chunks(max)
                .map(|c| c.iter().collect::<String>())
                .collect::<Vec<_>>()
        })
        .collect();
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        if let Some(last) = lines.last_mut() {
            let keep: String = last.chars().take(max_chars.saturating_sub(3)).collect();
            *last = format!("{}...", keep.trim_end());
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_from_generic_words() {
        assert_eq!(role_of("Ledger Service"), Role::Processing);
        assert_eq!(role_of("Relay API"), Role::Api);
        assert_eq!(role_of("Kiosk App"), Role::Client);
        assert_eq!(role_of("/frontend design system"), Role::Client);
        assert_eq!(role_of("Orders DB"), Role::Storage);
        assert_eq!(role_of("Acme"), Role::External);
    }

    #[test]
    fn wrapping() {
        assert_eq!(
            wrap("one two three four", 9),
            vec!["one two", "three", "four"]
        );
        assert_eq!(
            wrap_lines("aaa bbb ccc ddd eee", 7, 2),
            vec!["aaa bbb", "ccc..."]
        );
        // an overlong token is split, not left to run past its box
        assert_eq!(
            wrap_lines("see https://example.com/a/b", 8, 5),
            vec!["see", "https://", "example.", "com/a/b"]
        );
        assert_eq!(text_width("ab", 10.0, false), 11.0);
        assert_eq!(text_width("\u{4e2d}\u{6587}", 10.0, false), 20.0);
    }
}
