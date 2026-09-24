//! Screen-type metrics: confusion matrix, accuracy, and the
//! "0 CMS keyframes read as whiteboard" gate (spec 9.3).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::Tally;

/// Closed set of screen types (spec 6.7), including the document-mode types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenType {
    /// Diagram or whiteboard canvas.
    Whiteboard,
    /// Video call gallery or status screen.
    MeetGallery,
    /// Presentation slide.
    Slides,
    /// Code editor or terminal.
    Code,
    /// Content management studio.
    Cms,
    /// Chat application.
    Chat,
    /// Other web page.
    Web,
    /// Desktop, file browser, settings.
    Desktop,
    /// Issue tracker ticket (document mode).
    Ticket,
    /// Documentation page (document mode).
    Doc,
    /// Not classifiable.
    Unknown,
}

impl ScreenType {
    /// snake_case name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Whiteboard => "whiteboard",
            Self::MeetGallery => "meet_gallery",
            Self::Slides => "slides",
            Self::Code => "code",
            Self::Cms => "cms",
            Self::Chat => "chat",
            Self::Web => "web",
            Self::Desktop => "desktop",
            Self::Ticket => "ticket",
            Self::Doc => "doc",
            Self::Unknown => "unknown",
        }
    }

    /// Parses a snake_case name.
    pub fn parse(s: &str) -> Option<Self> {
        serde_json::from_value(serde_json::Value::String(s.trim().to_string())).ok()
    }
}

impl From<glassrip_vision::classify::ScreenType> for ScreenType {
    fn from(t: glassrip_vision::classify::ScreenType) -> Self {
        use glassrip_vision::classify::ScreenType as V;
        match t {
            V::Whiteboard => Self::Whiteboard,
            V::MeetGallery => Self::MeetGallery,
            V::Slides => Self::Slides,
            V::Code => Self::Code,
            V::Cms => Self::Cms,
            V::Chat => Self::Chat,
            V::Web => Self::Web,
            V::Desktop => Self::Desktop,
            V::Unknown => Self::Unknown,
        }
    }
}

/// Column label used in the confusion matrix for a gold label with no joined prediction.
pub const MISSED: &str = "missed";

/// Screen-type scores.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScreenScore {
    /// gold type -> predicted type (or `missed`) -> count.
    pub confusion: BTreeMap<String, BTreeMap<String, usize>>,
    /// Correct over all gold labels (missed ones count as wrong).
    pub tally: Tally,
    /// Gold `cms` predicted as `whiteboard` (must be 0).
    pub cms_as_whiteboard: usize,
    /// Gold labels that joined to no predicted keyframe.
    pub missed: usize,
}

impl ScreenScore {
    /// Records one gold label and its joined prediction.
    pub fn record(&mut self, gold: ScreenType, pred: Option<ScreenType>) {
        let col = pred.map_or(MISSED, |p| p.as_str());
        *self
            .confusion
            .entry(gold.as_str().to_string())
            .or_default()
            .entry(col.to_string())
            .or_default() += 1;
        self.tally.record(pred == Some(gold));
        if gold == ScreenType::Cms && pred == Some(ScreenType::Whiteboard) {
            self.cms_as_whiteboard += 1;
        }
        if pred.is_none() {
            self.missed += 1;
        }
    }

    /// Pools another score.
    pub fn add(&mut self, o: &ScreenScore) {
        for (g, row) in &o.confusion {
            let r = self.confusion.entry(g.clone()).or_default();
            for (p, n) in row {
                *r.entry(p.clone()).or_default() += n;
            }
        }
        self.tally.add(o.tally);
        self.cms_as_whiteboard += o.cms_as_whiteboard;
        self.missed += o.missed;
    }
}

/// Scores `(gold, joined prediction)` pairs.
pub fn score_screens(pairs: &[(ScreenType, Option<ScreenType>)]) -> ScreenScore {
    let mut s = ScreenScore::default();
    for &(g, p) in pairs {
        s.record(g, p);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ScreenType::*;

    #[test]
    fn confusion_accuracy_and_gate() {
        let s = score_screens(&[
            (Whiteboard, Some(Whiteboard)),
            (Whiteboard, Some(Unknown)),
            (Cms, Some(Whiteboard)),
            (Cms, Some(Cms)),
            (Chat, None),
        ]);
        // 2 correct of 5
        assert_eq!(
            s.tally,
            Tally {
                correct: 2,
                total: 5
            }
        );
        assert_eq!(s.cms_as_whiteboard, 1);
        assert_eq!(s.missed, 1);
        assert_eq!(s.confusion["cms"]["whiteboard"], 1);
        assert_eq!(s.confusion["chat"][MISSED], 1);
        assert_eq!(s.confusion["whiteboard"]["unknown"], 1);
    }

    #[test]
    fn parse_and_convert() {
        assert_eq!(ScreenType::parse("meet_gallery"), Some(MeetGallery));
        assert_eq!(ScreenType::parse("nope"), None);
        assert_eq!(
            ScreenType::from(glassrip_vision::classify::ScreenType::Cms),
            Cms
        );
        for t in [Whiteboard, Ticket, Doc, Unknown] {
            assert_eq!(ScreenType::parse(t.as_str()), Some(t));
        }
    }
}
