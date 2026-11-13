use regex::Regex;
use serde::Serialize;
use thiserror::Error;

/// Regex for modern Spanish license plates (since Sept 2000): 4 digits + 3 consonants
/// excluding vowels (A, E, I, O, U), Ñ, and Q.
pub const SPANISH_PLATE_REGEX: &str = r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$";

/// Broad European/alphanumeric fallback regex (4 to 10 uppercase alphanumeric characters).
pub const EU_GENERIC_PLATE_REGEX: &str = r"^[A-Z0-9]{4,10}$";

#[derive(Debug, Error)]
pub enum PostprocessError {
    #[error("Invalid regular expression '{pattern}': {source}")]
    InvalidRegex {
        pattern: String,
        #[source]
        source: regex::Error,
    },
}

/// Supported plate validation presets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlatePattern {
    /// Spanish modern format: `0000 BBB` (no vowels, Q, or Ñ).
    Spanish,
    /// Generic European alphanumeric format (`4..=10` chars).
    EuGeneric,
    /// User-supplied custom regex pattern.
    Custom(String),
}

impl PlatePattern {
    pub fn as_regex_str(&self) -> &str {
        match self {
            Self::Spanish => SPANISH_PLATE_REGEX,
            Self::EuGeneric => EU_GENERIC_PLATE_REGEX,
            Self::Custom(expr) => expr.as_str(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    pub pattern: PlatePattern,
    pub min_ocr_confidence: f32,
    pub enable_heuristics: bool,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            pattern: PlatePattern::Spanish,
            min_ocr_confidence: 0.50,
            enable_heuristics: true,
        }
    }
}

/// Bounding box in original image pixel coordinates.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BboxOutput {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Single recognized license plate result.
#[derive(Debug, Clone, Serialize)]
pub struct PlateDetectionOutput {
    pub plate: String,
    pub raw_text: String,
    pub valid_format: bool,
    pub heuristics_applied: bool,
    pub detection_confidence: f32,
    pub ocr_confidence: f32,
    pub bbox: BboxOutput,
}

/// Structured per-frame JSON output.
#[derive(Debug, Clone, Serialize)]
pub struct FrameOutput {
    pub frame_id: u64,
    pub timestamp_ms: u64,
    pub inference_ms: f64,
    pub detections: Vec<PlateDetectionOutput>,
}

/// Cleans, normalizes, corrects confusable characters, and validates OCR output.
pub struct PlatePostprocessor {
    config: PostprocessConfig,
    regex: Regex,
}

impl PlatePostprocessor {
    pub fn new(config: PostprocessConfig) -> Result<Self, PostprocessError> {
        let pattern_str = config.pattern.as_regex_str();
        let regex = Regex::new(pattern_str).map_err(|source| PostprocessError::InvalidRegex {
            pattern: pattern_str.to_string(),
            source,
        })?;

        Ok(Self { config, regex })
    }

    /// Strips whitespace/punctuation and converts ASCII characters to uppercase.
    pub fn clean_text(raw: &str) -> String {
        raw.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_uppercase())
            .collect()
    }

    /// Applies positional character-confusion heuristics for known plate formats.
    ///
    /// For Spanish `0000 BBB` (7 chars):
    /// - Positions `0..4` must be digits (`0-9`).
    /// - Positions `4..7` must be valid Spanish consonants (no vowels, `Q`, or `Ñ`).
    pub fn apply_heuristics(cleaned: &str, pattern: &PlatePattern) -> (String, bool) {
        if *pattern != PlatePattern::Spanish || cleaned.len() != 7 {
            return (cleaned.to_string(), false);
        }

        let mut corrected = String::with_capacity(7);
        let mut changed = false;

        for (idx, ch) in cleaned.chars().enumerate() {
            let mapped = if idx < 4 {
                to_digit_confusable(ch)
            } else {
                to_spanish_consonant_confusable(ch)
            };

            if mapped != ch {
                changed = true;
            }
            corrected.push(mapped);
        }

        (corrected, changed)
    }

    /// Formats a valid 7-character Spanish plate as `0000 BBB` for readability,
    /// or returns the normalized string as-is for other formats.
    pub fn format_display(&self, normalized: &str) -> String {
        if self.config.pattern == PlatePattern::Spanish
            && normalized.len() == 7
            && self.regex.is_match(normalized)
        {
            format!("{} {}", &normalized[..4], &normalized[4..])
        } else {
            normalized.to_string()
        }
    }

    /// Processes a raw OCR prediction and returns `None` if below confidence threshold.
    pub fn process(
        &self,
        raw_text: &str,
        ocr_confidence: f32,
        detection_confidence: f32,
        bbox: BboxOutput,
    ) -> Option<PlateDetectionOutput> {
        if ocr_confidence < self.config.min_ocr_confidence {
            return None;
        }

        let cleaned = Self::clean_text(raw_text);
        if cleaned.is_empty() {
            return None;
        }

        let (normalized, heuristics_applied) = if self.config.enable_heuristics {
            Self::apply_heuristics(&cleaned, &self.config.pattern)
        } else {
            (cleaned, false)
        };

        let valid_format = self.regex.is_match(&normalized);
        let display_plate = self.format_display(&normalized);

        Some(PlateDetectionOutput {
            plate: display_plate,
            raw_text: raw_text.to_string(),
            valid_format,
            heuristics_applied,
            detection_confidence,
            ocr_confidence,
            bbox,
        })
    }
}

fn to_digit_confusable(ch: char) -> char {
    match ch {
        'O' | 'Q' | 'D' => '0',
        'I' | 'L' => '1',
        'Z' => '2',
        'A' => '4',
        'S' => '5',
        'G' => '6',
        'T' => '7',
        'B' => '8',
        _ => ch,
    }
}

fn to_spanish_consonant_confusable(ch: char) -> char {
    match ch {
        // Spanish plates exclude vowels (including O) and Q, so 0 most likely corresponds to D.
        '0' => 'D',
        '1' => 'L',
        '2' => 'Z',
        '4' => 'H',
        '5' => 'S',
        '6' => 'G',
        '7' => 'T',
        '8' => 'B',
        '9' => 'P',
        _ => ch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spanish_plate_heuristics_and_validation() {
        let pp = PlatePostprocessor::new(PostprocessConfig::default())
            .expect("valid default Spanish regex");
        let bbox = BboxOutput {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 30.0,
        };

        let out = pp
            .process("12O4 8BC", 0.92, 0.88, bbox)
            .expect("should pass threshold");
        assert_eq!(out.plate, "1204 BBC");
        assert!(out.valid_format);
        assert!(out.heuristics_applied);

        let invalid_vowel = pp
            .process("1234 ABC", 0.95, 0.90, bbox)
            .expect("should pass threshold");
        assert!(!invalid_vowel.valid_format);
    }
}
