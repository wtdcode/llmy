use crate::model::OpenAIModel;

enum EstimationFactor {
    Model(OpenAIModel),
    Fixed(f64),
}

impl EstimationFactor {
    pub fn count_tokens(&self, text: &str) -> usize {
        match self {
            Self::Model(model) => model.config.count_tokens(text).unwrap(),
            Self::Fixed(factor) => (text.len() as f64 * (*factor)) as _,
        }
    }
}

/// Splits a large text into chunks that each fit within a token budget.
/// Maintains a byte-offset cursor tracking how much has been consumed.
pub struct TokenCursor {
    text: String,
    offset: usize,
    factor: EstimationFactor,
}

impl TokenCursor {
    pub fn new(text: String, model: OpenAIModel) -> Option<Self> {
        if model.config.encoding().is_none() {
            return None;
        }
        Some(Self {
            text,
            offset: 0,
            factor: EstimationFactor::Model(model),
        })
    }

    pub fn new_with_estimation_factor(text: String, factor: f64) -> Self {
        Self {
            text,
            offset: 0,
            factor: EstimationFactor::Fixed(factor),
        }
    }

    pub fn is_done(&self) -> bool {
        self.offset >= self.text.len()
    }

    /// Take the next slice that fits within `token_budget` tokens.
    /// Returns `None` if all text has been consumed.
    pub fn next_chunk(&mut self, token_budget: usize) -> Option<&str> {
        if self.is_done() {
            return None;
        }
        let remaining = &self.text[self.offset..];
        let remaining_tokens = self.factor.count_tokens(remaining);

        if remaining_tokens <= token_budget {
            self.offset = self.text.len();
            return Some(remaining);
        }

        // Binary search for the largest char-aligned byte offset that fits the token budget
        let mut lo: usize = 1;
        let mut hi: usize = remaining.len();
        let mut best = 0usize;

        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let boundary = char_boundary_before(remaining, mid);
            let tokens = self.factor.count_tokens(&remaining[..boundary]);

            if tokens <= token_budget {
                best = best.max(boundary);
                lo = mid + 1;
            } else {
                hi = mid.saturating_sub(1);
            }
        }

        if best == 0 {
            // Fallback: take at least some content even if it exceeds the budget
            let fallback = char_boundary_before(remaining, remaining.len().min(4096));
            self.offset += fallback;
            return Some(&self.text[self.offset - fallback..self.offset]);
        }

        let start = self.offset;
        self.offset += best;
        Some(&self.text[start..self.offset])
    }
}

/// Find the largest valid char boundary at or before `byte_offset`.
fn char_boundary_before(s: &str, byte_offset: usize) -> usize {
    let pos = byte_offset.min(s.len());
    (0..=pos)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: collect all chunks from a cursor with the given budget.
    fn collect_chunks(mut cursor: TokenCursor, budget: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        while let Some(chunk) = cursor.next_chunk(budget) {
            chunks.push(chunk.to_string());
        }
        chunks
    }

    #[test]
    fn small_text_fits_in_one_chunk() {
        // factor=1.0 means 1 token per byte
        let cursor = TokenCursor::new_with_estimation_factor("hello world".into(), 1.0);
        let chunks = collect_chunks(cursor, 100);
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn text_split_into_multiple_chunks() {
        // 20 bytes, factor=1.0, budget=10 tokens -> should split into 2 chunks
        let text = "aaaaaaaaaabbbbbbbbbb"; // 20 bytes
        let cursor = TokenCursor::new_with_estimation_factor(text.into(), 1.0);
        let chunks = collect_chunks(cursor, 10);
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );
        let reassembled: String = chunks.concat();
        assert_eq!(reassembled, text);
    }

    #[test]
    fn chunks_reassemble_to_original() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let cursor = TokenCursor::new_with_estimation_factor(text.clone(), 1.0);
        let chunks = collect_chunks(cursor, 50);
        let reassembled: String = chunks.concat();
        assert_eq!(reassembled, text);
    }

    #[test]
    fn each_chunk_respects_token_budget() {
        let text = "abcdefghij".repeat(100); // 1000 bytes
        let budget = 100;
        let factor = 1.0;
        let cursor = TokenCursor::new_with_estimation_factor(text.clone(), factor);
        let chunks = collect_chunks(cursor, budget);
        for (i, chunk) in chunks.iter().enumerate() {
            let tokens = (chunk.len() as f64 * factor) as usize;
            assert!(
                tokens <= budget,
                "chunk {i} has {tokens} tokens, exceeds budget {budget}"
            );
        }
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn multibyte_chars_do_not_panic() {
        // Each '你' is 3 bytes; ensure we never split mid-character
        let text = "你好世界".repeat(50); // 600 bytes
        let cursor = TokenCursor::new_with_estimation_factor(text.clone(), 1.0);
        let chunks = collect_chunks(cursor, 30);
        let reassembled: String = chunks.concat();
        assert_eq!(reassembled, text);
        for chunk in &chunks {
            assert!(chunk.is_char_boundary(0));
            assert!(chunk.is_char_boundary(chunk.len()));
        }
    }

    #[test]
    fn empty_text_returns_none() {
        let mut cursor = TokenCursor::new_with_estimation_factor(String::new(), 1.0);
        assert!(cursor.next_chunk(100).is_none());
        assert!(cursor.is_done());
    }

    #[test]
    fn exact_budget_fits_in_one_chunk() {
        // 10 bytes with factor=1.0, budget=10 -> fits exactly
        let text = "0123456789";
        let cursor = TokenCursor::new_with_estimation_factor(text.into(), 1.0);
        let chunks = collect_chunks(cursor, 10);
        assert_eq!(chunks, vec!["0123456789"]);
    }

    #[test]
    fn char_boundary_before_basic() {
        assert_eq!(char_boundary_before("hello", 3), 3);
        assert_eq!(char_boundary_before("hello", 100), 5);
        assert_eq!(char_boundary_before("hello", 0), 0);
        // '你' is 3 bytes; byte offset 1 is mid-char, should snap back to 0
        assert_eq!(char_boundary_before("你好", 1), 0);
        assert_eq!(char_boundary_before("你好", 3), 3);
    }
}
