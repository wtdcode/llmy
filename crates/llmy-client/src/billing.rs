use std::collections::BTreeMap;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::model::OpenAIModel;
use llmy_types::error::LLMYError;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Display for TokenUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Usage(inputs={}({}/{:.2}% cached), outputs={}({}/{:.2}% reasoning))",
            self.input_tokens,
            self.cache_tokens,
            if self.input_tokens == 0 {
                0.0f64
            } else {
                100f64 * self.cache_tokens as f64 / self.input_tokens as f64
            },
            self.output_tokens,
            self.reasoning_tokens,
            if self.output_tokens == 0 {
                0.0f64
            } else {
                100f64 * self.reasoning_tokens as f64 / self.output_tokens as f64
            },
        ))
    }
}

impl TokenUsage {
    pub fn input_cost(&self, model: &OpenAIModel) -> Decimal {
        let pricing = model.pricing();
        let input_without_cache = self.input_tokens.saturating_sub(self.cache_tokens);
        // A model without a dedicated cache-read price bills cached tokens at the
        // normal input rate (not free) — matches the original billing logic.
        let cache_price = pricing.input_cache_read.unwrap_or(pricing.input);
        Decimal::from(input_without_cache) * pricing.input
            + Decimal::from(self.cache_tokens) * cache_price
    }
    pub fn output_cost(&self, model: &OpenAIModel) -> Decimal {
        let pricing = model.pricing();
        Decimal::from(self.output_tokens) * pricing.output
    }

    pub fn cost(&self, model: &OpenAIModel) -> Decimal {
        self.input_cost(model) + self.output_cost(model)
    }

    pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (input_tokens, o1) = self.input_tokens.overflowing_add(rhs.input_tokens);
        let (output_tokens, o2) = self.output_tokens.overflowing_add(rhs.output_tokens);
        let (cache_tokens, o3) = self.cache_tokens.overflowing_add(rhs.cache_tokens);
        let (reasoning_tokens, o4) = self.reasoning_tokens.overflowing_add(rhs.reasoning_tokens);
        (
            Self {
                input_tokens,
                output_tokens,
                cache_tokens,
                reasoning_tokens,
            },
            o1 || o2 || o3 || o4,
        )
    }

    /// Field-wise overflowing subtract. The returned flag is `true` if *any*
    /// field underflowed; the value carries the wrapped result regardless.
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let (input_tokens, o1) = self.input_tokens.overflowing_sub(rhs.input_tokens);
        let (output_tokens, o2) = self.output_tokens.overflowing_sub(rhs.output_tokens);
        let (cache_tokens, o3) = self.cache_tokens.overflowing_sub(rhs.cache_tokens);
        let (reasoning_tokens, o4) = self.reasoning_tokens.overflowing_sub(rhs.reasoning_tokens);
        (
            Self {
                input_tokens,
                output_tokens,
                cache_tokens,
                reasoning_tokens,
            },
            o1 || o2 || o3 || o4,
        )
    }

    /// Field-wise wrapping add (each field wraps on overflow).
    pub fn wrapping_add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }

    /// Field-wise wrapping subtract (each field wraps on underflow).
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    /// Field-wise saturating add (each field clamps to `u64::MAX`).
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_add(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_add(rhs.output_tokens),
            cache_tokens: self.cache_tokens.saturating_add(rhs.cache_tokens),
            reasoning_tokens: self.reasoning_tokens.saturating_add(rhs.reasoning_tokens),
        }
    }

    /// Field-wise saturating subtract (each field clamps to `0`).
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_sub(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_sub(rhs.output_tokens),
            cache_tokens: self.cache_tokens.saturating_sub(rhs.cache_tokens),
            reasoning_tokens: self.reasoning_tokens.saturating_sub(rhs.reasoning_tokens),
        }
    }
}

impl Add for TokenUsage {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (sum, overflowed) = self.overflowing_add(rhs);
        assert!(!overflowed, "TokenUsage add overflowed: {self} + {rhs}");
        sum
    }
}

impl Sub for TokenUsage {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let (diff, overflowed) = self.overflowing_sub(rhs);
        assert!(!overflowed, "TokenUsage sub underflowed: {self} - {rhs}");
        diff
    }
}

impl AddAssign for TokenUsage {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for TokenUsage {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// A flat snapshot of the **root** of a [`BillingTree`]: total token usage, the
/// accumulated spend, and the budget cap. This is the cross-cutting "whole LLM"
/// view; per-scope numbers come from [`BillingTree::node_snapshot`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TokenBilling {
    pub tokens: TokenUsage,
    pub current: Decimal,
    pub cap: Decimal,
}

impl Display for TokenBilling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Billing(budget={:.4}/{}, usage={})",
            self.current, self.cap, self.tokens
        ))
    }
}

impl TokenBilling {
    pub fn new(cap: Decimal) -> Self {
        Self {
            tokens: TokenUsage::default(),
            current: Decimal::ZERO,
            cap,
        }
    }

    pub fn in_cap(&self) -> bool {
        self.current <= self.cap
    }
}

/// Identifier of a node in a [`BillingTree`]. Monotonic and stable — removing a
/// node never invalidates other ids (unlike a `Vec` index).
pub type NodeId = u64;

/// The always-present root node; carries the grand total and the budget cap.
pub const ROOT: NodeId = 0;

#[derive(Debug, Clone)]
struct BillingNode {
    name: Option<String>,
    parent: Option<NodeId>,
    /// This scope's accumulated usage, spend, and budget. The budget is always
    /// concrete: root holds the global cap, and a child is capped at its requested
    /// cap clamped to (or, if `None`, equal to) the parent's remaining budget at
    /// creation time.
    billing: TokenBilling,
}

/// A tree of [`TokenUsage`] accumulators (one node per scope) plus a flat
/// per-`debug_prefix` breakdown. Every billed call bubbles its delta from the
/// current node up to [`ROOT`], so the root cap can never be bypassed by scoping
/// — even a stale/removed scope still hits root. Every scope additionally has its
/// own cap (bounded by its parent's remaining budget), checked along the way.
#[derive(Debug)]
pub struct BillingTree {
    nodes: BTreeMap<NodeId, BillingNode>,
    next_id: NodeId,
    by_prefix: BTreeMap<String, TokenUsage>,
}

impl Drop for BillingTree {
    fn drop(&mut self) {
        // Fires exactly when the tree is torn down — i.e. the last handle to the
        // whole LLM is gone. That's the safe moment to log the root/session total
        // (root is a permanent sentinel and is never `remove`d, so it's still here).
        if let Some(root) = self.nodes.get(&ROOT) {
            tracing::info!("billing session total: {}", root.billing);
        }
        if !self.by_prefix.is_empty() {
            let summary = self
                .by_prefix
                .iter()
                .map(|(prefix, usage)| {
                    let prefix = if prefix.is_empty() { "(none)" } else { prefix };
                    format!("{prefix}: {usage}")
                })
                .collect::<Vec<_>>()
                .join(", ");
            tracing::debug!("billing summary by prefix: {summary}");
        }
    }
}

impl BillingTree {
    pub fn new(cap: Decimal) -> Self {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            ROOT,
            BillingNode {
                name: Some("root".to_string()),
                parent: None,
                billing: TokenBilling::new(cap), // root carries the global budget
            },
        );
        Self {
            nodes,
            next_id: ROOT,
            by_prefix: BTreeMap::new(),
        }
    }

    /// Open a child scope under `parent` and return its fresh id. The child's cap
    /// is its requested cap clamped to the parent's remaining budget
    /// (`parent.cap - parent.cost`); a `None` request takes exactly that
    /// remaining budget. So a scope can never be granted more headroom than its
    /// parent has left.
    pub fn alloc_child(
        &mut self,
        parent: NodeId,
        name: Option<String>,
        cap: Option<Decimal>,
    ) -> NodeId {
        let parent = if self.nodes.contains_key(&parent) {
            parent
        } else {
            ROOT
        };
        let remaining = {
            let p = self.nodes.get(&parent).expect("parent or root present");
            (p.billing.cap - p.billing.current).max(Decimal::ZERO)
        };
        let cap = cap.map(|c| c.min(remaining)).unwrap_or(remaining);

        self.next_id += 1;
        let id = self.next_id;
        self.nodes.insert(
            id,
            BillingNode {
                name,
                parent: Some(parent),
                billing: TokenBilling::new(cap),
            },
        );
        id
    }

    /// Record one call's `delta` against scope `node` (bubbling to root) and the
    /// global `prefix` bucket. Returns `Err` if any scope on the path (including
    /// root) exceeds its cap — reporting the deepest such scope.
    pub fn record(
        &mut self,
        node: NodeId,
        prefix: Option<&str>,
        model: &OpenAIModel,
        delta: TokenUsage,
    ) -> Result<(), LLMYError> {
        let dcost = delta.cost(model);

        // Dimension 1: bubble from the current node up the still-living chain,
        // remembering the first (deepest) scope whose cap is exceeded.
        let mut cur = Some(node);
        let mut hit_root = false;
        let mut violation: Option<LLMYError> = None;
        while let Some(id) = cur {
            let Some(n) = self.nodes.get_mut(&id) else {
                break; // stale handle to a removed scope
            };
            n.billing.tokens += delta;
            n.billing.current += dcost;
            hit_root |= id == ROOT;
            if violation.is_none() && !n.billing.in_cap() {
                violation = Some(LLMYError::Billing {
                    cap: n.billing.cap,
                    current: n.billing.current,
                    node: id,
                    scope: n.name.clone(),
                });
            }
            cur = n.parent;
        }
        // Cap safety: root is billed and checked no matter what (even if the
        // starting `node` was removed and the chain broke before reaching root).
        if !hit_root {
            let root = self.nodes.get_mut(&ROOT).expect("root node always present");
            root.billing.tokens += delta;
            root.billing.current += dcost;
            if violation.is_none() && !root.billing.in_cap() {
                violation = Some(LLMYError::Billing {
                    cap: root.billing.cap,
                    current: root.billing.current,
                    node: ROOT,
                    scope: root.name.clone(),
                });
            }
        }

        // Dimension 2: flat per-debug_prefix breakdown (None => "").
        *self
            .by_prefix
            .entry(prefix.unwrap_or("").to_string())
            .or_default() += delta;

        match violation {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Pre-flight cap check: returns `Err` if scope `node`, any ancestor, or the
    /// root is *already* over budget — without recording anything. Lets callers
    /// skip a request when the cap is already blown (reports the deepest scope).
    pub fn check_cap(&self, node: NodeId) -> Result<(), LLMYError> {
        let mut cur = Some(node);
        let mut saw_root = false;
        while let Some(id) = cur {
            let Some(n) = self.nodes.get(&id) else {
                break; // stale handle to a removed scope
            };
            if !n.billing.in_cap() {
                return Err(LLMYError::Billing {
                    cap: n.billing.cap,
                    current: n.billing.current,
                    node: id,
                    scope: n.name.clone(),
                });
            }
            saw_root |= id == ROOT;
            cur = n.parent;
        }
        if !saw_root {
            let root = self.nodes.get(&ROOT).expect("root node always present");
            if !root.billing.in_cap() {
                return Err(LLMYError::Billing {
                    cap: root.billing.cap,
                    current: root.billing.current,
                    node: ROOT,
                    scope: root.name.clone(),
                });
            }
        }
        Ok(())
    }

    /// Remove a scope, logging its final usage. Its children are re-parented to
    /// its parent so the chain to root stays intact; the node's totals already
    /// live in its ancestors.
    ///
    /// Root is a permanent sentinel — it anchors bubbling, the cap check, and
    /// every snapshot (all of which assume it exists), and a root handle is not
    /// guaranteed to outlive the scopes beneath it. So root is never removed
    /// here; it is freed only when the whole tree is dropped (see `Drop`).
    pub fn remove(&mut self, node: NodeId) {
        if node == ROOT {
            return;
        }
        let parent = match self.nodes.get(&node) {
            Some(n) => {
                tracing::info!("scope {:?} (#{}) {}", n.name, node, n.billing);
                n.parent.unwrap_or(ROOT)
            }
            None => return,
        };
        for n in self.nodes.values_mut() {
            if n.parent == Some(node) {
                n.parent = Some(parent);
            }
        }
        self.nodes.remove(&node);
    }

    /// Snapshot of one scope: its accumulated usage and spend, plus its cap. An
    /// unknown id yields an empty snapshot bounded by the root cap.
    pub fn node_snapshot(&self, node: NodeId) -> TokenBilling {
        match self.nodes.get(&node) {
            Some(n) => n.billing,
            None => TokenBilling::new(self.nodes[&ROOT].billing.cap),
        }
    }

    /// Flat snapshot of the root: grand-total usage, spend, and the global cap.
    pub fn root_snapshot(&self) -> TokenBilling {
        self.node_snapshot(ROOT)
    }

    /// The global budget cap (the root scope's cap).
    pub fn cap(&self) -> Decimal {
        self.nodes[&ROOT].billing.cap
    }

    /// The whole per-`debug_prefix` breakdown (`""` = calls with no prefix).
    pub fn usage_by_prefix(&self) -> BTreeMap<String, TokenUsage> {
        self.by_prefix.clone()
    }

    /// Usage for a single `debug_prefix` bucket.
    pub fn usage_for_prefix(&self, prefix: &str) -> TokenUsage {
        self.by_prefix.get(prefix).copied().unwrap_or_default()
    }

    /// Number of live nodes (including root). Test-only, for asserting pruning.
    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, output: u64, cache: u64, reasoning: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_tokens: cache,
            reasoning_tokens: reasoning,
        }
    }

    #[test]
    fn input_cost_without_cache_price_bills_cache_at_input_rate() {
        use crate::model::OpenAIModel;
        use std::str::FromStr;
        // "name,input,output" (per-1M USD) => no dedicated cache-read price.
        let model = OpenAIModel::from_str("billing-test,2,4").unwrap();
        // 100 input tokens, 40 of them cached. Cached tokens must NOT be free:
        // (60 + 40) * 2e-6 = 0.0002.
        let u = usage(100, 0, 40, 0);
        assert_eq!(u.input_cost(&model), rust_decimal::dec!(0.0002));
    }

    #[test]
    fn input_cost_with_cache_price_uses_cache_rate() {
        use crate::model::OpenAIModel;
        use std::str::FromStr;
        // "name,input,output,cache_read" => cached tokens priced at 1e-6.
        let model = OpenAIModel::from_str("billing-test,2,4,1").unwrap();
        // 60 * 2e-6 + 40 * 1e-6 = 0.00016.
        let u = usage(100, 0, 40, 0);
        assert_eq!(u.input_cost(&model), rust_decimal::dec!(0.00016));
    }

    #[test]
    fn token_usage_add_is_field_wise() {
        let a = usage(100, 40, 20, 5);
        let b = usage(30, 10, 8, 2);
        assert_eq!(a + b, usage(130, 50, 28, 7));
        // The diff is the inverse of the sum.
        assert_eq!((a + b) - b, a);
    }

    #[test]
    #[should_panic(expected = "TokenUsage sub underflowed")]
    fn token_usage_sub_panics_on_underflow() {
        let _ = usage(10, 0, 0, 0) - usage(50, 3, 1, 1);
    }

    #[test]
    #[should_panic(expected = "TokenUsage add overflowed")]
    fn token_usage_add_panics_on_overflow() {
        let _ = usage(u64::MAX, 0, 0, 0) + usage(1, 0, 0, 0);
    }

    #[test]
    fn token_usage_saturating_clamps_per_field() {
        // Underflowing fields clamp to 0, the rest subtract normally.
        assert_eq!(
            usage(10, 100, 0, 5).saturating_sub(usage(50, 30, 1, 2)),
            usage(0, 70, 0, 3)
        );
        // Overflowing fields clamp to u64::MAX.
        assert_eq!(
            usage(u64::MAX, 1, 0, 0).saturating_add(usage(1, 1, 0, 0)),
            usage(u64::MAX, 2, 0, 0)
        );
    }

    #[test]
    fn token_usage_overflowing_and_wrapping_agree_with_fields() {
        let (wrapped, overflowed) = usage(u64::MAX, 0, 0, 0).overflowing_add(usage(1, 0, 0, 0));
        assert!(overflowed);
        assert_eq!(wrapped, usage(0, 0, 0, 0));
        assert_eq!(
            usage(u64::MAX, 5, 0, 0).wrapping_add(usage(1, 0, 0, 0)),
            usage(0, 5, 0, 0)
        );

        // No out-of-range field ⇒ flag is false and the value matches `+`.
        let a = usage(7, 9, 2, 1);
        let b = usage(3, 4, 1, 1);
        assert_eq!(a.overflowing_add(b), (a + b, false));
    }

    // --- BillingTree -------------------------------------------------------

    /// Custom pricing "1,000,000 per-1M" => $1 per token, for easy cap math.
    fn model_1usd() -> OpenAIModel {
        use std::str::FromStr;
        OpenAIModel::from_str("captest,1000000,1000000").unwrap()
    }

    fn input(n: u64) -> TokenUsage {
        usage(n, 0, 0, 0)
    }

    #[test]
    fn record_bubbles_child_up_to_root() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(1000));
        let stage = tree.alloc_child(ROOT, Some("stage".into()), None);
        let sub = tree.alloc_child(stage, Some("sub".into()), None);

        tree.record(sub, None, &model, input(5)).unwrap();

        assert_eq!(tree.node_snapshot(sub).tokens.input_tokens, 5);
        assert_eq!(tree.node_snapshot(stage).tokens.input_tokens, 5);
        assert_eq!(tree.node_snapshot(ROOT).tokens.input_tokens, 5);
        assert_eq!(tree.root_snapshot().current, rust_decimal::dec!(5));
    }

    #[test]
    fn record_tracks_prefix_globally() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(1000));
        let a = tree.alloc_child(ROOT, None, None);

        tree.record(a, Some("planner"), &model, input(3)).unwrap();
        tree.record(ROOT, None, &model, input(2)).unwrap(); // no prefix => ""

        assert_eq!(tree.usage_for_prefix("planner").input_tokens, 3);
        assert_eq!(tree.usage_for_prefix("").input_tokens, 2);
    }

    #[test]
    fn scope_cap_errors_with_scope_identity_and_still_bills_root() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(1000)); // generous root
        let sub = tree.alloc_child(ROOT, Some("limited".into()), Some(rust_decimal::dec!(10)));

        tree.record(sub, None, &model, input(8)).unwrap(); // 8 <= 10
        let err = tree.record(sub, None, &model, input(5)).unwrap_err(); // 13 > 10

        match err {
            LLMYError::Billing {
                cap,
                current,
                node,
                scope,
            } => {
                assert_eq!(cap, rust_decimal::dec!(10));
                assert_eq!(current, rust_decimal::dec!(13));
                assert_eq!(node, sub);
                assert_eq!(scope.as_deref(), Some("limited"));
            }
            other => panic!("expected Billing, got {other:?}"),
        }
        // Cap safety: root is billed even though the scope cap was hit.
        assert_eq!(tree.node_snapshot(ROOT).tokens.input_tokens, 13);
    }

    #[test]
    fn root_cap_enforced_on_direct_spend() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(10)); // tight root
        let err = tree.record(ROOT, None, &model, input(11)).unwrap_err();
        match err {
            LLMYError::Billing { node, cap, .. } => {
                assert_eq!(node, ROOT);
                assert_eq!(cap, rust_decimal::dec!(10));
            }
            other => panic!("expected Billing, got {other:?}"),
        }
    }

    #[test]
    fn unrequested_child_cap_inherits_parent_remaining() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(10));
        let sub = tree.alloc_child(ROOT, Some("s".into()), None);

        // A `None` cap takes exactly the parent's remaining budget (10 - 0).
        assert_eq!(tree.node_snapshot(sub).cap, rust_decimal::dec!(10));
        // So the child itself is the (deepest) scope that blows the budget.
        let err = tree.record(sub, None, &model, input(11)).unwrap_err();
        match err {
            LLMYError::Billing { node, cap, .. } => {
                assert_eq!(node, sub);
                assert_eq!(cap, rust_decimal::dec!(10));
            }
            other => panic!("expected Billing, got {other:?}"),
        }
    }

    #[test]
    fn child_cap_is_clamped_to_parent_remaining() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(100));
        tree.record(ROOT, None, &model, input(70)).unwrap(); // root spent 70, 30 left

        // Requested 50 but only 30 remains => clamped to 30.
        let a = tree.alloc_child(ROOT, Some("a".into()), Some(rust_decimal::dec!(50)));
        assert_eq!(tree.node_snapshot(a).cap, rust_decimal::dec!(30));
        // None => exactly the 30 remaining.
        let b = tree.alloc_child(ROOT, Some("b".into()), None);
        assert_eq!(tree.node_snapshot(b).cap, rust_decimal::dec!(30));
    }

    #[test]
    fn remove_reparents_children_and_preserves_aggregates() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(1000));
        let stage = tree.alloc_child(ROOT, Some("stage".into()), None);
        let sub = tree.alloc_child(stage, Some("sub".into()), None);

        tree.record(sub, None, &model, input(4)).unwrap();
        tree.remove(stage); // sub re-parents to root
        tree.record(sub, None, &model, input(6)).unwrap();

        assert_eq!(tree.node_snapshot(sub).tokens.input_tokens, 10);
        assert_eq!(tree.node_snapshot(ROOT).tokens.input_tokens, 10);
    }

    #[test]
    fn check_cap_flags_already_over_budget_without_recording() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(10));
        let sub = tree.alloc_child(ROOT, Some("limited".into()), Some(rust_decimal::dec!(3)));

        // Within budget so far: pre-flight is clear.
        tree.record(sub, None, &model, input(3)).unwrap();
        assert!(tree.check_cap(sub).is_ok());

        // Push the scope over its own cap.
        let _ = tree.record(sub, None, &model, input(1)).unwrap_err();

        // Pre-flight now refuses, naming the offending scope, and records nothing.
        let before = tree.node_snapshot(sub).tokens;
        match tree.check_cap(sub).unwrap_err() {
            LLMYError::Billing { node, scope, .. } => {
                assert_eq!(node, sub);
                assert_eq!(scope.as_deref(), Some("limited"));
            }
            other => panic!("expected Billing, got {other:?}"),
        }
        assert_eq!(tree.node_snapshot(sub).tokens, before); // check_cap is read-only
    }

    #[test]
    fn stale_node_after_remove_still_bills_root() {
        let model = model_1usd();
        let mut tree = BillingTree::new(rust_decimal::dec!(1000));
        let sub = tree.alloc_child(ROOT, Some("sub".into()), None);
        tree.remove(sub);

        // Recording through a handle whose scope was removed must not lose the
        // tokens: root is still billed so the cap can't be bypassed.
        tree.record(sub, None, &model, input(7)).unwrap();
        assert_eq!(tree.node_snapshot(ROOT).tokens.input_tokens, 7);
    }
}
