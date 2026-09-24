//! Canonical JSON: a deterministic text form used for hashing.
//!
//! Rules:
//! - Object keys are sorted by their UTF-8 bytes (Rust `str` ordering), at every depth.
//! - No insignificant whitespace.
//! - Strings use `serde_json` escaping; numbers use `serde_json` formatting
//!   (integers as integers, floats via the shortest round-trip representation).
//! - `-0.0` is written as `0.0`.
//! - Non-finite floats (NaN, infinities) are rejected with
//!   [`CanonicalJsonError::NonFinite`] instead of silently becoming `null`.
//!
//! Key ordering never depends on how the input map was built, so the result is the
//! same whether or not any crate in the build enables `serde_json/preserve_order`.

use serde::Serialize;
use serde_json::Value;

mod finite;

/// Error produced when a value has no canonical JSON form.
#[derive(Debug, thiserror::Error)]
pub enum CanonicalJsonError {
    /// The value could not be converted to JSON.
    #[error("value is not representable as JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A float was NaN or infinite.
    #[error("non-finite float at `{path}`")]
    NonFinite {
        /// Location of the float (`$` is the root).
        path: String,
    },
}

/// Serializes `value` to canonical JSON text, rejecting non-finite floats.
pub fn to_canonical_string<T: Serialize + ?Sized>(value: &T) -> Result<String, CanonicalJsonError> {
    let value = to_checked_value(value)?;
    Ok(canonical_value_string(&value))
}

/// Converts `value` to a [`Value`], rejecting non-finite floats (which
/// `serde_json::to_value` would silently turn into `null`).
pub fn to_checked_value<T: Serialize + ?Sized>(value: &T) -> Result<Value, CanonicalJsonError> {
    finite::check(value)?;
    Ok(serde_json::to_value(value)?)
}

/// Writes an already-built [`Value`] as canonical JSON text.
pub fn canonical_value_string(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

/// blake3 (hex) of the canonical JSON text of `value`.
pub fn canonical_hash<T: Serialize + ?Sized>(value: &T) -> Result<String, CanonicalJsonError> {
    Ok(crate::blake3_hex(to_canonical_string(value)?.as_bytes()))
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_f64() && n.as_f64() == Some(0.0) {
                // Covers -0.0, which compares equal to 0.0.
                out.push_str("0.0");
            } else {
                out.push_str(&n.to_string());
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            out.push('{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_value(v, out);
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    // Serializing a &str to JSON cannot fail; fall back to a manual escape only to
    // avoid a panic path.
    match serde_json::to_string(s) {
        Ok(escaped) => out.push_str(&escaped),
        Err(_) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    #[test]
    fn sorts_keys_and_strips_whitespace() {
        let v = json!({"b": 1, "a": {"z": [1, 2, {"y": null, "x": true}], "c": "s"}});
        assert_eq!(
            canonical_value_string(&v),
            r#"{"a":{"c":"s","z":[1,2,{"x":true,"y":null}]},"b":1}"#
        );
    }

    #[test]
    fn deterministic_across_insertion_order() {
        let a: Value =
            serde_json::from_str(r#"{ "k1": 1, "k2": [ 1.5, "x" ], "k3": {"b":2,"a":1} }"#)
                .unwrap_or(Value::Null);
        let b: Value = serde_json::from_str(r#"{"k3":{"a":1,"b":2},"k2":[1.5,"x"],"k1":1}"#)
            .unwrap_or(Value::Null);
        assert_ne!(a, Value::Null);
        assert_eq!(canonical_value_string(&a), canonical_value_string(&b));
        for _ in 0..10 {
            assert_eq!(canonical_value_string(&a), canonical_value_string(&a));
        }
    }

    #[test]
    fn escapes_strings() {
        let v = json!({"q\"k": "line\nbreak\u{1}"});
        let s = canonical_value_string(&v);
        assert_eq!(s, r#"{"q\"k":"line\nbreak\u0001"}"#);
        let back: Value = serde_json::from_str(&s).unwrap_or(Value::Null);
        assert_eq!(back, v);
    }

    #[test]
    fn integers_and_floats_are_distinct() {
        assert_ne!(
            canonical_value_string(&json!({"x": 1})),
            canonical_value_string(&json!({"x": 1.0}))
        );
    }

    fn arb_json() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map(Value::Bool),
            any::<i64>().prop_map(|n| json!(n)),
            any::<u64>().prop_map(|n| json!(n)),
            any::<f64>()
                .prop_filter("finite", |f| f.is_finite())
                .prop_map(|f| json!(f)),
            ".{0,12}".prop_map(Value::String),
        ];
        leaf.prop_recursive(4, 48, 6, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..6).prop_map(Value::Array),
                prop::collection::vec((".{0,8}", inner), 0..6)
                    .prop_map(|kvs| { Value::Object(kvs.into_iter().collect()) }),
            ]
        })
    }

    /// Writes JSON with keys in reverse-sorted order and extra whitespace: a valid but
    /// non-canonical rendering of the same value.
    fn scrambled(value: &Value, out: &mut String) {
        match value {
            Value::Array(items) => {
                out.push_str("[ ");
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(" ,\n ");
                    }
                    scrambled(item, out);
                }
                out.push_str(" ]");
            }
            Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by(|a, b| b.0.cmp(a.0));
                out.push_str("{\n");
                for (i, (k, v)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\t");
                    }
                    write_string(k, out);
                    out.push_str(" : ");
                    scrambled(v, out);
                }
                out.push_str("\n}");
            }
            other => write_value(other, out),
        }
    }

    #[derive(Serialize)]
    struct Params {
        name: String,
        values: Vec<f64>,
        nested: Option<Box<Params>>,
    }

    #[test]
    fn non_finite_floats_rejected_with_path() {
        let p = Params {
            name: "a".into(),
            values: vec![1.0],
            nested: Some(Box::new(Params {
                name: "b".into(),
                values: vec![0.5, f64::NAN],
                nested: None,
            })),
        };
        match to_canonical_string(&p) {
            Err(CanonicalJsonError::NonFinite { path }) => assert_eq!(path, "$.nested.values[1]"),
            other => panic!("expected NonFinite, got {other:?}"),
        }
        assert!(to_canonical_string(&f64::INFINITY).is_err());
        assert!(
            to_canonical_string(&std::collections::BTreeMap::from([(
                "k",
                f32::NEG_INFINITY
            )]))
            .is_err()
        );
    }

    #[test]
    fn negative_zero_normalized() {
        assert_eq!(to_canonical_string(&-0.0f64).unwrap(), "0.0");
        assert_eq!(
            canonical_value_string(&json!({"x": -0.0})),
            canonical_value_string(&json!({"x": 0.0}))
        );
    }

    proptest! {
        #[test]
        fn non_finite_always_rejected(
            finite in any::<f64>().prop_filter("finite", |f| f.is_finite()),
            bad in prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
            pos in 0usize..4,
        ) {
            let mut values = vec![finite; 4];
            values[pos] = bad;
            let p = Params { name: "x".into(), values, nested: None };
            let rejected = matches!(to_canonical_string(&p), Err(CanonicalJsonError::NonFinite { .. }));
            prop_assert!(rejected);
        }

        #[test]
        fn finite_floats_accepted_and_sign_of_zero_ignored(f in any::<f64>().prop_filter("finite", |f| f.is_finite())) {
            let text = to_canonical_string(&f).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let back: f64 = serde_json::from_str(&text).map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(back, f);
            prop_assert_eq!(to_canonical_string(&(f * 0.0)).ok(), to_canonical_string(&0.0f64).ok());
        }

        #[test]
        fn round_trip_is_invariant(v in arb_json()) {
            let text = canonical_value_string(&v);
            let parsed: Value = serde_json::from_str(&text).map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(&parsed, &v);
            prop_assert_eq!(canonical_value_string(&parsed), text);
        }

        #[test]
        fn key_order_and_whitespace_independent(v in arb_json()) {
            let mut messy = String::new();
            scrambled(&v, &mut messy);
            let parsed: Value = serde_json::from_str(&messy).map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(canonical_value_string(&parsed), canonical_value_string(&v));
        }
    }
}
