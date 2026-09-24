//! Output schemas: schemars generation, `jsonschema` validation, and typed decoding.
//!
//! Validation is two-stage: the JSON document is first checked against the
//! schemars-generated schema, then deserialized into the caller's type through
//! `serde_path_to_error` so that type-level failures (for example unknown fields
//! rejected by `deny_unknown_fields`) also come back with a precise path.

use std::fmt;
use std::sync::Arc;

use schemars::generate::SchemaSettings;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{FieldError, Result, VisionError};

type TypedCheck = dyn Fn(&Value) -> std::result::Result<(), Vec<FieldError>> + Send + Sync;

/// The JSON schema for one output type, with a compiled validator.
#[derive(Clone)]
pub struct OutputSchema {
    name: String,
    schema: Value,
    validator: Arc<jsonschema::Validator>,
    typed_check: Arc<TypedCheck>,
}

impl fmt::Debug for OutputSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputSchema")
            .field("name", &self.name)
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

/// Generate the JSON schema for `T` with every subschema inlined.
///
/// Inlining avoids `$ref`/`$defs`, which keeps the schema inside the subset that
/// Ollama's grammar conversion handles reliably.
pub fn schema_value_for<T: JsonSchema>() -> Value {
    let generator = SchemaSettings::draft07()
        .with(|s| {
            s.inline_subschemas = true;
        })
        .into_generator();
    generator.into_root_schema_for::<T>().to_value()
}

impl OutputSchema {
    /// Build the schema and validators for `T`.
    pub fn for_type<T>() -> Result<Self>
    where
        T: JsonSchema + DeserializeOwned + 'static,
    {
        let schema = schema_value_for::<T>();
        let validator = jsonschema::validator_for(&schema)
            .map_err(|e| VisionError::Config(format!("generated schema is invalid: {e}")))?;
        let typed_check: Arc<TypedCheck> =
            Arc::new(|value: &Value| decode_value::<T>(value).map(|_| ()));
        Ok(Self {
            name: T::schema_name().into_owned(),
            schema,
            validator: Arc::new(validator),
            typed_check,
        })
    }

    /// Name of the output type.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Full schema document (including `$schema`).
    pub fn json(&self) -> &Value {
        &self.schema
    }

    /// Schema as sent in Ollama's `format` field (the `$schema` key removed).
    pub fn format_value(&self) -> Value {
        let mut v = self.schema.clone();
        if let Value::Object(map) = &mut v {
            map.remove("$schema");
        }
        v
    }

    /// Compact schema text for inclusion in the prompt.
    pub fn text(&self) -> String {
        self.format_value().to_string()
    }

    /// Schema as sent to the server's grammar constraint: [`Self::format_value`]
    /// without annotation keywords (`title`, `description`, `format`, `examples`,
    /// `default`). Annotations never constrain output, and dropping them keeps the
    /// grammar the server builds to the structural keywords (types, enums,
    /// required properties, bounded arrays). Property names are never touched.
    /// The prompt keeps [`Self::text`], so prompts and replay keys do not change.
    pub fn grammar_value(&self) -> Value {
        let mut v = self.format_value();
        strip_annotations(&mut v);
        v
    }

    /// A copy whose list budgets are reduced by `scale`: every array whose
    /// `maxItems` exceeds its `minItems` (fixed-length tuples such as boxes are
    /// left alone) gets `maxItems = max(minItems, 1, floor(maxItems * scale))`.
    /// Validation still runs against the Rust type as before.
    pub fn with_scaled_max_items(&self, scale: f64) -> Result<Self> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(VisionError::Config(format!(
                "list budget scale must be positive, got {scale}"
            )));
        }
        let mut schema = self.schema.clone();
        scale_max_items(&mut schema, scale);
        let validator = jsonschema::validator_for(&schema)
            .map_err(|e| VisionError::Config(format!("scaled schema is invalid: {e}")))?;
        Ok(Self {
            name: self.name.clone(),
            schema,
            validator: Arc::new(validator),
            typed_check: Arc::clone(&self.typed_check),
        })
    }

    /// Validate `value` against the schema, then against the Rust type.
    pub fn validate(&self, value: &Value) -> std::result::Result<(), Vec<FieldError>> {
        let errors: Vec<FieldError> = self
            .validator
            .iter_errors(value)
            .map(|e| FieldError {
                path: e.instance_path().to_string(),
                message: e.to_string(),
            })
            .collect();
        if !errors.is_empty() {
            return Err(errors);
        }
        (self.typed_check)(value)
    }
}

/// Annotation keywords that never constrain an instance.
const ANNOTATION_KEYWORDS: [&str; 5] = ["title", "description", "format", "examples", "default"];

/// Keywords whose value is one subschema.
const SUBSCHEMA_KEYWORDS: [&str; 5] = [
    "items",
    "additionalProperties",
    "not",
    "contains",
    "additionalItems",
];

/// Keywords whose value is a list of subschemas.
const SUBSCHEMA_LIST_KEYWORDS: [&str; 4] = ["anyOf", "oneOf", "allOf", "prefixItems"];

/// Keywords whose value maps names to subschemas.
const SUBSCHEMA_MAP_KEYWORDS: [&str; 4] =
    ["properties", "$defs", "definitions", "patternProperties"];

/// Visits `schema` and every nested subschema (never property names).
fn visit_schemas(schema: &mut Value, f: &mut dyn FnMut(&mut serde_json::Map<String, Value>)) {
    let Value::Object(map) = schema else { return };
    f(map);
    for k in SUBSCHEMA_KEYWORDS {
        if let Some(sub) = map.get_mut(k) {
            match sub {
                Value::Array(list) => list.iter_mut().for_each(|s| visit_schemas(s, f)),
                other => visit_schemas(other, f),
            }
        }
    }
    for k in SUBSCHEMA_LIST_KEYWORDS {
        if let Some(Value::Array(list)) = map.get_mut(k) {
            list.iter_mut().for_each(|s| visit_schemas(s, f));
        }
    }
    for k in SUBSCHEMA_MAP_KEYWORDS {
        if let Some(Value::Object(m)) = map.get_mut(k) {
            m.values_mut().for_each(|s| visit_schemas(s, f));
        }
    }
}

fn strip_annotations(schema: &mut Value) {
    visit_schemas(schema, &mut |m| {
        for k in ANNOTATION_KEYWORDS {
            m.remove(k);
        }
    });
}

fn scale_max_items(schema: &mut Value, scale: f64) {
    visit_schemas(schema, &mut |m| {
        let Some(max) = m.get("maxItems").and_then(Value::as_u64) else {
            return;
        };
        let min = m.get("minItems").and_then(Value::as_u64).unwrap_or(0);
        if max <= min {
            return;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let scaled = (max as f64 * scale).floor() as u64;
        m.insert("maxItems".into(), Value::from(scaled.max(min).max(1)));
    });
}

/// Append the schema instructions to a prompt.
pub fn prompt_with_schema(prompt: &str, schema: &OutputSchema) -> String {
    format!(
        "{}\n\nRespond with only a single JSON object that matches this JSON schema exactly \
         (no extra fields, no markdown):\n{}",
        prompt.trim_end(),
        schema.text()
    )
}

/// Parse model text as JSON, tolerating a surrounding markdown code fence.
pub fn parse_output_text(text: &str) -> std::result::Result<Value, FieldError> {
    let trimmed = strip_code_fence(text.trim());
    serde_json::from_str(trimmed).map_err(|e| FieldError {
        path: String::new(),
        message: format!("output is not valid JSON: {e}"),
    })
}

fn strip_code_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

fn path_to_pointer(path: &serde_path_to_error::Path) -> String {
    use serde_path_to_error::Segment;
    let mut out = String::new();
    for seg in path.iter() {
        out.push('/');
        match seg {
            Segment::Seq { index } => out.push_str(&index.to_string()),
            Segment::Map { key } => out.push_str(&key.replace('~', "~0").replace('/', "~1")),
            Segment::Enum { variant } => out.push_str(variant),
            Segment::Unknown => out.push('?'),
        }
    }
    out
}

/// Decode a JSON value into `T`, reporting the failing path.
pub fn decode_value<T: DeserializeOwned>(value: &Value) -> std::result::Result<T, Vec<FieldError>> {
    serde_path_to_error::deserialize::<_, T>(value).map_err(|e| {
        vec![FieldError {
            path: path_to_pointer(e.path()),
            message: e.inner().to_string(),
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::json;

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Inner {
        #[schemars(range(min = 0.0, max = 1.0))]
        score: f64,
    }

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Outer {
        kind: Kind,
        items: Vec<Inner>,
    }

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    #[serde(rename_all = "snake_case")]
    enum Kind {
        Alpha,
        Beta,
    }

    fn schema() -> OutputSchema {
        match OutputSchema::for_type::<Outer>() {
            Ok(s) => s,
            Err(e) => panic!("schema build failed: {e}"),
        }
    }

    #[test]
    fn schema_is_inlined_and_closed() {
        let s = schema();
        let text = s.text();
        assert!(!text.contains("$ref"), "{text}");
        assert!(!text.contains("$schema"));
        assert_eq!(s.json()["additionalProperties"], json!(false));
        assert_eq!(
            s.json()["properties"]["kind"]["enum"],
            json!(["alpha", "beta"])
        );
    }

    #[test]
    fn valid_document_passes() {
        let s = schema();
        let v = json!({"kind": "beta", "items": [{"score": 0.5}]});
        assert!(s.validate(&v).is_ok());
        let decoded: std::result::Result<Outer, _> = decode_value(&v);
        assert!(matches!(
            decoded,
            Ok(Outer {
                kind: Kind::Beta,
                ..
            })
        ));
    }

    #[test]
    fn schema_errors_carry_paths() {
        let s = schema();
        let v = json!({"kind": "gamma", "items": [{"score": 0.5}, {"score": 3.0}]});
        let errs = s.validate(&v).err().unwrap_or_default();
        let paths: Vec<&str> = errs.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"/kind"), "{paths:?}");
        assert!(paths.contains(&"/items/1/score"), "{paths:?}");
    }

    #[test]
    fn unknown_field_is_rejected_with_path() {
        let s = schema();
        let v = json!({"kind": "alpha", "items": [{"score": 0.1, "extra": 1}]});
        let errs = s.validate(&v).err().unwrap_or_default();
        assert!(errs.iter().any(|e| e.path == "/items/0"), "{errs:?}");
        // The typed decoder alone also names the path.
        let typed = decode_value::<Outer>(&v).err().unwrap_or_default();
        assert_eq!(typed[0].path, "/items/0/extra");
    }

    /// Documented output with a property literally named `format` and a box tuple.
    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Annotated {
        /// The output format name.
        format: String,
        /// A bounded list.
        #[schemars(length(max = 10))]
        rows: Vec<Row>,
    }

    /// One row.
    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Row {
        /// Box corners.
        bbox: [f64; 4],
        /// Score as a double.
        value: f64,
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn grammar_value_drops_annotations_but_keeps_properties() {
        let s = OutputSchema::for_type::<Annotated>().unwrap();
        assert!(
            s.text().contains("description"),
            "prompt schema is unchanged"
        );
        let g = s.grammar_value();
        let text = g.to_string();
        assert!(!text.contains("\"description\""), "{text}");
        assert!(!text.contains("\"title\""), "{text}");
        assert!(!text.contains("\"$schema\""), "{text}");
        // The property named `format` survives; only the keyword is removed.
        assert!(g["properties"]["format"].is_object(), "{text}");
        assert!(g["properties"]["rows"]["items"]["properties"]["value"]
            .get("format")
            .is_none());
        assert_eq!(g["properties"]["rows"]["maxItems"], json!(10));
        assert_eq!(g["required"], s.format_value()["required"]);
        assert_eq!(g["additionalProperties"], json!(false));
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn scaled_max_items_leaves_tuples_alone() {
        let s = OutputSchema::for_type::<Annotated>().unwrap();
        let half = s.with_scaled_max_items(0.5).unwrap();
        let rows = &half.json()["properties"]["rows"];
        assert_eq!(rows["maxItems"], json!(5));
        assert_eq!(rows["items"]["properties"]["bbox"]["maxItems"], json!(4));
        assert_eq!(rows["items"]["properties"]["bbox"]["minItems"], json!(4));
        let tiny = s.with_scaled_max_items(0.01).unwrap();
        assert_eq!(tiny.json()["properties"]["rows"]["maxItems"], json!(1));
        // The scaled validator enforces the smaller budget.
        let row = json!({"bbox": [0, 0, 1, 1], "value": 0.5});
        let six = json!({"format": "x", "rows": [row, row, row, row, row, row]});
        assert!(s.validate(&six).is_ok());
        assert!(half.validate(&six).is_err());
        assert!(s.with_scaled_max_items(0.0).is_err());
        assert!(s.with_scaled_max_items(f64::NAN).is_err());
    }

    #[test]
    fn code_fences_are_tolerated() {
        let v = parse_output_text("```json\n{\"a\": 1}\n```");
        assert_eq!(v.ok(), Some(json!({"a": 1})));
        assert!(parse_output_text("not json").is_err());
    }
}
