//! A serializer that produces nothing and fails on non-finite floats.

use std::fmt::Display;

use serde::Serialize;
use serde::ser;

use super::CanonicalJsonError;

#[derive(Debug)]
enum CheckError {
    NonFinite(String),
    Custom(String),
}

impl Display for CheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite(p) => write!(f, "non-finite float at {p}"),
            Self::Custom(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for CheckError {}

impl ser::Error for CheckError {
    fn custom<T: Display>(msg: T) -> Self {
        Self::Custom(msg.to_string())
    }
}

/// Walks `value`, failing on the first NaN or infinite float.
pub(super) fn check<T: Serialize + ?Sized>(value: &T) -> Result<(), CanonicalJsonError> {
    let mut checker = Checker {
        path: vec!["$".to_string()],
    };
    match value.serialize(&mut checker) {
        Ok(()) => Ok(()),
        Err(CheckError::NonFinite(path)) => Err(CanonicalJsonError::NonFinite { path }),
        // Other failures (for example a map key that is not a string) are reported
        // by serde_json during the real conversion.
        Err(CheckError::Custom(_)) => Ok(()),
    }
}

struct Checker {
    path: Vec<String>,
}

impl Checker {
    fn float(&self, v: f64) -> Result<(), CheckError> {
        if v.is_finite() {
            Ok(())
        } else {
            Err(CheckError::NonFinite(self.path.concat()))
        }
    }

    fn nested<T: Serialize + ?Sized>(
        &mut self,
        segment: String,
        value: &T,
    ) -> Result<(), CheckError> {
        self.path.push(segment);
        let result = value.serialize(&mut *self);
        self.path.pop();
        result
    }
}

/// Compound state: the checker plus the next element index.
struct Compound<'a> {
    checker: &'a mut Checker,
    index: usize,
}

impl Compound<'_> {
    fn element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CheckError> {
        let segment = format!("[{}]", self.index);
        self.index += 1;
        self.checker.nested(segment, value)
    }
}

macro_rules! ok_scalars {
    ($($method:ident($ty:ty)),* $(,)?) => {
        $(fn $method(self, _v: $ty) -> Result<(), CheckError> { Ok(()) })*
    };
}

impl<'a> ser::Serializer for &'a mut Checker {
    type Ok = ();
    type Error = CheckError;
    type SerializeSeq = Compound<'a>;
    type SerializeTuple = Compound<'a>;
    type SerializeTupleStruct = Compound<'a>;
    type SerializeTupleVariant = Compound<'a>;
    type SerializeMap = Compound<'a>;
    type SerializeStruct = Compound<'a>;
    type SerializeStructVariant = Compound<'a>;

    ok_scalars!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_i128(i128),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_u128(u128),
        serialize_char(char),
        serialize_str(&str),
        serialize_bytes(&[u8]),
    );

    fn serialize_f32(self, v: f32) -> Result<(), CheckError> {
        self.float(f64::from(v))
    }

    fn serialize_f64(self, v: f64) -> Result<(), CheckError> {
        self.float(v)
    }

    fn serialize_none(self) -> Result<(), CheckError> {
        Ok(())
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), CheckError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<(), CheckError> {
        Ok(())
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<(), CheckError> {
        Ok(())
    }

    fn serialize_unit_variant(
        self,
        _n: &'static str,
        _i: u32,
        _v: &'static str,
    ) -> Result<(), CheckError> {
        Ok(())
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _n: &'static str,
        value: &T,
    ) -> Result<(), CheckError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _n: &'static str,
        _i: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), CheckError> {
        self.nested(format!(".{variant}"), value)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Compound<'a>, CheckError> {
        Ok(Compound {
            checker: self,
            index: 0,
        })
    }

    fn serialize_tuple(self, _len: usize) -> Result<Compound<'a>, CheckError> {
        Ok(Compound {
            checker: self,
            index: 0,
        })
    }

    fn serialize_tuple_struct(
        self,
        _n: &'static str,
        _len: usize,
    ) -> Result<Compound<'a>, CheckError> {
        Ok(Compound {
            checker: self,
            index: 0,
        })
    }

    fn serialize_tuple_variant(
        self,
        _n: &'static str,
        _i: u32,
        _v: &'static str,
        _len: usize,
    ) -> Result<Compound<'a>, CheckError> {
        Ok(Compound {
            checker: self,
            index: 0,
        })
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Compound<'a>, CheckError> {
        Ok(Compound {
            checker: self,
            index: 0,
        })
    }

    fn serialize_struct(self, _n: &'static str, _len: usize) -> Result<Compound<'a>, CheckError> {
        Ok(Compound {
            checker: self,
            index: 0,
        })
    }

    fn serialize_struct_variant(
        self,
        _n: &'static str,
        _i: u32,
        _v: &'static str,
        _len: usize,
    ) -> Result<Compound<'a>, CheckError> {
        Ok(Compound {
            checker: self,
            index: 0,
        })
    }
}

impl ser::SerializeSeq for Compound<'_> {
    type Ok = ();
    type Error = CheckError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CheckError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CheckError> {
        Ok(())
    }
}

impl ser::SerializeTuple for Compound<'_> {
    type Ok = ();
    type Error = CheckError;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CheckError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CheckError> {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for Compound<'_> {
    type Ok = ();
    type Error = CheckError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CheckError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CheckError> {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for Compound<'_> {
    type Ok = ();
    type Error = CheckError;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CheckError> {
        self.element(value)
    }
    fn end(self) -> Result<(), CheckError> {
        Ok(())
    }
}

impl ser::SerializeMap for Compound<'_> {
    type Ok = ();
    type Error = CheckError;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), CheckError> {
        // Keys are recorded for the path when they are strings.
        let label = match serde_json::to_value(key) {
            Ok(serde_json::Value::String(s)) => format!(".{s}"),
            _ => "{key}".to_string(),
        };
        self.checker.nested("{key}".to_string(), key)?;
        self.checker.path.push(label);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), CheckError> {
        let result = value.serialize(&mut *self.checker);
        self.checker.path.pop();
        result
    }
    fn end(self) -> Result<(), CheckError> {
        Ok(())
    }
}

impl ser::SerializeStruct for Compound<'_> {
    type Ok = ();
    type Error = CheckError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), CheckError> {
        self.checker.nested(format!(".{key}"), value)
    }
    fn end(self) -> Result<(), CheckError> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for Compound<'_> {
    type Ok = ();
    type Error = CheckError;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), CheckError> {
        self.checker.nested(format!(".{key}"), value)
    }
    fn end(self) -> Result<(), CheckError> {
        Ok(())
    }
}
