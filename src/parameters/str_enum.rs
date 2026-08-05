//! `str_enum!` — declare an enumerated parameter once and get its XGBoost
//! spelling, `Display`, `FromStr`, and serde impls for free.
//!
//! Every enumerated parameter in XGBoost is declared in C++ with a list of
//! `.add_enum("name", Value)` clauses. This macro is the Rust counterpart: the
//! variant list and the wire spellings live in exactly one place, so a value
//! can never serialise to a name that `FromStr` would not accept.

macro_rules! str_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident : $param:literal {
            $( $(#[$variant_meta:meta])* $variant:ident = $text:literal ),+ $(,)?
        }
        default = $default:ident;
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        $vis enum $name {
            $( $(#[$variant_meta])* $variant ),+
        }

        impl $name {
            /// Every accepted value, in declaration order.
            pub const ALL: &'static [Self] = &[ $( Self::$variant ),+ ];

            /// The XGBoost spelling of this value.
            pub const fn as_str(self) -> &'static str {
                match self { $( Self::$variant => $text ),+ }
            }

            /// The XGBoost parameter this enum belongs to.
            pub const fn parameter_name() -> &'static str {
                $param
            }
        }

        impl ::std::default::Default for $name {
            fn default() -> Self {
                Self::$default
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = $crate::error::Error;

            fn from_str(s: &str) -> $crate::error::Result<Self> {
                match s {
                    $( $text => Ok(Self::$variant), )+
                    other => Err($crate::error::Error::parse(
                        $param,
                        other,
                        format!("expected one of: {}", [$($text),+].join(", ")),
                    )),
                }
            }
        }

        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::std::result::Result<Self, D::Error> {
                let text = <::std::string::String as ::serde::Deserialize>::deserialize(deserializer)?;
                text.parse().map_err(::serde::de::Error::custom)
            }
        }
    };
}

pub(crate) use str_enum;
