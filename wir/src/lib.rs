//! WIR v0: Weight-Space Intermediate Representation.
//!
//! A reference compiler (WIR source -> one-hidden-layer ReLU weights) and
//! decompiler (weights -> canonical WIR). See `docs/WIR.md` in yumon-pet for
//! the language and `wir/README.md` for usage.
//!
//! ```
//! let src = "wir 0.1\ninput x: R1\noutput y: R1\ny = hinge at 0 turn 1 + hinge at 1 turn -1\n";
//! let compiled = wir::compile(src, Default::default()).unwrap();
//! let back = wir::decompile(&compiled.weights, &Default::default()).unwrap();
//! assert_eq!(wir::print(&back, wir::NumFormat::CANONICAL), wir::canonical_text(src).unwrap());
//! ```

// `!(x <= tol)` is deliberate throughout: a NaN must fail the check.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity)]

pub mod ant;
pub mod park;
pub mod ast;
pub mod canon;
pub mod codebook;
pub mod compiler;
pub mod deep;
pub mod games;
pub mod error;
pub mod fluid;
pub mod swarm;
pub mod tactics;
pub mod io;
pub mod lexer;
pub mod linalg;
pub mod model;
pub mod num;
pub mod parser;
pub mod printer;
pub mod superpose;

pub use canon::{canonicalize, decompile, DecompileOptions, Tolerances};
pub use codebook::{codebook, encode};
pub use compiler::{compile, lower_str, CompileOptions, Compiled, CostLine};
pub use error::{Result, Warning, WirError};
pub use model::{Mat, Program, Unit, UnitKind, Weights};
pub use num::{NumFormat, Precision};
pub use parser::parse;
pub use printer::print;

/// `print(canonicalize(lower(src)))` at canonical precision.
pub fn canonical_text(src: &str) -> Result<String> {
    let p = lower_str(src)?;
    Ok(print(&canonicalize(&p, Tolerances::default()), NumFormat::CANONICAL))
}
