//! Syntax module: lexer, parser, and AST.

pub mod ast;
mod parser;
mod token;

pub use ast::{Constraint, Expr, Statement, TensorRef};
pub use parser::{parse, parse_equation, Parser};
pub use token::Token;
