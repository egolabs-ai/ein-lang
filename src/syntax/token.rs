//! Token definitions for Ein lexer.
//!
//! Uses the `logos` crate for fast lexing.

use logos::Logos;

/// Tokens for the Ein language.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]  // Skip whitespace
#[logos(skip r"//[^\n]*")]     // Skip line comments
pub enum Token {
    // Brackets
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,

    // Punctuation
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("?")]
    Question,
    #[token(":")]
    Colon,

    // Directives
    #[token("@param")]
    Param,
    #[token("@embedding")]
    Embedding,
    #[token("@forward")]
    Forward,

    // Operators
    #[token("=")]
    Eq,
    #[token("!=")]
    NotEq,
    #[token("<-")]
    Arrow,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,

    // Built-in functions - activations
    #[token("sigmoid")]
    Sigmoid,
    #[token("relu")]
    Relu,
    #[token("step")]
    Step,
    #[token("softmax")]
    Softmax,
    #[token("gelu")]
    Gelu,
    #[token("tanh")]
    Tanh,

    // Built-in functions - math
    #[token("sqrt")]
    Sqrt,
    #[token("exp")]
    Exp,
    #[token("log")]
    Log,

    // Built-in functions - reductions
    #[token("sum")]
    Sum,
    #[token("mean")]
    Mean,
    #[token("lnorm")]
    LayerNorm,

    // Built-in functions - loss
    #[token("cross_entropy")]
    CrossEntropy,
    #[token("mse")]
    Mse,

    // Built-in functions - embedding
    #[token("embed")]
    Embed,

    // Built-in functions - masking (for causal attention)
    #[token("causal_mask")]
    CausalMask,
    #[token("mask_fill")]
    MaskFill,
    #[token("neg_inf")]
    NegInf,

    // Built-in functions - position embeddings
    #[token("arange")]
    Arange,
    #[token("sin_pos")]
    SinPos,

    // Built-in functions - tensor manipulation
    #[token("reshape")]
    Reshape,
    #[token("transpose")]
    Transpose,
    #[token("view")]
    View,
    #[token("size")]
    Size,

    // Identifiers (variable names, tensor names)
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Literals
    #[regex(r"-?[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),

    #[regex(r"-?[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Int(i64),
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::Dot => write!(f, "."),
            Token::Question => write!(f, "?"),
            Token::Colon => write!(f, ":"),
            Token::Param => write!(f, "@param"),
            Token::Embedding => write!(f, "@embedding"),
            Token::Forward => write!(f, "@forward"),
            Token::Eq => write!(f, "="),
            Token::NotEq => write!(f, "!="),
            Token::Arrow => write!(f, "<-"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Sigmoid => write!(f, "sigmoid"),
            Token::Relu => write!(f, "relu"),
            Token::Step => write!(f, "step"),
            Token::Softmax => write!(f, "softmax"),
            Token::Gelu => write!(f, "gelu"),
            Token::Tanh => write!(f, "tanh"),
            Token::Sqrt => write!(f, "sqrt"),
            Token::Exp => write!(f, "exp"),
            Token::Log => write!(f, "log"),
            Token::Sum => write!(f, "sum"),
            Token::Mean => write!(f, "mean"),
            Token::LayerNorm => write!(f, "lnorm"),
            Token::CrossEntropy => write!(f, "cross_entropy"),
            Token::Mse => write!(f, "mse"),
            Token::Embed => write!(f, "embed"),
            Token::CausalMask => write!(f, "causal_mask"),
            Token::MaskFill => write!(f, "mask_fill"),
            Token::NegInf => write!(f, "neg_inf"),
            Token::Arange => write!(f, "arange"),
            Token::SinPos => write!(f, "sin_pos"),
            Token::Reshape => write!(f, "reshape"),
            Token::Transpose => write!(f, "transpose"),
            Token::View => write!(f, "view"),
            Token::Size => write!(f, "size"),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Float(n) => write!(f, "{}", n),
            Token::Int(n) => write!(f, "{}", n),
        }
    }
}