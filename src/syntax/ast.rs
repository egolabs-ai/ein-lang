//! Abstract Syntax Tree for Ein.

/// A reference to a tensor with indices.
/// Example: `W[i,j]` or `Parent(x,y)`
#[derive(Debug, Clone, PartialEq)]
pub struct TensorRef {
    pub name: String,
    pub indices: Vec<String>,
}

/// Type specification for parameters.
/// Example: `Float[768, 768]`
#[derive(Debug, Clone, PartialEq)]
pub struct TypeSpec {
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Initialization strategy for parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamInit {
    /// Xavier/Glorot initialization (default)
    Xavier,
    /// Zeros initialization
    Zeros,
    /// Normal distribution with stddev
    Normal(f64),
    /// Uniform distribution in range
    Uniform(f64, f64),
}

/// A constraint on variables in a rule.
/// Example: `x != y`
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// Variables must not be equal: `x != y`
    NotEqual(String, String),
}

/// Binary operators for tensor expressions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
        }
    }
}

/// An expression on the right-hand side of an equation.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A tensor reference: `W[i,j]`
    Ref(TensorRef),

    /// A join of multiple tensors: `W[i,j] X[j]` (space-separated)
    Join(Vec<Expr>),

    /// Function application: `sigmoid(expr)` or `cross_entropy(logits, targets)`
    Apply { func: String, args: Vec<Expr> },

    /// A literal number
    Literal(f64),

    /// Binary operation: `X + Y`, `X * Y`, etc.
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    /// Dense tensor literal: `[[1.0, 2.0], [3.0, 4.0]]`
    Array(Vec<Expr>),
}

/// A statement in Ein.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Equation: `Y[i] = W[i,j] X[j]`
    Equation { lhs: TensorRef, rhs: Expr },

    /// Rule (Datalog style): `Ancestor(x,y) <- Parent(x,y)` or with constraints
    /// `Sibling(x,y) <- Parent(p,x) Parent(p,y), x != y`
    Rule {
        head: TensorRef,
        body: Vec<TensorRef>,
        constraints: Vec<Constraint>,
    },

    /// Fact: `Parent(Alice, Bob).`
    Fact(TensorRef),

    /// Query: `Ancestor(x, y)?`
    Query(TensorRef),

    /// Parameter declaration: `@param W: Float[768, 768]`
    /// Creates a learnable tensor with gradient tracking
    ParamDecl {
        name: String,
        type_spec: TypeSpec,
        init: ParamInit,
    },

    /// Embedding declaration: `@embedding TokenEmbed: vocab=65 dim=384`
    /// Creates a learnable embedding lookup table
    EmbeddingDecl {
        name: String,
        vocab_size: usize,
        embed_dim: usize,
    },

    /// Forward declaration: `@forward X = embed(E, Inputs)`
    /// Deferred equation - stored but not evaluated until :forward command
    ForwardDecl {
        lhs: TensorRef,
        rhs: Expr,
    },
}