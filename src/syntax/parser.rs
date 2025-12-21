//! Parser for Ein syntax.
//!
//! Converts tokens into AST with proper operator precedence.
//!
//! Precedence (lowest to highest):
//! 1. +, - (additive)
//! 2. *, / (multiplicative)
//! 3. space (tensor join/einsum)
//! 4. function application, literals, parentheses

use super::ast::{BinOp, Constraint, Expr, ParamInit, Statement, TensorRef, TypeSpec};
use super::token::Token;
use logos::Logos;
use std::iter::Peekable;

/// Parser state.
pub struct Parser<'a> {
    tokens: Peekable<Box<dyn Iterator<Item = Token> + 'a>>,
}

impl<'a> Parser<'a> {
    /// Create a new parser from input string.
    pub fn new(input: &'a str) -> Self {
        let lexer = Token::lexer(input);
        let iter: Box<dyn Iterator<Item = Token> + 'a> =
            Box::new(lexer.filter_map(|t| t.ok()));
        Self {
            tokens: iter.peekable(),
        }
    }

    /// Peek at the next token without consuming it.
    fn peek(&mut self) -> Option<&Token> {
        self.tokens.peek()
    }

    /// Consume and return the next token.
    fn next(&mut self) -> Option<Token> {
        self.tokens.next()
    }

    /// Expect a specific token, return error if not found.
    fn expect(&mut self, expected: Token) -> Result<(), String> {
        match self.next() {
            Some(t) if t == expected => Ok(()),
            Some(t) => Err(format!("Expected {}, got {}", expected, t)),
            None => Err(format!("Expected {}, got end of input", expected)),
        }
    }

    /// Check if the current token is a function keyword.
    fn is_function_token(token: &Token) -> bool {
        matches!(
            token,
            Token::Sigmoid
                | Token::Relu
                | Token::Step
                | Token::Softmax
                | Token::Gelu
                | Token::Tanh
                | Token::Sqrt
                | Token::Exp
                | Token::Log
                | Token::Sum
                | Token::Mean
                | Token::LayerNorm
                | Token::CrossEntropy
                | Token::Mse
                | Token::Embed
                | Token::CausalMask
                | Token::MaskFill
                | Token::NegInf
                | Token::Arange
                | Token::SinPos
                | Token::Reshape
                | Token::Transpose
                | Token::View
                | Token::Size
        )
    }

    /// Get function name from token.
    fn function_name(token: &Token) -> &'static str {
        match token {
            Token::Sigmoid => "sigmoid",
            Token::Relu => "relu",
            Token::Step => "step",
            Token::Softmax => "softmax",
            Token::Gelu => "gelu",
            Token::Tanh => "tanh",
            Token::Sqrt => "sqrt",
            Token::Exp => "exp",
            Token::Log => "log",
            Token::Sum => "sum",
            Token::Mean => "mean",
            Token::LayerNorm => "lnorm",
            Token::CrossEntropy => "cross_entropy",
            Token::Mse => "mse",
            Token::Embed => "embed",
            Token::CausalMask => "causal_mask",
            Token::MaskFill => "mask_fill",
            Token::NegInf => "neg_inf",
            Token::Arange => "arange",
            Token::SinPos => "sin_pos",
            Token::Reshape => "reshape",
            Token::Transpose => "transpose",
            Token::View => "view",
            Token::Size => "size",
            _ => panic!("Not a function token"),
        }
    }

    /// Parse a tensor reference: `Name[i,j]` or `Name(x,y)`
    fn parse_tensor_ref(&mut self) -> Result<TensorRef, String> {
        let name = match self.next() {
            Some(Token::Ident(s)) => s,
            Some(t) => return Err(format!("Expected identifier, got {}", t)),
            None => return Err("Expected identifier".into()),
        };

        // Check for bracket style [i,j] or paren style (x,y)
        let (_open, close) = match self.peek() {
            Some(Token::LBracket) => (Token::LBracket, Token::RBracket),
            Some(Token::LParen) => (Token::LParen, Token::RParen),
            _ => {
                // No indices - scalar or bare name
                return Ok(TensorRef {
                    name,
                    indices: vec![],
                });
            }
        };

        self.next(); // consume open bracket/paren

        let mut indices = Vec::new();
        loop {
            match self.peek() {
                Some(t) if *t == close => {
                    self.next();
                    break;
                }
                Some(Token::Comma) => {
                    self.next();
                }
                Some(Token::Ident(_)) => {
                    if let Some(Token::Ident(idx)) = self.next() {
                        indices.push(idx);
                    }
                }
                Some(t) => return Err(format!("Unexpected token in indices: {}", t)),
                None => return Err("Unexpected end of input in indices".into()),
            }
        }

        Ok(TensorRef { name, indices })
    }

    /// Parse an array literal: `[1.0, 2.0]` or `[[1.0, 2.0], [3.0, 4.0]]`
    fn parse_array(&mut self) -> Result<Expr, String> {
        self.expect(Token::LBracket)?;

        let mut elements = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBracket) => {
                    self.next();
                    break;
                }
                Some(Token::Comma) => {
                    self.next();
                }
                Some(Token::LBracket) => {
                    // Nested array
                    let nested = self.parse_array()?;
                    elements.push(nested);
                }
                Some(Token::Int(_) | Token::Float(_)) => {
                    let val = match self.next().unwrap() {
                        Token::Int(n) => n as f64,
                        Token::Float(f) => f,
                        _ => unreachable!(),
                    };
                    elements.push(Expr::Literal(val));
                }
                Some(Token::Minus) => {
                    // Handle negative numbers
                    self.next();
                    let val = match self.next() {
                        Some(Token::Int(n)) => -(n as f64),
                        Some(Token::Float(f)) => -f,
                        Some(t) => return Err(format!("Expected number after minus, got {}", t)),
                        None => return Err("Expected number after minus".into()),
                    };
                    elements.push(Expr::Literal(val));
                }
                Some(t) => return Err(format!("Unexpected token in array: {}", t)),
                None => return Err("Unexpected end of input in array".into()),
            }
        }

        Ok(Expr::Array(elements))
    }

    /// Parse a primary expression (highest precedence).
    /// Primary: literal, tensor ref, function call, parenthesized expr, array
    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            // Array literal: [1.0, 2.0] or [[1.0, 2.0], [3.0, 4.0]]
            Some(Token::LBracket) => self.parse_array(),

            // Parenthesized expression
            Some(Token::LParen) => {
                self.next();
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }

            // Function application: sigmoid(x) or cross_entropy(logits, targets)
            Some(t) if Self::is_function_token(t) => {
                let func = Self::function_name(&self.next().unwrap()).to_string();
                self.expect(Token::LParen)?;

                // Parse comma-separated arguments
                let mut args = Vec::new();
                loop {
                    match self.peek() {
                        Some(Token::RParen) => {
                            self.next();
                            break;
                        }
                        Some(Token::Comma) => {
                            self.next();
                        }
                        _ => {
                            let arg = self.parse_expr()?;
                            args.push(arg);
                        }
                    }
                }

                Ok(Expr::Apply { func, args })
            }

            // Number literal
            Some(Token::Int(_) | Token::Float(_)) => {
                let val = match self.next().unwrap() {
                    Token::Int(n) => n as f64,
                    Token::Float(f) => f,
                    _ => unreachable!(),
                };
                Ok(Expr::Literal(val))
            }

            // Tensor reference: Name or Name[i,j]
            Some(Token::Ident(_)) => {
                let tensor_ref = self.parse_tensor_ref()?;
                Ok(Expr::Ref(tensor_ref))
            }

            Some(t) => Err(format!("Unexpected token in expression: {}", t)),
            None => Err("Unexpected end of input in expression".into()),
        }
    }

    /// Parse a join expression (tensor product via space).
    /// Higher precedence than arithmetic.
    fn parse_join(&mut self) -> Result<Expr, String> {
        let first = self.parse_primary()?;

        // Check for more primaries (space-separated join)
        // Only join if next token is an identifier (tensor ref)
        let mut terms = vec![first];
        while let Some(Token::Ident(_)) = self.peek() {
            let next_expr = self.parse_primary()?;
            terms.push(next_expr);
        }

        if terms.len() == 1 {
            Ok(terms.pop().unwrap())
        } else {
            Ok(Expr::Join(terms))
        }
    }

    /// Parse multiplicative expression (* /).
    fn parse_multiplicative(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_join()?;

        while let Some(Token::Star | Token::Slash) = self.peek() {
            let op = match self.next().unwrap() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                _ => unreachable!(),
            };
            let right = self.parse_join()?;
            left = Expr::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse additive expression (+ -).
    fn parse_additive(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_multiplicative()?;

        while let Some(Token::Plus | Token::Minus) = self.peek() {
            let op = match self.next().unwrap() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => unreachable!(),
            };
            let right = self.parse_multiplicative()?;
            left = Expr::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse an expression (RHS of equation).
    pub fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_additive()
    }

    /// Parse a type specification: `Float[768, 768]`
    fn parse_type_spec(&mut self) -> Result<TypeSpec, String> {
        // Parse dtype (Float, Int, etc.)
        let dtype = match self.next() {
            Some(Token::Ident(s)) => s,
            Some(t) => return Err(format!("Expected type name, got {}", t)),
            None => return Err("Expected type name".into()),
        };

        // Parse shape: [dim1, dim2, ...]
        self.expect(Token::LBracket)?;

        let mut shape = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RBracket) => {
                    self.next();
                    break;
                }
                Some(Token::Comma) => {
                    self.next();
                }
                Some(Token::Int(_)) => {
                    if let Some(Token::Int(n)) = self.next() {
                        shape.push(n as usize);
                    }
                }
                Some(t) => return Err(format!("Unexpected token in shape: {}", t)),
                None => return Err("Unexpected end of input in shape".into()),
            }
        }

        Ok(TypeSpec { dtype, shape })
    }

    /// Parse a @param declaration: `@param W: Float[768, 768]`
    fn parse_param_decl(&mut self) -> Result<Statement, String> {
        self.expect(Token::Param)?;

        // Parse parameter name
        let name = match self.next() {
            Some(Token::Ident(s)) => s,
            Some(t) => return Err(format!("Expected parameter name, got {}", t)),
            None => return Err("Expected parameter name".into()),
        };

        // Expect colon
        self.expect(Token::Colon)?;

        // Parse type specification
        let type_spec = self.parse_type_spec()?;

        // Default initialization is Xavier
        let init = ParamInit::Xavier;

        Ok(Statement::ParamDecl {
            name,
            type_spec,
            init,
        })
    }

    /// Parse an @embedding declaration: `@embedding Name: vocab=65 dim=384`
    fn parse_embedding_decl(&mut self) -> Result<Statement, String> {
        self.expect(Token::Embedding)?;

        // Parse embedding name
        let name = match self.next() {
            Some(Token::Ident(s)) => s,
            Some(t) => return Err(format!("Expected embedding name, got {}", t)),
            None => return Err("Expected embedding name".into()),
        };

        // Expect colon
        self.expect(Token::Colon)?;

        // Parse kwargs: vocab=N dim=M
        let mut vocab_size: Option<usize> = None;
        let mut embed_dim: Option<usize> = None;

        loop {
            match self.peek() {
                Some(Token::Ident(key)) => {
                    let key = key.clone();
                    self.next();

                    self.expect(Token::Eq)?;

                    let value = match self.next() {
                        Some(Token::Int(n)) => n as usize,
                        Some(t) => return Err(format!("Expected integer value, got {}", t)),
                        None => return Err("Expected integer value".into()),
                    };

                    match key.as_str() {
                        "vocab" => vocab_size = Some(value),
                        "dim" => embed_dim = Some(value),
                        _ => return Err(format!("Unknown embedding parameter: {}", key)),
                    }
                }
                _ => break,
            }
        }

        let vocab_size = vocab_size.ok_or("Missing 'vocab' parameter for @embedding")?;
        let embed_dim = embed_dim.ok_or("Missing 'dim' parameter for @embedding")?;

        Ok(Statement::EmbeddingDecl {
            name,
            vocab_size,
            embed_dim,
        })
    }

    /// Parse a @forward declaration: `@forward X = embed(E, Inputs)`
    /// This is a deferred equation that gets stored but not evaluated until :forward
    fn parse_forward_decl(&mut self) -> Result<Statement, String> {
        self.expect(Token::Forward)?;

        // Parse the equation: LHS = RHS
        let lhs = self.parse_tensor_ref()?;
        self.expect(Token::Eq)?;
        let rhs = self.parse_expr()?;

        Ok(Statement::ForwardDecl { lhs, rhs })
    }

    /// Parse a statement.
    pub fn parse_statement(&mut self) -> Result<Statement, String> {
        // Check for @param declaration
        if let Some(Token::Param) = self.peek() {
            return self.parse_param_decl();
        }

        // Check for @embedding declaration
        if let Some(Token::Embedding) = self.peek() {
            return self.parse_embedding_decl();
        }

        // Check for @forward declaration (deferred equation)
        if let Some(Token::Forward) = self.peek() {
            return self.parse_forward_decl();
        }

        // Check for array literal assignment: X = [1, 2, 3]
        if let Some(Token::Ident(_)) = self.peek() {
            let first = self.parse_tensor_ref()?;

            match self.peek() {
                // Equation: `Y[i] = expr` or `X = [1, 2, 3]`
                Some(Token::Eq) => {
                    self.next();
                    let rhs = self.parse_expr()?;
                    Ok(Statement::Equation { lhs: first, rhs })
                }

                // Rule: `Head(x,y) <- Body1(x,z) Body2(z,y), x != y`
                Some(Token::Arrow) => {
                    self.next();
                    let mut body = Vec::new();
                    let mut constraints = Vec::new();

                    // Parse body atoms and constraints
                    while let Some(Token::Ident(_)) = self.peek() {
                        let tensor_ref = self.parse_tensor_ref()?;

                        // Check if this is a constraint: `x != y`
                        if tensor_ref.indices.is_empty() {
                            if let Some(Token::NotEq) = self.peek() {
                                self.next(); // consume !=
                                let other = match self.next() {
                                    Some(Token::Ident(s)) => s,
                                    Some(t) => {
                                        return Err(format!(
                                            "Expected variable in constraint, got {}",
                                            t
                                        ))
                                    }
                                    None => return Err("Expected variable in constraint".into()),
                                };
                                constraints.push(Constraint::NotEqual(tensor_ref.name, other));
                            } else {
                                body.push(tensor_ref);
                            }
                        } else {
                            body.push(tensor_ref);
                        }

                        // Skip optional comma
                        if let Some(Token::Comma) = self.peek() {
                            self.next();
                        }
                    }
                    Ok(Statement::Rule {
                        head: first,
                        body,
                        constraints,
                    })
                }

                // Query: `Name(x,y)?`
                Some(Token::Question) => {
                    self.next();
                    Ok(Statement::Query(first))
                }

                // Fact: `Name(x,y).`
                Some(Token::Dot) => {
                    self.next();
                    Ok(Statement::Fact(first))
                }

                // Just a tensor ref on its own (treat as query)
                None => Ok(Statement::Query(first)),

                Some(t) => Err(format!("Unexpected token after tensor ref: {}", t)),
            }
        } else {
            Err("Expected identifier at start of statement".into())
        }
    }
}

/// Convenience function to parse a single statement.
pub fn parse(input: &str) -> Result<Statement, String> {
    let mut parser = Parser::new(input);
    parser.parse_statement()
}

/// Convenience function to parse an equation specifically.
pub fn parse_equation(input: &str) -> Result<(TensorRef, Expr), String> {
    match parse(input)? {
        Statement::Equation { lhs, rhs } => Ok((lhs, rhs)),
        other => Err(format!("Expected equation, got {:?}", other)),
    }
}