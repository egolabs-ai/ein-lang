//! Knowledge base for forward chaining.
//!
//! Stores relations (as sparse boolean tensors) and rules.
//! Implements semi-naive fixpoint evaluation.

use crate::syntax::ast::{Constraint, Statement};
use crate::tensor::SparseBool;
use indexmap::IndexMap;
use rustc_hash::FxHashMap;

/// Knowledge base storing relations and rules.
pub struct KnowledgeBase {
    /// Relations by name -> SparseBool
    relations: IndexMap<String, SparseBool>,
    /// Rules: head predicate derived from body predicates
    rules: Vec<Rule>,
    /// Symbol table: string -> integer ID
    symbols: FxHashMap<String, usize>,
    /// Reverse symbol table: ID -> string
    symbols_rev: Vec<String>,
}

/// A parsed rule ready for forward chaining.
#[derive(Debug, Clone)]
pub struct Rule {
    pub head_name: String,
    pub head_vars: Vec<String>,
    pub body: Vec<BodyAtom>,
    pub constraints: Vec<RuleConstraint>,
}

/// A constraint on variables in a rule.
#[derive(Debug, Clone)]
pub enum RuleConstraint {
    /// Variables must not be equal: `x != y`
    NotEqual(String, String),
}

/// An atom in a rule body.
#[derive(Debug, Clone)]
pub struct BodyAtom {
    pub name: String,
    pub vars: Vec<String>,
}

impl KnowledgeBase {
    /// Create a new empty knowledge base.
    pub fn new() -> Self {
        Self {
            relations: IndexMap::new(),
            rules: Vec::new(),
            symbols: FxHashMap::default(),
            symbols_rev: Vec::new(),
        }
    }

    /// Intern a symbol, returning its ID.
    pub fn intern(&mut self, s: &str) -> usize {
        if let Some(&id) = self.symbols.get(s) {
            id
        } else {
            let id = self.symbols_rev.len();
            self.symbols.insert(s.to_string(), id);
            self.symbols_rev.push(s.to_string());
            id
        }
    }

    /// Get the symbol for an ID.
    pub fn symbol(&self, id: usize) -> Option<&str> {
        self.symbols_rev.get(id).map(|s| s.as_str())
    }

    /// Get or create a relation with the given arity.
    pub fn get_or_create_relation(&mut self, name: &str, arity: usize) -> &mut SparseBool {
        if !self.relations.contains_key(name) {
            self.relations.insert(name.to_string(), SparseBool::new(arity));
        }
        self.relations.get_mut(name).unwrap()
    }

    /// Get a relation by name.
    pub fn get_relation(&self, name: &str) -> Option<&SparseBool> {
        self.relations.get(name)
    }

    /// Add a fact like Parent(Alice, Bob).
    pub fn add_fact(&mut self, name: &str, args: &[&str]) {
        let arity = args.len();
        let tuple: Vec<usize> = args.iter().map(|s| self.intern(s)).collect();
        self.get_or_create_relation(name, arity).insert(tuple);
    }

    /// Add a rule from a parsed statement.
    pub fn add_rule(&mut self, stmt: &Statement) {
        if let Statement::Rule {
            head,
            body,
            constraints,
        } = stmt
        {
            let rule = Rule {
                head_name: head.name.clone(),
                head_vars: head.indices.clone(),
                body: body
                    .iter()
                    .map(|b| BodyAtom {
                        name: b.name.clone(),
                        vars: b.indices.clone(),
                    })
                    .collect(),
                constraints: constraints
                    .iter()
                    .map(|c| match c {
                        Constraint::NotEqual(a, b) => RuleConstraint::NotEqual(a.clone(), b.clone()),
                    })
                    .collect(),
            };
            self.rules.push(rule);
        }
    }

    /// Add a fact from a parsed statement.
    pub fn add_fact_stmt(&mut self, stmt: &Statement) {
        if let Statement::Fact(f) = stmt {
            let args: Vec<&str> = f.indices.iter().map(|s| s.as_str()).collect();
            self.add_fact(&f.name, &args);
        }
    }

    /// Run forward chaining until fixpoint.
    /// Returns the number of iterations performed.
    pub fn forward_chain(&mut self) -> usize {
        let mut iterations = 0;
        let max_iterations = 1000; // Safety limit

        loop {
            if iterations >= max_iterations {
                break;
            }

            let old_total: usize = self.relations.values().map(|r| r.len()).sum();

            // Apply each rule once
            for rule in &self.rules.clone() {
                self.apply_rule(rule);
            }

            let new_total: usize = self.relations.values().map(|r| r.len()).sum();
            iterations += 1;

            // Fixpoint reached when no new facts
            if new_total == old_total {
                break;
            }
        }

        iterations
    }

    /// Apply a single rule, deriving new facts.
    fn apply_rule(&mut self, rule: &Rule) {
        if rule.body.is_empty() {
            return;
        }

        // Handle single-body rules: Head(x,y) <- Body(x,y)
        if rule.body.len() == 1 {
            let body_atom = &rule.body[0];
            if let Some(body_rel) = self.relations.get(&body_atom.name) {
                // Build mapping from body vars to head vars
                let derived = self.project_and_rename(body_rel, &body_atom.vars, &rule.head_vars);

                // Union into head relation (with constraint filtering)
                let arity = rule.head_vars.len();
                let head_rel = self.get_or_create_relation(&rule.head_name, arity);
                for tuple in derived.tuples {
                    if satisfies_constraints(&tuple, &rule.head_vars, &rule.constraints) {
                        head_rel.insert(tuple);
                    }
                }
            }
            return;
        }

        // Handle two-body rules: Head(x,z) <- Body1(x,y), Body2(y,z)
        if rule.body.len() == 2 {
            let body1 = &rule.body[0];
            let body2 = &rule.body[1];

            let rel1 = match self.relations.get(&body1.name) {
                Some(r) => r.clone(),
                None => return,
            };
            let rel2 = match self.relations.get(&body2.name) {
                Some(r) => r.clone(),
                None => return,
            };

            // Find shared variables and their positions
            if let Some((idx1, idx2)) = find_join_indices(&body1.vars, &body2.vars) {
                let joined = rel1.join(&rel2, idx1, idx2);

                // Build combined var list after join
                let mut combined_vars = body1.vars.clone();
                for (i, var) in body2.vars.iter().enumerate() {
                    if i != idx2 {
                        combined_vars.push(var.clone());
                    }
                }

                // Filter by constraints before projection
                let filtered = filter_by_constraints(&joined, &combined_vars, &rule.constraints);

                // Project to head variables
                let derived = self.project_and_rename(&filtered, &combined_vars, &rule.head_vars);

                // Union into head
                let arity = rule.head_vars.len();
                let head_rel = self.get_or_create_relation(&rule.head_name, arity);
                for tuple in derived.tuples {
                    head_rel.insert(tuple);
                }
            }
        }
    }

    /// Project relation to keep only variables in target, reordering as needed.
    fn project_and_rename(
        &self,
        rel: &SparseBool,
        source_vars: &[String],
        target_vars: &[String],
    ) -> SparseBool {
        // Build index mapping: for each target var, which source position?
        let indices: Vec<usize> = target_vars
            .iter()
            .filter_map(|tv| source_vars.iter().position(|sv| sv == tv))
            .collect();

        if indices.len() != target_vars.len() {
            // Some target vars not in source - return empty
            return SparseBool::new(target_vars.len());
        }

        rel.project(&indices)
    }

    /// Query a relation, returning all tuples as string vectors.
    pub fn query(&self, name: &str) -> Vec<Vec<String>> {
        match self.relations.get(name) {
            Some(rel) => rel
                .tuples
                .iter()
                .map(|tuple| {
                    tuple
                        .iter()
                        .map(|&id| self.symbols_rev[id].clone())
                        .collect()
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Query a relation with filters.
    /// Constants (uppercase first char) filter results; variables (lowercase) match anything.
    /// Example: `Ancestor(Alice, x)` filters for Alice in first position.
    pub fn query_filtered(&self, name: &str, args: &[String]) -> Vec<Vec<String>> {
        let all_results = self.query(name);
        if args.is_empty() {
            return all_results;
        }

        // Build filters: (position, value) for constant arguments
        let filters: Vec<(usize, &str)> = args
            .iter()
            .enumerate()
            .filter_map(|(i, arg)| {
                // Treat as constant if starts with uppercase letter
                if arg.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    Some((i, arg.as_str()))
                } else {
                    None
                }
            })
            .collect();

        if filters.is_empty() {
            return all_results;
        }

        // Filter results
        all_results
            .into_iter()
            .filter(|tuple| {
                filters
                    .iter()
                    .all(|(pos, val)| tuple.get(*pos).is_some_and(|t| t == *val))
            })
            .collect()
    }

    /// Get all relation names.
    pub fn relation_names(&self) -> Vec<&String> {
        self.relations.keys().collect()
    }

    /// Save all facts to a file.
    /// Format: one fact per line as `RelationName(arg1, arg2, ...)`
    pub fn save_facts(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;

        for (name, rel) in &self.relations {
            for tuple in &rel.tuples {
                let args: Vec<&str> = tuple
                    .iter()
                    .map(|&id| self.symbols_rev[id].as_str())
                    .collect();
                writeln!(file, "{}({}).", name, args.join(", "))?;
            }
        }

        Ok(())
    }

    /// Load facts from a file.
    /// Returns the number of facts loaded.
    pub fn load_facts(&mut self, path: &str) -> std::io::Result<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with("//") {
                continue;
            }

            // Parse fact: Name(arg1, arg2, ...).
            if let Some(fact_str) = line.strip_suffix('.') {
                if let Some(lparen) = fact_str.find('(') {
                    if let Some(rparen) = fact_str.rfind(')') {
                        let name = fact_str[..lparen].trim();
                        let args_str = &fact_str[lparen + 1..rparen];
                        let args: Vec<&str> = args_str
                            .split(',')
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty())
                            .collect();

                        self.add_fact(name, &args);
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }

    /// Get the total number of facts across all relations.
    pub fn total_facts(&self) -> usize {
        self.relations.values().map(|r| r.len()).sum()
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

/// Find indices for joining two body atoms on a shared variable.
fn find_join_indices(vars1: &[String], vars2: &[String]) -> Option<(usize, usize)> {
    for (i, v1) in vars1.iter().enumerate() {
        for (j, v2) in vars2.iter().enumerate() {
            if v1 == v2 {
                return Some((i, j));
            }
        }
    }
    None
}

/// Check if a tuple satisfies all constraints.
fn satisfies_constraints(tuple: &[usize], vars: &[String], constraints: &[RuleConstraint]) -> bool {
    for constraint in constraints {
        match constraint {
            RuleConstraint::NotEqual(var_a, var_b) => {
                // Find positions of both variables
                let pos_a = vars.iter().position(|v| v == var_a);
                let pos_b = vars.iter().position(|v| v == var_b);

                if let (Some(a), Some(b)) = (pos_a, pos_b) {
                    // If positions found and values are equal, constraint fails
                    if tuple[a] == tuple[b] {
                        return false;
                    }
                }
                // If variables not found, constraint is vacuously satisfied
            }
        }
    }
    true
}

/// Filter a relation by constraints, returning only tuples that satisfy all constraints.
fn filter_by_constraints(
    rel: &SparseBool,
    vars: &[String],
    constraints: &[RuleConstraint],
) -> SparseBool {
    if constraints.is_empty() {
        return rel.clone();
    }

    let mut result = SparseBool::new(rel.arity);
    for tuple in &rel.tuples {
        if satisfies_constraints(tuple, vars, constraints) {
            result.insert(tuple.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_facts() {
        let mut kb = KnowledgeBase::new();
        kb.add_fact("Parent", &["Alice", "Bob"]);
        kb.add_fact("Parent", &["Bob", "Carol"]);

        let parent = kb.get_relation("Parent").unwrap();
        assert_eq!(parent.len(), 2);
    }

    #[test]
    fn test_symbol_table() {
        let mut kb = KnowledgeBase::new();
        let id1 = kb.intern("Alice");
        let id2 = kb.intern("Bob");
        let id3 = kb.intern("Alice"); // Same as id1

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(kb.symbol(id1), Some("Alice"));
        assert_eq!(kb.symbol(id2), Some("Bob"));
    }
}