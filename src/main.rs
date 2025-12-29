//! Ein REPL - Interactive tensor logic interpreter.

use candle_core::Tensor;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use ein::syntax::ast::{Expr, TensorRef};
use ein::{parse, KBData, KnowledgeBase, Result, Runtime, Statement, TextData};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::env;
use std::fs;
use std::path::Path;

/// Deferred statements from @forward declarations
/// Each tuple contains: (TensorRef, Expr, original source string without @forward)
type DeferredStmts = Vec<(TensorRef, Expr, String)>;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let mut rt = Runtime::new();
    let mut kb = KnowledgeBase::new();

    // If a file argument is provided, execute it
    if args.len() > 1 {
        let file_path = &args[1];
        match load_file(file_path, &mut rt, &mut kb) {
            Ok(file_deferred) => {
                println!("Executed: {}", file_path);
                if !file_deferred.is_empty() {
                    println!("  {} deferred @forward statement(s) loaded", file_deferred.len());
                }
                // Run forward chaining automatically after loading
                let iterations = kb.forward_chain();
                if iterations > 0 {
                    println!("Forward chaining completed in {} iterations", iterations);
                }
            }
            Err(e) => {
                eprintln!("Error loading {}: {}", file_path, e);
                std::process::exit(1);
            }
        }

        // If --repl flag is passed, continue to REPL after executing file
        if args.len() > 2 && args[2] == "--repl" {
            return run_repl(rt, kb);
        }

        // Otherwise, show results and exit
        print_state(&rt, &kb);
        return Ok(());
    }

    // No file argument - run interactive REPL
    println!("Ein v0.1.0 - A tensor logic language");
    println!("Type :help for commands, :quit to exit\n");

    run_repl(rt, kb)
}

fn run_repl(mut rt: Runtime, mut kb: KnowledgeBase) -> Result<()> {
    let mut rl = DefaultEditor::new().expect("Failed to create editor");
    let mut text_data: Option<TextData> = None;
    let mut kb_data: Option<KBData> = None;
    let mut deferred: DeferredStmts = Vec::new();

    loop {
        let readline = rl.readline("ein> ");

        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(trimmed);

                if trimmed.starts_with(':') {
                    if !handle_command(trimmed, &mut rt, &mut kb, &mut text_data, &mut kb_data, &mut deferred) {
                        break;
                    }
                } else {
                    handle_statement(trimmed, &mut rt, &mut kb, &mut deferred);
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("Bye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

/// Handle REPL commands (starting with :)
/// Returns false if REPL should exit
fn handle_command(cmd: &str, rt: &mut Runtime, kb: &mut KnowledgeBase, text_data: &mut Option<TextData>, kb_data: &mut Option<KBData>, deferred: &mut DeferredStmts) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let command = parts[0];

    match command {
        ":quit" | ":q" | ":exit" => {
            println!("Bye!");
            return false;
        }

        ":help" | ":h" | ":?" => {
            print_help();
        }

        ":run" | ":r" => {
            let iterations = kb.forward_chain();
            println!("Forward chaining completed in {} iterations", iterations);
        }

        ":tensors" | ":t" => {
            let names = rt.tensor_names();
            if names.is_empty() {
                println!("No tensors defined");
            } else {
                println!("Tensors:");
                for name in names {
                    if let Some(t) = rt.get_tensor(name) {
                        println!("  {} : {:?}", name, t.dims());
                    }
                }
            }
        }

        ":relations" | ":rel" => {
            let names = kb.relation_names();
            if names.is_empty() {
                println!("No relations defined");
            } else {
                println!("Relations:");
                for name in names {
                    if let Some(rel) = kb.get_relation(name) {
                        println!("  {} : {} facts (arity {})", name, rel.len(), rel.arity);
                    }
                }
            }
        }

        ":show" | ":s" => {
            if parts.len() < 2 {
                println!("Usage: :show <name>");
            } else {
                let name = parts[1];
                // Try tensor first
                if let Some(t) = rt.get_tensor(name) {
                    println!("{} = {:?}", name, t);
                } else {
                    // Try relation
                    let facts = kb.query(name);
                    if facts.is_empty() {
                        println!("Unknown tensor or relation: {}", name);
                    } else {
                        println!("{}:", name);
                        for fact in facts {
                            println!("  {}({})", name, fact.join(", "));
                        }
                    }
                }
            }
        }

        ":print" | ":p" => {
            if parts.len() < 2 {
                println!("Usage: :print <name>");
            } else {
                let name = parts[1];
                if let Some(t) = rt.get_tensor(name) {
                    print_tensor(name, t);
                } else {
                    println!("Unknown tensor: {}", name);
                }
            }
        }

        ":zeros" => {
            // :zeros Name dim1 dim2 ...
            if parts.len() < 3 {
                println!("Usage: :zeros <name> <dim1> [dim2] ...");
            } else {
                let name = parts[1];
                let dims: std::result::Result<Vec<usize>, _> =
                    parts[2..].iter().map(|s| s.parse()).collect();
                match dims {
                    Ok(d) => {
                        let tensor = Tensor::zeros(&d[..], candle_core::DType::F32, rt.device());
                        match tensor {
                            Ok(t) => {
                                rt.set_tensor(name, t);
                                println!("Created tensor {} with shape {:?}", name, d);
                            }
                            Err(e) => println!("Error: {}", e),
                        }
                    }
                    Err(e) => println!("Invalid dimensions: {}", e),
                }
            }
        }

        ":ones" => {
            // :ones Name dim1 dim2 ...
            if parts.len() < 3 {
                println!("Usage: :ones <name> <dim1> [dim2] ...");
            } else {
                let name = parts[1];
                let dims: std::result::Result<Vec<usize>, _> =
                    parts[2..].iter().map(|s| s.parse()).collect();
                match dims {
                    Ok(d) => {
                        let tensor = Tensor::ones(&d[..], candle_core::DType::F32, rt.device());
                        match tensor {
                            Ok(t) => {
                                rt.set_tensor(name, t);
                                println!("Created tensor {} with shape {:?}", name, d);
                            }
                            Err(e) => println!("Error: {}", e),
                        }
                    }
                    Err(e) => println!("Invalid dimensions: {}", e),
                }
            }
        }

        ":rand" => {
            // :rand Name dim1 dim2 ...
            if parts.len() < 3 {
                println!("Usage: :rand <name> <dim1> [dim2] ...");
            } else {
                let name = parts[1];
                let dims: std::result::Result<Vec<usize>, _> =
                    parts[2..].iter().map(|s| s.parse()).collect();
                match dims {
                    Ok(d) => {
                        let tensor =
                            Tensor::rand(0.0f32, 1.0, &d[..], rt.device());
                        match tensor {
                            Ok(t) => {
                                rt.set_tensor(name, t);
                                println!("Created tensor {} with shape {:?}", name, d);
                            }
                            Err(e) => println!("Error: {}", e),
                        }
                    }
                    Err(e) => println!("Invalid dimensions: {}", e),
                }
            }
        }

        ":clear" => {
            *rt = Runtime::new();
            *kb = KnowledgeBase::new();
            *text_data = None;
            *kb_data = None;
            println!("Cleared all tensors, relations, and data");
        }

        ":save" => {
            // :save <path> [tensors|facts|all]
            if parts.len() < 2 {
                println!("Usage: :save <path.safetensors|path.facts> [tensors|facts|all]");
                println!("  .safetensors - saves tensor parameters");
                println!("  .facts       - saves Datalog facts");
            } else {
                let path = parts[1];
                let what = parts.get(2).copied().unwrap_or("auto");

                let save_tensors = match what {
                    "tensors" => true,
                    "facts" => false,
                    "all" => true,
                    _ => path.ends_with(".safetensors"),
                };
                let save_facts = match what {
                    "tensors" => false,
                    "facts" => true,
                    "all" => true,
                    _ => path.ends_with(".facts"),
                };

                if save_tensors {
                    let tensor_path = if path.ends_with(".safetensors") {
                        path.to_string()
                    } else {
                        format!("{}.safetensors", path)
                    };
                    match rt.save_checkpoint(&tensor_path) {
                        Ok(()) => println!("Saved {} parameters to {}", rt.param_names().len(), tensor_path),
                        Err(e) => println!("Error saving tensors: {}", e),
                    }
                }

                if save_facts {
                    let facts_path = if path.ends_with(".facts") {
                        path.to_string()
                    } else {
                        format!("{}.facts", path)
                    };
                    match kb.save_facts(&facts_path) {
                        Ok(()) => println!("Saved {} facts to {}", kb.total_facts(), facts_path),
                        Err(e) => println!("Error saving facts: {}", e),
                    }
                }

                if !save_tensors && !save_facts {
                    println!("Nothing to save. Use .safetensors or .facts extension.");
                }
            }
        }

        ":load_checkpoint" | ":lc" => {
            // :load_checkpoint <path.safetensors|path.facts>
            if parts.len() < 2 {
                println!("Usage: :load_checkpoint <path.safetensors|path.facts>");
            } else {
                let path = parts[1];

                if path.ends_with(".safetensors") {
                    match rt.load_checkpoint(path) {
                        Ok(count) => println!("Loaded {} tensors from {}", count, path),
                        Err(e) => println!("Error loading tensors: {}", e),
                    }
                } else if path.ends_with(".facts") {
                    match kb.load_facts(path) {
                        Ok(count) => println!("Loaded {} facts from {}", count, path),
                        Err(e) => println!("Error loading facts: {}", e),
                    }
                } else {
                    println!("Unknown file type. Use .safetensors or .facts extension.");
                }
            }
        }

        ":train" => {
            // :train LossTensor epochs=100 lr=0.001 optimizer=sgd
            if parts.len() < 2 {
                println!("Usage: :train <loss_tensor> [epochs=N] [lr=F] [optimizer=sgd|adamw]");
            } else {
                let loss_name = parts[1];

                // Parse optional kwargs
                let mut epochs = 100usize;
                let mut lr = 0.01f64;
                let mut optimizer = "sgd".to_string();
                let mut batch_size = 32usize;

                for part in &parts[2..] {
                    if let Some((key, value)) = part.split_once('=') {
                        match key {
                            "epochs" => epochs = value.parse().unwrap_or(100),
                            "lr" => lr = value.parse().unwrap_or(0.01),
                            "optimizer" => optimizer = value.to_string(),
                            "batch_size" => batch_size = value.parse().unwrap_or(32),
                            _ => println!("Unknown option: {}", key),
                        }
                    }
                }

                match run_training(rt, text_data, loss_name, epochs, lr, &optimizer, batch_size) {
                    Ok(final_loss) => {
                        println!("Training complete. Final loss: {:.6}", final_loss);
                    }
                    Err(e) => println!("Training error: {}", e),
                }
            }
        }

        ":ema" => {
            // :ema <target_param> <source_param> [tau=0.99]
            // Performs: target = tau * target + (1-tau) * source
            // Used for JEPA target encoder EMA updates
            if parts.len() < 3 {
                println!("Usage: :ema <target_param> <source_param> [tau=0.99]");
            } else {
                let target_name = parts[1];
                let source_name = parts[2];
                let mut tau = 0.99f32;

                // Parse optional tau
                for part in &parts[3..] {
                    if let Some((key, value)) = part.split_once('=') {
                        if key == "tau" {
                            tau = value.parse().unwrap_or(0.99);
                        }
                    }
                }

                // Get source and target parameters
                match (rt.get_param(target_name), rt.get_param(source_name)) {
                    (Some(target_var), Some(source_var)) => {
                        let target_tensor = target_var.as_tensor();
                        let source_tensor = source_var.as_tensor();

                        // Compute EMA update: target = tau * target + (1-tau) * source
                        let tau_scaled = target_tensor * tau as f64;
                        let one_minus_tau = 1.0 - tau;
                        let one_minus_tau_scaled = source_tensor * one_minus_tau as f64;

                        match tau_scaled.and_then(|t| one_minus_tau_scaled.and_then(|s| t.broadcast_add(&s))) {
                            Ok(updated) => {
                                if let Err(e) = target_var.set(&updated) {
                                    println!("Error updating {}: {}", target_name, e);
                                } else {
                                    rt.set_tensor(target_name, updated);
                                    println!("EMA update: {} <- {} (tau={})", target_name, source_name, tau);
                                }
                            }
                            Err(e) => println!("EMA computation error: {}", e),
                        }
                    }
                    (None, _) => println!("Parameter not found: {}", target_name),
                    (_, None) => println!("Parameter not found: {}", source_name),
                }
            }
        }

        ":load" | ":l" => {
            if parts.len() < 2 {
                println!("Usage: :load <file.ein>");
            } else {
                let file_path = parts[1];
                match load_file(file_path, rt, kb) {
                    Ok(file_deferred) => {
                        println!("Loaded: {}", file_path);
                        if !file_deferred.is_empty() {
                            println!("  {} deferred @forward statement(s)", file_deferred.len());
                            deferred.extend(file_deferred);
                        }
                    }
                    Err(e) => println!("Error loading file: {}", e),
                }
            }
        }

        ":load_text" => {
            // :load_text <file> [seq_len=32]
            if parts.len() < 2 {
                println!("Usage: :load_text <file.txt> [seq_len=N]");
            } else {
                let file_path = parts[1];
                let mut seq_len = 32usize;

                // Parse optional seq_len
                for part in &parts[2..] {
                    if let Some((key, value)) = part.split_once('=') {
                        if key == "seq_len" {
                            seq_len = value.parse().unwrap_or(32);
                        }
                    }
                }

                match load_text_file(file_path, seq_len, rt) {
                    Ok(td) => {
                        println!("Loaded text: {} chars, vocab_size={}, seq_len={}",
                            td.data.len(), td.vocab_size(), seq_len);
                        println!("Vocabulary: {:?}", &td.tokenizer.idx_to_char[..td.vocab_size().min(50)]);
                        *text_data = Some(td);
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
        }

        ":batch" => {
            // :batch [batch_size=4]
            if let Some(td) = text_data {
                let mut batch_size = 4usize;
                for part in &parts[1..] {
                    if let Some((key, value)) = part.split_once('=') {
                        if key == "batch_size" {
                            batch_size = value.parse().unwrap_or(4);
                        }
                    }
                }

                match td.get_batch(batch_size, rt.device()) {
                    Ok((inputs, targets)) => {
                        rt.set_tensor("Inputs", inputs.clone());
                        rt.set_tensor("Targets", targets.clone());
                        println!("Created batch: Inputs {:?}, Targets {:?}", inputs.dims(), targets.dims());
                    }
                    Err(e) => println!("Error creating batch: {}", e),
                }
            } else {
                println!("No text loaded. Use :load_text first.");
            }
        }

        ":generate" => {
            // :generate <seed_text> [length=100] [temperature=1.0]
            if let Some(td) = text_data {
                if parts.len() < 2 {
                    println!("Usage: :generate <seed_text> [length=N] [temperature=F]");
                } else {
                    let seed = parts[1];
                    let mut length = 100usize;
                    let mut temperature = 1.0f32;

                    for part in &parts[2..] {
                        if let Some((key, value)) = part.split_once('=') {
                            match key {
                                "length" => length = value.parse().unwrap_or(100),
                                "temperature" => temperature = value.parse().unwrap_or(1.0),
                                _ => {}
                            }
                        }
                    }

                    match generate_text(rt, td, seed, length, temperature) {
                        Ok(text) => println!("{}", text),
                        Err(e) => println!("Generation error: {}", e),
                    }
                }
            } else {
                println!("No text loaded. Use :load_text first.");
            }
        }

        ":vocab" => {
            // Show vocabulary info
            if let Some(td) = text_data {
                println!("Vocabulary size: {}", td.vocab_size());
                println!("Characters: {:?}", td.tokenizer.idx_to_char);
            } else {
                println!("No text loaded. Use :load_text first.");
            }
        }

        ":forward" | ":f" => {
            // Execute deferred forward pass equations and register for training
            if deferred.is_empty() {
                println!("No deferred statements. Use @forward to define them.");
            } else {
                println!("Evaluating {} deferred statement(s)...", deferred.len());
                for (lhs, rhs, src) in deferred.iter() {
                    match rt.eval_expr_to(lhs, rhs) {
                        Ok(()) => {
                            if let Some(t) = rt.get_tensor(&lhs.name) {
                                println!("  {} = {:?}", lhs.name, t.dims());
                            }
                            // Add the original source string to forward pass for training
                            rt.add_forward_statement(src);
                        }
                        Err(e) => {
                            println!("  Error evaluating {}: {}", lhs.name, e);
                        }
                    }
                }
            }
        }

        ":deferred" => {
            // Show deferred statements
            if deferred.is_empty() {
                println!("No deferred statements.");
            } else {
                println!("Deferred statements ({}):", deferred.len());
                for (lhs, _rhs, src) in deferred.iter() {
                    println!("  @forward {} (src: {})", lhs.name, src);
                }
            }
        }

        ":load_kb" => {
            // :load_kb <train.txt> [test=<test.txt>] [dim=128]
            if parts.len() < 2 {
                println!("Usage: :load_kb <train.txt> [test=<test.txt>] [dim=N]");
            } else {
                let train_path = parts[1];
                let mut test_path: Option<&str> = None;
                let mut dim = 128usize;

                for part in &parts[2..] {
                    if let Some((key, value)) = part.split_once('=') {
                        match key {
                            "test" => test_path = Some(value),
                            "dim" => dim = value.parse().unwrap_or(128),
                            _ => println!("Unknown option: {}", key),
                        }
                    }
                }

                match KBData::from_tsv(train_path, test_path, dim) {
                    Ok(kd) => {
                        println!("Loaded KB: {} entities, {} relations, {} train triples",
                            kd.num_entities(), kd.num_relations(), kd.train_triples.len());
                        if !kd.test_triples.is_empty() {
                            println!("  {} test triples", kd.test_triples.len());
                        }
                        // Set metadata as tensors for Ein programs
                        let num_entities = kd.num_entities();
                        let num_relations = kd.num_relations();
                        rt.set_tensor("NumEntities", Tensor::new(&[num_entities as f32], rt.device()).unwrap());
                        rt.set_tensor("NumRelations", Tensor::new(&[num_relations as f32], rt.device()).unwrap());
                        rt.set_tensor("EmbedDim", Tensor::new(&[dim as f32], rt.device()).unwrap());
                        *kb_data = Some(kd);
                    }
                    Err(e) => println!("Error loading KB: {}", e),
                }
            }
        }

        ":kb_batch" => {
            // :kb_batch [batch_size=128] [neg_ratio=10]
            if let Some(kd) = kb_data {
                let mut batch_size = 128usize;
                let mut neg_ratio = 10usize;

                for part in &parts[1..] {
                    if let Some((key, value)) = part.split_once('=') {
                        match key {
                            "batch_size" => batch_size = value.parse().unwrap_or(128),
                            "neg_ratio" => neg_ratio = value.parse().unwrap_or(10),
                            _ => {}
                        }
                    }
                }

                // Use a pseudo-random step for the batch
                let step = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as usize)
                    .unwrap_or(0);

                match kd.get_batch(batch_size, step, neg_ratio, rt.device()) {
                    Ok((heads, rels, tails, neg_tails)) => {
                        rt.set_tensor("Heads", heads.clone());
                        rt.set_tensor("Relations", rels.clone());
                        rt.set_tensor("Tails", tails.clone());
                        rt.set_tensor("NegTails", neg_tails.clone());
                        println!("Created KB batch: Heads {:?}, Relations {:?}, Tails {:?}, NegTails {:?}",
                            heads.dims(), rels.dims(), tails.dims(), neg_tails.dims());
                    }
                    Err(e) => println!("Error creating batch: {}", e),
                }
            } else {
                println!("No KB loaded. Use :load_kb first.");
            }
        }

        ":eval_kb" => {
            // :eval_kb [samples=1000]
            if let Some(kd) = kb_data {
                let mut num_samples = 1000usize;

                for part in &parts[1..] {
                    if let Some((key, value)) = part.split_once('=') {
                        if key == "samples" {
                            num_samples = value.parse().unwrap_or(1000);
                        }
                    }
                }

                // Get entity and relation embeddings
                let entity_emb = rt.get_tensor("EntityEmb");
                let relation_emb = rt.get_tensor("RelationEmb");

                match (entity_emb, relation_emb) {
                    (Some(e), Some(r)) => {
                        match kd.evaluate(e, r, num_samples) {
                            Ok((mrr, h1, h3, h10)) => {
                                println!("Link Prediction Results ({} test samples):", num_samples);
                                println!("  MRR:     {:.4}", mrr);
                                println!("  Hits@1:  {:.4}", h1);
                                println!("  Hits@3:  {:.4}", h3);
                                println!("  Hits@10: {:.4}", h10);
                            }
                            Err(e) => println!("Evaluation error: {}", e),
                        }
                    }
                    _ => {
                        println!("Missing embeddings. Define EntityEmb and RelationEmb tensors first.");
                    }
                }
            } else {
                println!("No KB loaded. Use :load_kb first.");
            }
        }

        ":train_kb" => {
            // :train_kb [epochs=100] [lr=0.001] [dim=128] [batch_size=128] [neg_ratio=10] [margin=1.0]
            if let Some(kd) = kb_data {
                let mut epochs = 100usize;
                let mut lr = 0.001f64;
                let mut batch_size = 128usize;
                let mut neg_ratio = 10usize;
                let mut margin = 1.0f32;

                for part in &parts[1..] {
                    if let Some((key, value)) = part.split_once('=') {
                        match key {
                            "epochs" => epochs = value.parse().unwrap_or(100),
                            "lr" => lr = value.parse().unwrap_or(0.001),
                            "batch_size" => batch_size = value.parse().unwrap_or(128),
                            "neg_ratio" => neg_ratio = value.parse().unwrap_or(10),
                            "margin" => margin = value.parse().unwrap_or(1.0),
                            _ => println!("Unknown option: {}", key),
                        }
                    }
                }

                match run_kb_training(rt, kd, epochs, lr, batch_size, neg_ratio, margin) {
                    Ok((mrr, h1, h3, h10)) => {
                        println!("Training complete.");
                        println!("Final metrics:");
                        println!("  MRR:     {:.4}", mrr);
                        println!("  Hits@1:  {:.4}", h1);
                        println!("  Hits@3:  {:.4}", h3);
                        println!("  Hits@10: {:.4}", h10);
                    }
                    Err(e) => println!("Training error: {}", e),
                }
            } else {
                println!("No KB loaded. Use :load_kb first.");
            }
        }

        _ => {
            println!("Unknown command: {}. Type :help for available commands.", command);
        }
    }

    true
}

/// Handle Ein statements (equations, facts, rules, queries)
fn handle_statement(input: &str, rt: &mut Runtime, kb: &mut KnowledgeBase, deferred: &mut DeferredStmts) {
    match parse(input) {
        Ok(stmt) => match &stmt {
            Statement::Equation { lhs, .. } => {
                // Tensor equation - evaluate in Runtime
                match rt.eval(input) {
                    Ok(()) => {
                        if let Some(t) = rt.get_tensor(&lhs.name) {
                            println!("{} = {:?}", lhs.name, t);
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            Statement::Fact(_) => {
                kb.add_fact_stmt(&stmt);
                println!("Fact added.");
            }

            Statement::Rule {
                head,
                body,
                constraints,
            } => {
                kb.add_rule(&stmt);
                let constraint_str = if constraints.is_empty() {
                    String::new()
                } else {
                    format!(" with {} constraint(s)", constraints.len())
                };
                println!(
                    "Rule added: {}(...) <- {} body atom(s){}",
                    head.name,
                    body.len(),
                    constraint_str
                );
            }

            Statement::Query(q) => {
                // Query the knowledge base with filters
                let results = kb.query_filtered(&q.name, &q.indices);
                if results.is_empty() {
                    println!("No results for {}.", q.name);
                } else {
                    println!("Results for {}:", q.name);
                    for fact in results {
                        println!("  {}({})", q.name, fact.join(", "));
                    }
                }
            }

            Statement::ParamDecl { name, type_spec, .. } => {
                // Parameter declaration - create in Runtime
                match rt.eval(input) {
                    Ok(()) => {
                        if let Some(t) = rt.get_tensor(name) {
                            println!(
                                "@param {} : {}[{}] = {:?}",
                                name,
                                type_spec.dtype,
                                type_spec.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
                                t.dims()
                            );
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            Statement::EmbeddingDecl { name, vocab_size, embed_dim } => {
                // Embedding declaration - create in Runtime
                match rt.eval(input) {
                    Ok(()) => {
                        if let Some(t) = rt.get_tensor(name) {
                            println!(
                                "@embedding {} : vocab={} dim={} = {:?}",
                                name,
                                vocab_size,
                                embed_dim,
                                t.dims()
                            );
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            Statement::ForwardDecl { lhs, rhs } => {
                // Deferred statement - store for later evaluation
                // Extract the source (input without @forward prefix)
                let src = input.strip_prefix("@forward").unwrap_or(input).trim().to_string();
                deferred.push((lhs.clone(), rhs.clone(), src));
                println!("@forward {} = ... (deferred, run :forward to evaluate)", lhs.name);
            }
        },
        Err(e) => {
            println!("Parse error: {}", e);
        }
    }
}

fn print_help() {
    println!(
        r#"Ein Commands:
  :help, :h, :?     Show this help
  :quit, :q         Exit the REPL
  :load <file>      Load and execute a .ein file
  :run, :r          Run forward chaining on rules
  :train <loss>     Train parameters (optimizer=sgd|adamw epochs=N lr=F)
  :tensors, :t      List all tensors
  :relations, :rel  List all relations
  :show <name>      Show raw tensor/relation
  :print, :p <name> Pretty-print tensor values
  :zeros <n> <d..>  Create zero tensor
  :ones <n> <d..>   Create ones tensor
  :rand <n> <d..>   Create random tensor
  :clear            Clear all state

Checkpoints:
  :save <path.safetensors>        Save model parameters
  :save <path.facts>              Save Datalog facts
  :load_checkpoint <path>, :lc    Load saved checkpoint

Language Modeling:
  :load_text <file> [seq_len=N]   Load text file for LM training
  :batch [batch_size=N]           Create training batch (Inputs, Targets)
  :generate <seed> [length=N]     Generate text from trained model
  :vocab                          Show vocabulary info

Ein Syntax:
  Y[i] = W[i,j] X[j]              Tensor equation (Einstein notation)
  Y[i] = sigmoid(W[i,j] X[j])     With activation function
  Parent(Alice, Bob).             Add a fact
  Ancestor(x,y) <- Parent(x,y)    Add a rule
  Sibling(x,y) <- Parent(p,x) Parent(p,y), x != y
                                  Rule with inequality constraint
  Ancestor(x, y)?                 Query all results
  Ancestor(Alice, x)?             Filtered query (Alice's descendants)

Command line:
  ein                             Start interactive REPL
  ein <file.ein>                  Execute file and show results
  ein <file.ein> --repl           Execute file then start REPL
"#
    );
}

/// Load and execute a .ein file
/// Returns any deferred @forward statements
fn load_file(path: &str, rt: &mut Runtime, kb: &mut KnowledgeBase) -> std::result::Result<DeferredStmts, String> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(format!("File not found: {}", path.display()));
    }

    let contents = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let mut deferred: DeferredStmts = Vec::new();

    for (line_num, line) in contents.lines().enumerate() {
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        // Handle commands in files (like :run) - skip text commands in file mode
        if trimmed.starts_with(':') {
            // For file loading, we don't support text/forward commands
            let cmd_parts: Vec<&str> = trimmed.split_whitespace().collect();
            let cmd = cmd_parts[0];
            if !matches!(cmd, ":load_text" | ":batch" | ":generate" | ":vocab" | ":forward" | ":f" | ":load_kb" | ":kb_batch" | ":eval_kb") {
                let mut dummy_text_data: Option<TextData> = None;
                let mut dummy_kb_data: Option<KBData> = None;
                let mut dummy_deferred: DeferredStmts = Vec::new();
                handle_command(trimmed, rt, kb, &mut dummy_text_data, &mut dummy_kb_data, &mut dummy_deferred);
            }
            continue;
        }

        // Parse and execute statement
        match parse(trimmed) {
            Ok(stmt) => match &stmt {
                Statement::Equation { .. } => {
                    if let Err(e) = rt.eval(trimmed) {
                        return Err(format!("Line {}: {}", line_num + 1, e));
                    }
                }
                Statement::Fact(_) => {
                    kb.add_fact_stmt(&stmt);
                }
                Statement::Rule { .. } => {
                    kb.add_rule(&stmt);
                }
                Statement::Query(q) => {
                    // Run forward chaining before query to ensure all derived facts exist
                    kb.forward_chain();
                    // Execute filtered query and print results
                    let results = kb.query_filtered(&q.name, &q.indices);
                    if results.is_empty() {
                        println!("Query {}({}) => No results", q.name, q.indices.join(", "));
                    } else {
                        println!("Query {}({}) => {} results:", q.name, q.indices.join(", "), results.len());
                        for fact in results {
                            println!("  {}({})", q.name, fact.join(", "));
                        }
                    }
                }
                Statement::ParamDecl { .. } => {
                    if let Err(e) = rt.eval(trimmed) {
                        return Err(format!("Line {}: {}", line_num + 1, e));
                    }
                }
                Statement::EmbeddingDecl { .. } => {
                    if let Err(e) = rt.eval(trimmed) {
                        return Err(format!("Line {}: {}", line_num + 1, e));
                    }
                }
                Statement::ForwardDecl { lhs, rhs } => {
                    // Store for later evaluation (strip @forward prefix)
                    let src = trimmed.strip_prefix("@forward").unwrap_or(trimmed).trim().to_string();
                    deferred.push((lhs.clone(), rhs.clone(), src));
                }
            },
            Err(e) => {
                return Err(format!("Line {}: Parse error: {}", line_num + 1, e));
            }
        }
    }

    Ok(deferred)
}

/// Pretty-print a tensor with its values
fn print_tensor(name: &str, t: &Tensor) {
    let dims = t.dims();
    print!("{} : {:?} = ", name, dims);

    match dims.len() {
        0 => {
            // Scalar
            if let Ok(v) = t.to_scalar::<f32>() {
                println!("{:.4}", v);
            } else {
                println!("{:?}", t);
            }
        }
        1 => {
            // 1D vector
            if let Ok(v) = t.to_vec1::<f32>() {
                let formatted: Vec<String> = v.iter().map(|x| format!("{:.4}", x)).collect();
                println!("[{}]", formatted.join(", "));
            } else {
                println!("{:?}", t);
            }
        }
        2 => {
            // 2D matrix
            if let Ok(v) = t.to_vec2::<f32>() {
                println!();
                for row in v {
                    let formatted: Vec<String> = row.iter().map(|x| format!("{:>8.4}", x)).collect();
                    println!("  [{}]", formatted.join(", "));
                }
            } else {
                println!("{:?}", t);
            }
        }
        _ => {
            // Higher dimensions - just show shape and raw debug
            println!("{:?}", t);
        }
    }
}

/// Run training loop with SGD or AdamW optimizer
fn run_training(
    rt: &mut Runtime,
    text_data: &mut Option<TextData>,
    loss_name: &str,
    epochs: usize,
    lr: f64,
    optimizer_type: &str,
    batch_size: usize,
) -> std::result::Result<f32, String> {
    // Check that we have parameters to train
    let params = rt.all_params();
    if params.is_empty() {
        return Err("No parameters to train. Define parameters with @param.".to_string());
    }

    let use_adamw = optimizer_type.to_lowercase() == "adamw";
    let opt_name = if use_adamw { "AdamW" } else { "SGD" };
    let has_text_data = text_data.is_some();

    println!(
        "Training {} parameters for {} epochs with {}(lr={}){}",
        params.len(),
        epochs,
        opt_name,
        lr,
        if has_text_data { " [random batches]" } else { "" }
    );

    // Create AdamW optimizer if requested
    let mut adamw_opt = if use_adamw {
        let adamw_params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        Some(
            AdamW::new(params.clone(), adamw_params)
                .map_err(|e| format!("Failed to create AdamW optimizer: {}", e))?
        )
    } else {
        None
    };

    let mut final_loss = 0.0f32;

    for epoch in 0..epochs {
        // Get a new random batch for each epoch if text_data is available
        if let Some(ref td) = text_data {
            let (inputs, targets) = td.get_batch_at_step(batch_size, epoch, rt.device())
                .map_err(|e| format!("Batch error: {}", e))?;
            rt.set_tensor("Inputs", inputs);
            rt.set_tensor("Targets", targets);
        }

        // Re-run forward pass to build computation graph
        rt.run_forward_pass()
            .map_err(|e| format!("Forward pass error: {}", e))?;

        // Get the loss tensor
        let loss = rt
            .get_tensor(loss_name)
            .ok_or_else(|| format!("Loss tensor '{}' not found", loss_name))?
            .clone();

        // Get loss value for logging
        let loss_val: f32 = loss
            .to_scalar()
            .map_err(|e| format!("Loss must be a scalar: {}", e))?;
        final_loss = loss_val;

        // Compute gradients via backpropagation
        let grads = loss
            .backward()
            .map_err(|e| format!("Backward pass error: {}", e))?;

        // Apply gradients using chosen optimizer
        if let Some(ref mut opt) = adamw_opt {
            // AdamW step
            opt.step(&grads)
                .map_err(|e| format!("AdamW step error: {}", e))?;
            // Sync updated params back to runtime tensors
            rt.sync_params_from_vars()
                .map_err(|e| format!("Param sync error: {}", e))?;
        } else {
            // SGD update
            rt.apply_gradients(&grads, lr)
                .map_err(|e| format!("Gradient update error: {}", e))?;
        }

        // Print progress every 10% of epochs
        if epoch % (epochs / 10).max(1) == 0 || epoch == epochs - 1 {
            println!("  Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss_val);
        }
    }

    Ok(final_loss)
}

/// Print the current state (relations and tensors)
fn print_state(rt: &Runtime, kb: &KnowledgeBase) {
    // Print relations
    let rel_names = kb.relation_names();
    if !rel_names.is_empty() {
        println!("\nRelations:");
        for name in rel_names {
            let facts = kb.query(name);
            if !facts.is_empty() {
                println!("  {}:", name);
                for fact in facts {
                    println!("    {}({})", name, fact.join(", "));
                }
            }
        }
    }

    // Print tensors
    let tensor_names = rt.tensor_names();
    if !tensor_names.is_empty() {
        println!("\nTensors:");
        for name in tensor_names {
            if let Some(t) = rt.get_tensor(name) {
                println!("  {} : {:?}", name, t.dims());
            }
        }
    }
}

/// Load a text file and create TextData for language modeling
fn load_text_file(path: &str, seq_len: usize, _rt: &Runtime) -> std::result::Result<TextData, String> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(format!("File not found: {}", path.display()));
    }

    let text = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    if text.len() < seq_len + 1 {
        return Err(format!("Text too short. Need at least {} characters, got {}",
            seq_len + 1, text.len()));
    }

    Ok(TextData::from_text(&text, seq_len))
}

/// Generate text using the current model
/// Expects a "Logits" tensor to be defined that computes logits from an "Input" tensor
fn generate_text(
    rt: &mut Runtime,
    text_data: &TextData,
    seed: &str,
    length: usize,
    temperature: f32,
) -> std::result::Result<String, String> {
    // Encode seed text
    let mut indices: Vec<usize> = text_data.tokenizer.encode(seed);
    if indices.is_empty() {
        return Err("Seed text contains no valid characters".to_string());
    }

    // Generate one character at a time
    for _ in 0..length {
        // Take the last token as input (simple approach)
        let last_idx = *indices.last().unwrap();
        let input_data = vec![last_idx as f32];
        let input_tensor = Tensor::new(input_data, rt.device())
            .map_err(|e| format!("Failed to create input tensor: {}", e))?
            .reshape(&[1, 1])  // Shape [batch=1, seq=1] for model compatibility
            .map_err(|e| format!("Failed to reshape input: {}", e))?;

        // Set as Inputs tensor (model expects plural)
        rt.set_tensor("Inputs", input_tensor);
        // Also need a dummy target for the forward pass
        let target_tensor = Tensor::zeros(&[1, 1], candle_core::DType::F32, rt.device())
            .map_err(|e| format!("Failed to create target tensor: {}", e))?;
        rt.set_tensor("Targets", target_tensor);

        // Run forward pass
        rt.run_forward_pass()
            .map_err(|e| format!("Forward pass error: {}", e))?;

        // Get logits
        let logits = rt.get_tensor("Logits")
            .ok_or_else(|| "Logits tensor not found. Define a model that outputs to 'Logits'.".to_string())?
            .clone();

        // Apply temperature and sample
        let next_idx = sample_from_logits(&logits, temperature)
            .map_err(|e| format!("Sampling error: {}", e))?;

        indices.push(next_idx);
    }

    // Decode back to text
    Ok(text_data.tokenizer.decode(&indices))
}

/// Sample from logits with temperature
fn sample_from_logits(logits: &Tensor, temperature: f32) -> std::result::Result<usize, String> {
    use std::time::{SystemTime, UNIX_EPOCH};

    // Get logits as vector
    let logits_flat = logits.flatten_all()
        .map_err(|e| format!("Failed to flatten logits: {}", e))?;
    let logits_vec: Vec<f32> = logits_flat.to_vec1()
        .map_err(|e| format!("Failed to convert logits: {}", e))?;

    if logits_vec.is_empty() {
        return Err("Empty logits".to_string());
    }

    // Apply temperature
    let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

    // Sample from the distribution using system time as seed
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345);

    // Simple LCG random number generator
    static mut RNG_STATE: u64 = 0;
    unsafe {
        RNG_STATE = RNG_STATE.wrapping_add(seed);
        RNG_STATE = RNG_STATE.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rand_val = (RNG_STATE >> 33) as f32 / (u32::MAX as f32);

        // Sample from categorical distribution
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand_val < cumsum {
                return Ok(i);
            }
        }
    }

    // Fallback to last index
    Ok(probs.len() - 1)
}

/// Run knowledge base embedding training (DistMult model with margin loss)
fn run_kb_training(
    rt: &mut Runtime,
    kb_data: &KBData,
    epochs: usize,
    lr: f64,
    batch_size: usize,
    neg_ratio: usize,
    margin: f32,
) -> std::result::Result<(f64, f64, f64, f64), String> {
    use candle_core::Var;

    let num_entities = kb_data.num_entities();
    let num_relations = kb_data.num_relations();
    let dim = kb_data.dim;
    let device = rt.device().clone();

    println!(
        "Training DistMult on {} entities, {} relations, {} triples",
        num_entities, num_relations, kb_data.train_triples.len()
    );
    println!(
        "  dim={}, epochs={}, lr={}, batch_size={}, neg_ratio={}, margin={}",
        dim, epochs, lr, batch_size, neg_ratio, margin
    );

    // Initialize embeddings with small random values
    let entity_emb = Var::randn(0.0f32, 0.1, &[num_entities, dim], &device)
        .map_err(|e| format!("Failed to create entity embeddings: {}", e))?;
    let relation_emb = Var::randn(0.0f32, 0.1, &[num_relations, dim], &device)
        .map_err(|e| format!("Failed to create relation embeddings: {}", e))?;

    // Create optimizer with the Vars directly
    let all_vars = vec![entity_emb.clone(), relation_emb.clone()];

    let adamw_params = ParamsAdamW {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };
    let mut optimizer = AdamW::new(all_vars, adamw_params)
        .map_err(|e| format!("Failed to create optimizer: {}", e))?;

    let num_batches = kb_data.train_triples.len() / batch_size;
    let mut final_loss = 0.0f32;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for batch_idx in 0..num_batches.max(1) {
            let step = epoch * num_batches + batch_idx;

            // Get batch of training triples with negative samples
            let (heads, relations, tails, neg_tails) = kb_data
                .get_batch(batch_size, step, neg_ratio, &device)
                .map_err(|e| format!("Batch error: {}", e))?;

            // Convert f32 indices to u32 for index_select
            let heads_u32 = heads.to_dtype(candle_core::DType::U32)
                .map_err(|e| format!("Heads dtype: {}", e))?;
            let relations_u32 = relations.to_dtype(candle_core::DType::U32)
                .map_err(|e| format!("Relations dtype: {}", e))?;
            let tails_u32 = tails.to_dtype(candle_core::DType::U32)
                .map_err(|e| format!("Tails dtype: {}", e))?;
            let neg_tails_u32 = neg_tails.to_dtype(candle_core::DType::U32)
                .map_err(|e| format!("NegTails dtype: {}", e))?;

            // Batched embedding lookup using index_select
            let entity_emb_tensor = entity_emb.as_tensor();
            let relation_emb_tensor = relation_emb.as_tensor();

            // Get head, relation, tail embeddings: [batch_size, dim]
            let h_emb = entity_emb_tensor.index_select(&heads_u32, 0)
                .map_err(|e| format!("h_emb index_select: {}", e))?;
            let r_emb = relation_emb_tensor.index_select(&relations_u32, 0)
                .map_err(|e| format!("r_emb index_select: {}", e))?;
            let t_emb = entity_emb_tensor.index_select(&tails_u32, 0)
                .map_err(|e| format!("t_emb index_select: {}", e))?;

            // DistMult positive scores: sum(h * r * t, dim=-1) -> [batch_size]
            let pos_scores = ((&h_emb * &r_emb)
                .map_err(|e| format!("h*r: {}", e))?
                * &t_emb)
                .map_err(|e| format!("(h*r)*t: {}", e))?
                .sum(1)
                .map_err(|e| format!("sum: {}", e))?;

            // Get negative tail embeddings: [batch_size, neg_ratio, dim]
            let neg_tails_flat = neg_tails_u32.flatten_all()
                .map_err(|e| format!("neg_tails flatten: {}", e))?;
            let neg_t_emb_flat = entity_emb_tensor.index_select(&neg_tails_flat, 0)
                .map_err(|e| format!("neg_t_emb index_select: {}", e))?;
            let neg_t_emb = neg_t_emb_flat.reshape(&[batch_size, neg_ratio, dim])
                .map_err(|e| format!("neg_t_emb reshape: {}", e))?;

            // Expand h_emb and r_emb for broadcasting: [batch_size, neg_ratio, dim]
            let h_emb_exp = h_emb.unsqueeze(1)
                .map_err(|e| format!("h_emb unsqueeze: {}", e))?
                .broadcast_as(&[batch_size, neg_ratio, dim])
                .map_err(|e| format!("h_emb broadcast: {}", e))?;
            let r_emb_exp = r_emb.unsqueeze(1)
                .map_err(|e| format!("r_emb unsqueeze: {}", e))?
                .broadcast_as(&[batch_size, neg_ratio, dim])
                .map_err(|e| format!("r_emb broadcast: {}", e))?;

            // DistMult negative scores: [batch_size, neg_ratio]
            let neg_scores = ((&h_emb_exp * &r_emb_exp)
                .map_err(|e| format!("neg h*r: {}", e))?
                * &neg_t_emb)
                .map_err(|e| format!("neg (h*r)*t: {}", e))?
                .sum(2)
                .map_err(|e| format!("neg sum: {}", e))?;

            // Margin loss: max(0, margin - pos_score + neg_score)
            let pos_expanded = pos_scores.unsqueeze(1)
                .map_err(|e| format!("Expand pos: {}", e))?
                .broadcast_as(&[batch_size, neg_ratio])
                .map_err(|e| format!("Broadcast pos: {}", e))?;

            let margin_tensor = Tensor::new(&[margin], &device)
                .map_err(|e| format!("Margin tensor: {}", e))?
                .broadcast_as(&[batch_size, neg_ratio])
                .map_err(|e| format!("Broadcast margin: {}", e))?;

            let loss_per_sample = (margin_tensor - &pos_expanded + &neg_scores)
                .map_err(|e| format!("Loss calc: {}", e))?
                .relu()
                .map_err(|e| format!("Relu: {}", e))?;

            let loss = loss_per_sample.mean_all()
                .map_err(|e| format!("Mean: {}", e))?;

            let loss_val: f32 = loss.to_scalar()
                .map_err(|e| format!("Loss to scalar: {}", e))?;
            epoch_loss += loss_val;

            // Backward pass
            let grads = loss.backward()
                .map_err(|e| format!("Backward: {}", e))?;

            // Optimizer step
            optimizer.step(&grads)
                .map_err(|e| format!("Optimizer step: {}", e))?;
        }

        final_loss = epoch_loss / num_batches.max(1) as f32;

        // Print progress
        if epoch % (epochs / 10).max(1) == 0 || epoch == epochs - 1 {
            println!("  Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, final_loss);
        }
    }

    // Store trained embeddings in runtime
    let entity_emb_final = entity_emb.as_tensor().clone();
    let relation_emb_final = relation_emb.as_tensor().clone();
    rt.set_tensor("EntityEmb", entity_emb_final.clone());
    rt.set_tensor("RelationEmb", relation_emb_final.clone());

    // Evaluate on test set
    let num_test_samples = kb_data.test_triples.len().min(1000);
    let (mrr, h1, h3, h10) = kb_data.evaluate(&entity_emb_final, &relation_emb_final, num_test_samples)
        .map_err(|e| format!("Evaluation error: {}", e))?;

    Ok((mrr, h1, h3, h10))
}