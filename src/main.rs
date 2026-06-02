mod lexer;
mod ast;
mod parser;
mod vm;
mod eval;
mod ns;
mod repl;
mod graph;
mod animate;
#[cfg(feature = "gpu")]
mod gpu;

fn main() {
    // Run the evaluator on a worker thread with a large stack. mathlang's only
    // iteration construct is recursion, so deep recursion is normal; the default
    // 8 MB main-thread stack overflows (and aborts the process) at ~1500 frames.
    // A 1 GB stack lets the catchable recursion-depth guard in eval.rs trip first.
    let worker = std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(run)
        .expect("failed to spawn evaluator thread");
    match worker.join() {
        Ok(code) => std::process::exit(code),
        Err(_)   => std::process::exit(1), // panic already reported
    }
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        repl::run_repl();
        return 0;
    }

    // Support:  m -f file.math ['expr']
    //   -f file.math           — load file, show nothing (useful as init)
    //   -f file.math 'expr'    — load file, then evaluate expr
    if args[0] == "-f" {
        if args.len() < 2 {
            eprintln!("usage: m -f <file.math> [expression]");
            return 1;
        }
        let path = &args[1];
        let mut env = eval::Env::new();
        repl::import_file(path, path, &mut env, false);
        if args.len() >= 3 {
            let expr = args[2..].join(" ");
            let ok = repl::eval_line(&expr, &mut env, false);
            if !ok { return 1; }
        }
        return 0;
    }

    let expr = args.join(" ");
    if expr.starts_with('!') {
        eprintln!("! commands are REPL-only. Run 'm' with no arguments to enter the REPL.");
        return 1;
    }
    let mut env = eval::Env::new();
    let ok = repl::eval_line(&expr, &mut env, false);
    if !ok { return 1; }
    0
}
