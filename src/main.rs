mod lexer;
mod ast;
mod parser;
mod eval;
mod repl;
mod graph;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        repl::run_repl();
        return;
    }
    let expr = args.join(" ");
    if expr.starts_with('!') {
        eprintln!("! commands are REPL-only. Run 'm' with no arguments to enter the REPL.");
        std::process::exit(1);
    }
    let mut env = eval::Env::new();
    let ok = repl::eval_line(&expr, &mut env, false);
    if !ok { std::process::exit(1); }
}
