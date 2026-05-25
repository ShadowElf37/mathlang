mod lexer;
mod ast;
mod parser;
mod eval;
mod repl;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        repl::run_repl();
        return;
    }
    let mut env = eval::Env::new();
    let ok = repl::eval_line(&args.join(" "), &mut env, false);
    if !ok { std::process::exit(1); }
}
