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

    // Support:  m -f file.math ['expr']
    //   -f file.math           — load file, show nothing (useful as init)
    //   -f file.math 'expr'    — load file, then evaluate expr
    if args[0] == "-f" {
        if args.len() < 2 {
            eprintln!("usage: m -f <file.math> [expression]");
            std::process::exit(1);
        }
        let path = &args[1];
        let mut env = eval::Env::new();
        repl::import_file(path, path, &mut env, false);
        if args.len() >= 3 {
            let expr = args[2..].join(" ");
            let ok = repl::eval_line(&expr, &mut env, false);
            if !ok { std::process::exit(1); }
        }
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
