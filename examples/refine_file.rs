use std::env;
use std::fs;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: refine_file <input.txt> [model] [agents]");
        std::process::exit(1);
    }

    let text = fs::read_to_string(&args[1])?;
    let model = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or(glassrip::stitch::refine::DEFAULT_REFINE_MODEL);
    let agents: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    let result = glassrip::stitch::refine::refine_text(&text, model, agents).await?;

    let out_path = format!("{}.refined.txt", args[1].trim_end_matches(".txt"));
    fs::write(&out_path, &result)?;
    println!("Written to {out_path}");

    Ok(())
}
