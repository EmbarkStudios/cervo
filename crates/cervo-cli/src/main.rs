/*!
Command line tools for cervo.
*/

mod commands;

use anyhow::Result;
use clap::Parser;
use commands::Command;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cervo {
    #[clap(subcommand)]
    command: Command,
}

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .try_init();

    let args = Cervo::parse();
    commands::run(args.command)
}
