/*!
Command line tools for tractor.
*/

mod commands;

use anyhow::Result;
use clap::Parser;
use commands::Command;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Tractor {
    #[clap(subcommand)]
    command: Command,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Tractor::parse();

    commands::run(args.command)
}
