/*!
Command line tools for cervo.
*/

#![allow(unsafe_code)]
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
    tracing_subscriber::fmt::init();
    let args = Cervo::parse();

    commands::run(args.command)
}
