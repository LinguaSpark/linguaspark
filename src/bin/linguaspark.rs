use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use linguaspark::{
    Asset, DecodeOptions, LoadOptions, ModelArchive, ModelAssets, Translator, VocabularyAssets,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("linguaspark: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let command = args.next().ok_or_else(usage)?;
    let args = args.collect::<Vec<_>>();
    match command.to_str() {
        Some("inspect") => inspect(&args),
        Some("translate") => translate(args, false),
        Some("bench") => translate(args, true),
        _ => Err(usage().into()),
    }
}

fn inspect(args: &[OsString]) -> Result<(), Box<dyn std::error::Error>> {
    let [path] = args else {
        return Err("usage: linguaspark inspect MODEL[.gz]".into());
    };
    let model = ModelArchive::load(read_asset(path)?, None)?;
    println!("model version: {}", model.config.version);
    println!("architecture: {}", model.config.model_type);
    println!("embedding dimension: {}", model.config.dim_emb);
    println!("encoder layers: {}", model.config.enc_depth);
    println!("decoder layers: {}", model.config.dec_depth);
    println!("parameters: {} tensors", model.tensor_count());
    Ok(())
}

fn translate(args: Vec<OsString>, benchmark: bool) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 4 {
        return Err(
            "usage: linguaspark translate MODEL SRC_VOCAB TRG_VOCAB SHORTLIST [--beam-size N] [TEXT]"
                .into(),
        );
    }
    let mut args = args.into_iter();
    let model = args.next().ok_or_else(usage)?;
    let source_vocab = args.next().ok_or_else(usage)?;
    let target_vocab = args.next().ok_or_else(usage)?;
    let shortlist = args.next().ok_or_else(usage)?;
    let mut decode = DecodeOptions::default();
    let mut text_parts = Vec::new();
    while let Some(argument) = args.next() {
        if argument == "--beam-size" {
            let value = args.next().ok_or("--beam-size requires a value")?;
            decode.beam_size = value.to_string_lossy().parse()?;
        } else {
            text_parts.push(argument.to_string_lossy().into_owned());
        }
    }

    let load_start = Instant::now();
    let vocabularies = if source_vocab == target_vocab {
        VocabularyAssets::Shared(read_asset(&source_vocab)?)
    } else {
        VocabularyAssets::Split {
            source: read_asset(&source_vocab)?,
            target: read_asset(&target_vocab)?,
        }
    };
    let translator = Translator::from_assets(
        ModelAssets {
            model: read_asset(&model)?,
            vocabularies,
            shortlist: read_asset(&shortlist)?,
        },
        LoadOptions::default(),
    )?;
    let load_elapsed = load_start.elapsed();

    let input = if text_parts.is_empty() {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        input
    } else {
        text_parts.join(" ")
    };
    let lines = input.lines().collect::<Vec<_>>();

    if benchmark {
        let warmup = translator.translate_many(&lines, &decode)?;
        std::hint::black_box(warmup);
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(translator.translate_many(&lines, &decode)?);
        }
        let elapsed = start.elapsed();
        println!("load_ms={:.3}", load_elapsed.as_secs_f64() * 1000.0);
        println!(
            "translate_ms={:.3}",
            elapsed.as_secs_f64() * 1000.0 / f64::from(iterations)
        );
        println!("sentences={}", lines.len());
        if let Some((rss_kib, peak_kib)) = linux_memory_kib() {
            println!("rss_mib={:.3}", rss_kib as f64 / 1024.0);
            println!("peak_rss_mib={:.3}", peak_kib as f64 / 1024.0);
        }
    } else {
        for translation in translator.translate_many(&lines, &decode)? {
            println!("{}", translation.text);
        }
    }
    Ok(())
}

fn linux_memory_kib() -> Option<(u64, u64)> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let value = |name: &str| {
        status.lines().find_map(|line| {
            let rest = line.strip_prefix(name)?;
            rest.split_whitespace().next()?.parse::<u64>().ok()
        })
    };
    Some((value("VmRSS:")?, value("VmHWM:")?))
}

fn read_asset(path: impl AsRef<Path>) -> Result<Asset, io::Error> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    if path.to_string_lossy().ends_with(".gz") {
        Ok(Asset::gzip(bytes))
    } else {
        Ok(Asset::raw(bytes))
    }
}

fn usage() -> &'static str {
    "usage:\n  linguaspark inspect MODEL[.gz]\n  linguaspark translate MODEL SRC_VOCAB TRG_VOCAB SHORTLIST [--beam-size N] [TEXT]\n  linguaspark bench MODEL SRC_VOCAB TRG_VOCAB SHORTLIST [--beam-size N] [TEXT]"
}
