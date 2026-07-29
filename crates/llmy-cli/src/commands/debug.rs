use std::path::PathBuf;

use chrono::{DateTime, Local};
use clap::Args;
use color_eyre::eyre::eyre;
use llmy_client::debug::{LLMDebugRow, Sqlite3DebugDB};
use serde::Serialize;

fn format_timestamp(ts: i64) -> String {
    DateTime::from_timestamp(ts, 0)
        .map(|dt| {
            dt.with_timezone(&Local)
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        })
        .unwrap_or_else(|| ts.to_string())
}

#[derive(Args)]
pub struct ListReqArgs {
    /// SQLite debug database. Falls back to `LLM_DEBUG` when omitted.
    #[arg(short, long, env = "LLM_DEBUG")]
    db: String,

    /// Filter to a specific client id. Defaults to the most recent client.
    #[arg(long)]
    client_id: Option<i64>,

    /// Filter to a specific cache key.
    #[arg(long)]
    cache_key: Option<String>,

    /// Filter to a specific debug prefix (the label each call was recorded
    /// under, e.g. `chat` or `planner`).
    #[arg(long)]
    debug_prefix: Option<String>,
}

pub async fn run_list_req(args: ListReqArgs) -> color_eyre::Result<()> {
    let store = Sqlite3DebugDB::open_existing(&args.db).await?;

    let client_id = match args.client_id {
        Some(id) => Some(id),
        None => store.latest_client_id().await?,
    };
    if client_id.is_none() {
        eprintln!(
            "warning: no client rows in {}; listing without filter",
            args.db
        );
    } else {
        println!("Listing for client id {}", client_id.unwrap());
    }

    let rows = store
        .list_filtered(
            client_id,
            args.cache_key.as_deref(),
            args.debug_prefix.as_deref(),
        )
        .await?;

    println!("{}", LIST_HEADER);
    for row in rows {
        println!("{}", format_row_summary(&row));
    }
    Ok(())
}

#[derive(Args)]
pub struct DumpReqArgs {
    /// SQLite debug database. Falls back to `LLM_DEBUG` when omitted.
    #[arg(short, long, env = "LLM_DEBUG")]
    db: String,

    /// Row id from the `llm_debug` table.
    #[arg(short, long)]
    req_id: i64,

    /// Print the row as a single JSON object instead of the human view.
    #[arg(short, long, default_value_t = false)]
    json: bool,
}

#[derive(Args)]
pub struct DumpClientArgs {
    /// SQLite debug database. Falls back to `LLM_DEBUG` when omitted.
    #[arg(short, long, env = "LLM_DEBUG")]
    db: String,

    /// Client id to dump. Defaults to the most recent one.
    #[arg(short, long)]
    client_id: Option<i64>,

    /// Output folder. Created if missing.
    #[arg(short, long)]
    output: PathBuf,
}

pub async fn run_dump_client(args: DumpClientArgs) -> color_eyre::Result<()> {
    let store = Sqlite3DebugDB::open_existing(&args.db).await?;

    let client_id = match args.client_id {
        Some(id) => id,
        None => store
            .latest_client_id()
            .await?
            .ok_or_else(|| eyre!("no client rows in {}", args.db))?,
    };

    std::fs::create_dir_all(&args.output)?;
    let rows = store.list_filtered(Some(client_id), None, None).await?;
    if rows.is_empty() {
        eprintln!("no rows for client_id={} in {}", client_id, args.db);
        return Ok(());
    }

    let width = (rows.last().map(|r| r.id).unwrap_or(0).max(1) as f64)
        .log10()
        .floor() as usize
        + 1;
    let width = width.max(6);

    for row in &rows {
        let stem = format!(
            "{}-{:0>width$}",
            sanitize(&row.debug_prefix),
            row.id,
            width = width
        );
        let xml_path = args.output.join(format!("{stem}.xml"));
        let jsonl_path = args.output.join(format!("{stem}.jsonl"));

        std::fs::write(&xml_path, build_conversation_xml(row))?;
        std::fs::write(&jsonl_path, build_jsonl(row)?)?;
    }
    println!(
        "dumped {} request(s) for client_id={} into {}",
        rows.len(),
        client_id,
        args.output.display()
    );
    Ok(())
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn build_conversation_xml(row: &LLMDebugRow) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "<!-- id={} client_id={} model={} endpoint={} debug_prefix={} timestamp={} -->\n",
        row.id,
        row.client_id,
        row.model_name,
        row.endpoint,
        row.debug_prefix,
        format_timestamp(row.timestamp),
    ));
    s.push_str(&row.full_conversation);
    if !s.ends_with('\n') {
        s.push('\n');
    }
    s
}

#[derive(Serialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
enum JsonlEntry {
    Request(serde_json::Value),
    Response(serde_json::Value),
    Billing(BillingEntry),
}

#[derive(Serialize)]
struct BillingEntry {
    input_without_cached_tokens: Option<i64>,
    cached_tokens: Option<i64>,
    cache_write_tokens: Option<i64>,
    output_without_reasoning_tokens: Option<i64>,
    reasoning_tokens: Option<i64>,
    current_usage_usd: f64,
    cap_usd: f64,
}

fn build_jsonl(row: &LLMDebugRow) -> color_eyre::Result<String> {
    let req: serde_json::Value = serde_json::from_str(&row.raw_req)?;
    let resp: serde_json::Value = match row.raw_resp.as_deref() {
        Some(s) => serde_json::from_str(s)?,
        None => serde_json::Value::Null,
    };
    let entries = [
        JsonlEntry::Request(req),
        JsonlEntry::Response(resp),
        JsonlEntry::Billing(BillingEntry {
            input_without_cached_tokens: row.input_without_cached_tokens,
            cached_tokens: row.cached_tokens,
            cache_write_tokens: row.cache_write_tokens,
            output_without_reasoning_tokens: row.output_without_reasoning_tokens,
            reasoning_tokens: row.reasoning_tokens,
            current_usage_usd: row.current_usage_usd,
            cap_usd: row.cap_usd,
        }),
    ];

    let mut out = String::new();
    for entry in &entries {
        out.push_str(&serde_json::to_string(entry)?);
        out.push('\n');
    }
    Ok(out)
}

pub async fn run_dump_req(args: DumpReqArgs) -> color_eyre::Result<()> {
    let store = Sqlite3DebugDB::open_existing(&args.db).await?;

    let row = store
        .get_row(args.req_id)
        .await?
        .ok_or_else(|| eyre!("no llm_debug row with id {}", args.req_id))?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&row)?);
    } else {
        print_row_human(&row);
    }
    Ok(())
}

const LIST_HEADER: &str = "id\tclient\tts\tmodel\tendpoint\tdeployment\tcache_key\tinput\tcached\tcache_write\toutput\treasoning\tusage_usd\tresp";

fn format_row_summary(row: &LLMDebugRow) -> String {
    let cache = row.cache_key.as_deref().unwrap_or("-");
    let deploy = row.azure_deployment.as_deref().unwrap_or("-");
    let resp = if row.raw_resp.is_some() {
        "ok"
    } else {
        "pending"
    };
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}",
        row.id,
        row.client_id,
        format_timestamp(row.timestamp),
        row.model_name,
        row.endpoint,
        deploy,
        cache,
        opt_i64(row.input_without_cached_tokens),
        opt_i64(row.cached_tokens),
        opt_i64(row.cache_write_tokens),
        opt_i64(row.output_without_reasoning_tokens),
        opt_i64(row.reasoning_tokens),
        row.current_usage_usd,
        resp,
    )
}

fn print_row_human(row: &LLMDebugRow) {
    println!("=== Metadata ===");
    println!("id:                 {}", row.id);
    println!("client_id:          {}", row.client_id);
    println!("model_name:         {}", row.model_name);
    println!("endpoint:           {}", row.endpoint);
    println!(
        "azure_deployment:   {}",
        row.azure_deployment.as_deref().unwrap_or("-")
    );
    println!(
        "cache_key:          {}",
        row.cache_key.as_deref().unwrap_or("-")
    );
    println!("timestamp:          {}", format_timestamp(row.timestamp));
    println!(
        "input_tokens:       {}",
        opt_i64(row.input_without_cached_tokens)
    );
    println!("cached_tokens:      {}", opt_i64(row.cached_tokens));
    println!("cache_write_tokens: {}", opt_i64(row.cache_write_tokens));
    println!(
        "output_tokens:      {}",
        opt_i64(row.output_without_reasoning_tokens)
    );
    println!("reasoning_tokens:   {}", opt_i64(row.reasoning_tokens));
    println!("current_usage_usd:  {}", row.current_usage_usd);
    println!("cap_usd:            {}", row.cap_usd);
    println!();
    println!("=== Conversation ===");
    println!("{}", row.full_conversation);
    if let Some(text) = &row.resp_text_content {
        println!();
        println!("=== Response Text ===");
        println!("{}", text);
    }
}

fn opt_i64(v: Option<i64>) -> String {
    match v {
        Some(v) => v.to_string(),
        None => "-".to_string(),
    }
}
