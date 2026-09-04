//! Read-only file tools for the harness: Read (numbered lines), Grep
//! (regex search across the tree), Stat (metadata / directory listing) and
//! Search (glob file finding). There is deliberately no write or edit tool —
//! the harness output channel is structured output, not file mutation.

use std::path::{Path, PathBuf};

use globset::{Glob, GlobSetBuilder};
use grep::pcre2::RegexMatcherBuilder;
use grep::searcher::sinks::UTF8;
use grep::searcher::{BinaryDetection, SearcherBuilder};
use ignore::WalkBuilder;
use llmy_types::error::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;

const DEFAULT_READ_LINES: usize = 2_000;
const MAX_LINE_CHARS: usize = 2_000;
const DEFAULT_GREP_MATCHES: usize = 200;
const DEFAULT_SEARCH_RESULTS: usize = 500;
const MAX_DIR_ENTRIES: usize = 500;

/// Shared root-anchored path handling for the read-only file tools.
#[derive(Debug, Clone)]
pub struct FileToolContext {
    root: PathBuf,
}

impl FileToolContext {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        }
    }

    fn display_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

/// Arguments accepted by [`ReadTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadArgs {
    /// Path of the file to read. Relative paths resolve against the harness
    /// root.
    pub file_path: PathBuf,
    /// 1-based line number to start reading from. Defaults to 1.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Number of lines to read. Defaults to 2000.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Reads a file with `cat -n` style line numbers.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadArgs,
    invoke = read,
    name = "read",
    description = "Read a file. Returns numbered lines (line_number\\tcontent). Use offset/limit to read a specific line range of large files; long lines are truncated.",
)]
pub struct ReadTool {
    context: FileToolContext,
}

impl ReadTool {
    pub fn new(context: FileToolContext) -> Self {
        Self { context }
    }

    async fn read(&self, args: ReadArgs) -> Result<String, LLMYError> {
        let path = self.context.resolve(&args.file_path);
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) => {
                return Ok(format!("Cannot read {}: {}", path.display(), error));
            }
        };
        if bytes.contains(&0) {
            return Ok(format!(
                "{} looks binary ({} bytes); refusing to render it as text.",
                path.display(),
                bytes.len()
            ));
        }
        let content = String::from_utf8_lossy(&bytes);
        let offset = args.offset.unwrap_or(1).max(1);
        let limit = args.limit.unwrap_or(DEFAULT_READ_LINES).max(1);

        let total_lines = content.lines().count();
        if total_lines == 0 {
            return Ok(format!("{} is empty.", path.display()));
        }
        if offset > total_lines {
            return Ok(format!(
                "{} has {} lines; offset {} is past the end.",
                path.display(),
                total_lines,
                offset
            ));
        }

        let mut out = vec![];
        for (index, line) in content.lines().enumerate().skip(offset - 1).take(limit) {
            let rendered: String = if line.chars().count() > MAX_LINE_CHARS {
                let cut: String = line.chars().take(MAX_LINE_CHARS).collect();
                format!("{cut}[line truncated]")
            } else {
                line.to_string()
            };
            out.push(format!("{:>6}\t{}", index + 1, rendered));
        }
        let end = (offset - 1 + out.len()).min(total_lines);
        if end < total_lines {
            out.push(format!(
                "[showing lines {}..{} of {}; continue with offset={}]",
                offset,
                end,
                total_lines,
                end + 1
            ));
        }
        Ok(out.join("\n"))
    }
}

/// Arguments accepted by [`GrepTool`].
#[derive(Deserialize, JsonSchema)]
pub struct GrepArgs {
    /// Regular expression (PCRE2) to search for.
    pub pattern: String,
    /// File or directory to search. Defaults to the harness root.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Only search files whose path matches this glob (e.g. "*.sol",
    /// "src/**/*.rs").
    #[serde(default)]
    pub glob: Option<String>,
    /// Case-insensitive matching.
    #[serde(default)]
    pub case_insensitive: Option<bool>,
    /// Maximum matching lines returned. Defaults to 200.
    #[serde(default)]
    pub max_matches: Option<usize>,
    /// Only list matching file paths instead of matching lines.
    #[serde(default)]
    pub files_only: Option<bool>,
}

/// Searches file contents with PCRE2 regexes, honoring .gitignore.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = GrepArgs,
    invoke = grep,
    name = "grep",
    description = "Search file contents with a PCRE2 regular expression. Respects .gitignore. Returns `path:line:content` matches (or matching paths with files_only). Scope with `path` and `glob`.",
)]
pub struct GrepTool {
    context: FileToolContext,
}

impl GrepTool {
    pub fn new(context: FileToolContext) -> Self {
        Self { context }
    }

    async fn grep(&self, args: GrepArgs) -> Result<String, LLMYError> {
        let target = self
            .context
            .resolve(args.path.as_deref().unwrap_or(Path::new(".")));
        if !target.exists() {
            return Ok(format!("Path {} does not exist.", target.display()));
        }

        let matcher = match RegexMatcherBuilder::new()
            .caseless(args.case_insensitive.unwrap_or(false))
            .build(&args.pattern)
        {
            Ok(matcher) => matcher,
            Err(error) => {
                return Ok(format!("Invalid pattern {:?}: {}", args.pattern, error));
            }
        };

        let glob_set = match args.glob.as_deref() {
            Some(glob) => {
                let mut builder = GlobSetBuilder::new();
                let parsed = match Glob::new(glob) {
                    Ok(parsed) => parsed,
                    Err(error) => return Ok(format!("Invalid glob {:?}: {}", glob, error)),
                };
                builder.add(parsed);
                // Also accept a bare-name glob against the file name, so
                // "*.sol" matches nested files without needing "**/*.sol".
                if !glob.contains('/')
                    && let Ok(named) = Glob::new(&format!("**/{glob}"))
                {
                    builder.add(named);
                }
                match builder.build() {
                    Ok(set) => Some(set),
                    Err(error) => return Ok(format!("Invalid glob {:?}: {}", glob, error)),
                }
            }
            None => None,
        };

        let max_matches = args.max_matches.unwrap_or(DEFAULT_GREP_MATCHES).max(1);
        let files_only = args.files_only.unwrap_or(false);

        let mut searcher = SearcherBuilder::new()
            .binary_detection(BinaryDetection::quit(0))
            .line_number(true)
            .build();

        let mut lines = vec![];
        let mut matched_files = vec![];
        let mut hit_limit = false;

        let walker = WalkBuilder::new(&target).build();
        for entry in walker {
            if hit_limit {
                break;
            }
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    tracing::debug!("grep walk error: {}", error);
                    continue;
                }
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            if let Some(set) = &glob_set {
                let candidate = path.strip_prefix(&target).unwrap_or(path);
                if !set.is_match(candidate) && !set.is_match(path) {
                    continue;
                }
            }

            let display = self.context.display_path(path);
            let mut file_matched = false;
            let sink = UTF8(|line_number, line| {
                file_matched = true;
                if files_only {
                    // One hit is enough to list the file.
                    return Ok(false);
                }
                let rendered: String = if line.chars().count() > MAX_LINE_CHARS {
                    line.chars().take(MAX_LINE_CHARS).collect()
                } else {
                    line.trim_end_matches('\n').to_string()
                };
                lines.push(format!("{display}:{line_number}:{rendered}"));
                Ok(lines.len() < max_matches)
            });
            if let Err(error) = searcher.search_path(&matcher, path, sink) {
                tracing::debug!("grep failed on {}: {}", path.display(), error);
            }
            if file_matched {
                matched_files.push(display);
            }
            if !files_only && lines.len() >= max_matches {
                hit_limit = true;
            }
            if files_only && matched_files.len() >= max_matches {
                hit_limit = true;
            }
        }

        if files_only {
            if matched_files.is_empty() {
                return Ok(format!("No files match {:?}.", args.pattern));
            }
            let mut out = format!(
                "Found {} matching file(s):\n{}",
                matched_files.len(),
                matched_files.join("\n")
            );
            if hit_limit {
                out.push_str(&format!("\n[reached the limit of {max_matches} files]"));
            }
            return Ok(out);
        }

        if lines.is_empty() {
            return Ok(format!("No matches for {:?}.", args.pattern));
        }
        let mut out = format!(
            "Found {} matching line(s):\n{}",
            lines.len(),
            lines.join("\n")
        );
        if hit_limit {
            out.push_str(&format!("\n[reached the limit of {max_matches} matches; narrow the pattern or scope to see more]"));
        }
        Ok(out)
    }
}

/// Arguments accepted by [`StatTool`].
#[derive(Deserialize, JsonSchema)]
pub struct StatArgs {
    /// File or directory to inspect.
    pub path: PathBuf,
}

/// Reports file metadata; for directories, lists the entries.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = StatArgs,
    invoke = stat,
    name = "stat",
    description = "Inspect a path: for files, size/kind/mtime and line count; for directories, the entry listing (name, kind, size).",
)]
pub struct StatTool {
    context: FileToolContext,
}

impl StatTool {
    pub fn new(context: FileToolContext) -> Self {
        Self { context }
    }

    async fn stat(&self, args: StatArgs) -> Result<String, LLMYError> {
        let path = self.context.resolve(&args.path);
        let meta = match tokio::fs::symlink_metadata(&path).await {
            Ok(meta) => meta,
            Err(error) => return Ok(format!("Cannot stat {}: {}", path.display(), error)),
        };

        let modified = meta
            .modified()
            .ok()
            .map(|time| chrono::DateTime::<chrono::Utc>::from(time).to_rfc3339())
            .unwrap_or_else(|| "unknown".to_string());

        if meta.is_dir() {
            let mut entries = vec![];
            let mut reader = tokio::fs::read_dir(&path).await?;
            while let Some(entry) = reader.next_entry().await? {
                let entry_meta = entry.metadata().await;
                let (kind, size) = match &entry_meta {
                    Ok(meta) if meta.is_dir() => ("dir", 0),
                    Ok(meta) => ("file", meta.len()),
                    Err(_) => ("?", 0),
                };
                entries.push(format!(
                    "{}\t{}\t{}",
                    entry.file_name().to_string_lossy(),
                    kind,
                    size
                ));
                if entries.len() >= MAX_DIR_ENTRIES {
                    entries.push(format!("[listing capped at {MAX_DIR_ENTRIES} entries]"));
                    break;
                }
            }
            entries.sort();
            return Ok(format!(
                "{} is a directory (modified {}), {} entries:\n{}",
                path.display(),
                modified,
                entries.len(),
                entries.join("\n")
            ));
        }

        let kind = if meta.file_type().is_symlink() {
            "symlink"
        } else {
            "file"
        };
        let mut out = format!(
            "{}: {} of {} bytes, modified {}",
            path.display(),
            kind,
            meta.len(),
            modified
        );
        if meta.is_file()
            && meta.len() < 4 * 1024 * 1024
            && let Ok(bytes) = tokio::fs::read(&path).await
            && !bytes.contains(&0)
        {
            let lines = String::from_utf8_lossy(&bytes).lines().count();
            out.push_str(&format!(", {lines} lines"));
        }
        Ok(out)
    }
}

/// Arguments accepted by [`SearchTool`].
#[derive(Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// Glob pattern to match file paths against (e.g. "**/*.sol",
    /// "src/*.rs", "Move.toml").
    pub pattern: String,
    /// Directory to search under. Defaults to the harness root.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Maximum number of paths returned. Defaults to 500.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Finds files by glob, most recently modified first.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = SearchArgs,
    invoke = search,
    name = "search",
    description = "Find files by glob pattern (e.g. \"**/*.sol\"). Respects .gitignore and returns matching paths sorted by modification time, newest first.",
)]
pub struct SearchTool {
    context: FileToolContext,
}

impl SearchTool {
    pub fn new(context: FileToolContext) -> Self {
        Self { context }
    }

    async fn search(&self, args: SearchArgs) -> Result<String, LLMYError> {
        let target = self
            .context
            .resolve(args.path.as_deref().unwrap_or(Path::new(".")));
        if !target.is_dir() {
            return Ok(format!("{} is not a directory.", target.display()));
        }

        let mut builder = GlobSetBuilder::new();
        let parsed = match Glob::new(&args.pattern) {
            Ok(parsed) => parsed,
            Err(error) => return Ok(format!("Invalid glob {:?}: {}", args.pattern, error)),
        };
        builder.add(parsed);
        if !args.pattern.contains('/')
            && let Ok(named) = Glob::new(&format!("**/{}", args.pattern))
        {
            builder.add(named);
        }
        let glob_set = match builder.build() {
            Ok(set) => set,
            Err(error) => return Ok(format!("Invalid glob {:?}: {}", args.pattern, error)),
        };

        let limit = args.limit.unwrap_or(DEFAULT_SEARCH_RESULTS).max(1);
        let mut matches = vec![];
        for entry in WalkBuilder::new(&target).build() {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    tracing::debug!("search walk error: {}", error);
                    continue;
                }
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            let candidate = path.strip_prefix(&target).unwrap_or(path);
            if !glob_set.is_match(candidate) {
                continue;
            }
            let mtime = entry.metadata().ok().and_then(|meta| meta.modified().ok());
            matches.push((mtime, self.context.display_path(path)));
        }

        if matches.is_empty() {
            return Ok(format!("No files match {:?}.", args.pattern));
        }
        matches.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let total = matches.len();
        let listed = matches
            .into_iter()
            .take(limit)
            .map(|(_, path)| path)
            .collect::<Vec<_>>();
        let mut out = format!("Found {} file(s):\n{}", total, listed.join("\n"));
        if total > limit {
            out.push_str(&format!("\n[showing the {limit} most recently modified]"));
        }
        Ok(out)
    }
}
