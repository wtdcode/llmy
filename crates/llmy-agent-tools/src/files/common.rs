use std::{
    fmt::Write,
    fs::Metadata,
    io,
    path::{Component, Path, PathBuf},
};

use grep::matcher::Matcher;
use grep::pcre2::{RegexMatcher, RegexMatcherBuilder};
use grep::searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkMatch};
use ignore::WalkBuilder;
use ignore::overrides::{Override, OverrideBuilder};
use llmy_agent::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::AsyncReadExt;

pub const MAX_READ_BYTES: usize = 8192;

/// Default cap on the number of matching lines a directory grep returns.
pub const DEFAULT_GREP_MAX_MATCHES: usize = 50;

/// Maximum number of bytes rendered for a single matching line before it is
/// truncated. Keeps a single minified line from flooding the output.
const MAX_GREP_LINE_BYTES: usize = 512;

/// Joins a relative path to `cwd` while rejecting absolute and parent-traversal paths.
pub fn sanitize_join_relative_path(cwd: &Path, rpath: &Path) -> Result<PathBuf, String> {
    if rpath.is_absolute() {
        return Err(format!("{:?} is an absolute path", rpath));
    }
    if rpath.components().any(|t| t == Component::ParentDir) {
        return Err(format!("{:?} contains '..'", rpath));
    }

    Ok(cwd.join(rpath))
}

pub fn render_binary_preview(bytes: &[u8]) -> String {
    let mut out = String::new();

    for (line_index, chunk) in bytes.chunks(16).enumerate() {
        let _ = write!(out, "{:08x}: ", line_index * 16);
        for byte_index in 0..16 {
            if let Some(byte) = chunk.get(byte_index) {
                let _ = write!(out, "{:02x} ", byte);
            } else {
                out.push_str("   ");
            }
        }
        out.push(' ');
        for byte in chunk {
            let ch = if byte.is_ascii_graphic() || *byte == b' ' {
                char::from(*byte)
            } else {
                '.'
            };
            out.push(ch);
        }
        if (line_index + 1) * 16 < bytes.len() {
            out.push('\n');
        }
    }

    out
}

fn entry_type(meta: &Metadata) -> &'static str {
    if meta.is_dir() {
        "directory"
    } else if meta.is_file() {
        "file"
    } else if meta.file_type().is_symlink() {
        "symlink"
    } else {
        ""
    }
}

pub fn list_files_with_display_paths<F>(
    fpaths: Vec<PathBuf>,
    mut display_path: F,
) -> Result<Vec<String>, LLMYError>
where
    F: FnMut(&Path) -> Result<PathBuf, LLMYError>,
{
    let mut lns = vec![];
    for fp in fpaths {
        let meta = fp.metadata()?;
        let canonical_path = fp.canonicalize()?;
        let display_path = display_path(&canonical_path)?;
        let ln = format!("{:?}\t{}\t{}", display_path, entry_type(&meta), meta.len());
        lns.push(ln);
    }
    Ok(lns)
}

pub fn list_files_relative(cwd: &Path, fpaths: Vec<PathBuf>) -> Result<Vec<String>, LLMYError> {
    let canonical_root = cwd.canonicalize()?;
    list_files_with_display_paths(fpaths, move |canonical_path| {
        let relative_path = canonical_path
            .strip_prefix(&canonical_root)
            .unwrap_or(canonical_path);
        Ok(relative_path.to_path_buf())
    })
}

pub fn list_files_absolute(fpaths: Vec<PathBuf>) -> Result<Vec<String>, LLMYError> {
    list_files_with_display_paths(fpaths, |canonical_path| Ok(canonical_path.to_path_buf()))
}

/// Arguments accepted by the relative and absolute read-file tools.
#[derive(Deserialize, JsonSchema, Default)]
pub struct ReadFileToolArgs {
    /// Path to the file to read.
    pub file_path: PathBuf,
    /// Optional 1-based line number to start reading from. Defaults to 1.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Optional number of lines to read from `start_line`.
    #[serde(default)]
    pub line_count: Option<usize>,
}

/// Arguments accepted by the relative list-directory tool.
#[derive(Deserialize, JsonSchema)]
pub struct ListDirectoryToolArgs {
    /// Directory path to list.
    pub relative_path: PathBuf,
}

/// Arguments accepted by file-finding tools.
#[derive(Deserialize, JsonSchema)]
pub struct FindFileArgs {
    /// Directory to search recursively.
    pub directory: PathBuf,
    /// Glob pattern matched against each file name, for example `*.rs`.
    pub file_name_pattern: String,
}

/// Arguments accepted by the directory-grep tools.
#[derive(Deserialize, JsonSchema)]
pub struct GrepDirectoryArgs {
    /// Directory to search recursively for file contents.
    pub directory: PathBuf,
    /// PCRE2 regular expression. Only lines matching it are reported.
    pub pattern: String,
    /// Optional PCRE2 regular expression. Lines that also match this pattern are
    /// dropped from the results, like piping through `grep -v`.
    #[serde(default)]
    pub invert_pattern: Option<String>,
    /// Optional glob patterns (globset / gitignore syntax, e.g. `*.rs`,
    /// `src/**/*.ts`) restricting which files are searched. When non-empty, only
    /// files matching at least one pattern are searched.
    #[serde(default)]
    pub include: Vec<String>,
    /// Optional glob patterns (globset / gitignore syntax) excluding files from
    /// the search. Applied on top of `include`.
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Search case-insensitively. Defaults to false.
    #[serde(default)]
    pub case_insensitive: bool,
    /// Maximum number of matching lines to return. Defaults to 50.
    #[serde(default)]
    pub max_matches: Option<usize>,
}

/// Arguments accepted by file-writing tools.
#[derive(Deserialize, JsonSchema)]
pub struct WriteFileArgs {
    /// Destination file path.
    pub file_path: PathBuf,
    /// Full file contents to write.
    pub content: String,
}

/// Arguments accepted by file-deletion tools.
#[derive(Deserialize, JsonSchema)]
pub struct DeleteFileArgs {
    /// File path to delete.
    pub file_path: PathBuf,
}

/// Arguments accepted by file-edit tools.
#[derive(Deserialize, JsonSchema)]
pub struct EditFileArgs {
    /// File path to modify.
    pub file_path: PathBuf,
    /// The text to replace.
    pub old_string: String,
    /// The replacement text.
    pub new_string: String,
    /// Replace all occurrences instead of exactly one.
    #[serde(default)]
    pub replace_all: bool,
}

fn validate_read_range(args: &ReadFileToolArgs) -> Result<(usize, Option<usize>), String> {
    let start_line = args.start_line.unwrap_or(1);
    if start_line == 0 {
        return Err("start_line must be greater than or equal to 1".to_string());
    }
    if matches!(args.line_count, Some(0)) {
        return Err("line_count must be greater than or equal to 1".to_string());
    }

    Ok((start_line, args.line_count))
}

fn truncate_text(mut text: String) -> String {
    if text.len() <= MAX_READ_BYTES {
        return text;
    }

    let mut cutoff = MAX_READ_BYTES;
    while cutoff > 0 && !text.is_char_boundary(cutoff) {
        cutoff -= 1;
    }
    text.truncate(cutoff);
    text
}

fn total_line_count(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.split_inclusive('\n').count()
    }
}

fn slice_text_lines(
    text: &str,
    display_path: &Path,
    start_line: usize,
    line_count: Option<usize>,
) -> String {
    let total_lines = total_line_count(text);
    if start_line > 1 && start_line > total_lines {
        return format!(
            "Start line {} is beyond the end of file {:?}, which has {} lines",
            start_line, display_path, total_lines
        );
    }

    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    if lines.is_empty() {
        return String::new();
    }

    let start_index = start_line.saturating_sub(1).min(lines.len());
    let end_index = line_count
        .map(|count| start_index.saturating_add(count))
        .unwrap_or(lines.len())
        .min(lines.len());

    truncate_text(lines[start_index..end_index].concat())
}

pub async fn read_file_at_path(
    target_path: &Path,
    display_path: &Path,
    args: &ReadFileToolArgs,
) -> Result<String, LLMYError> {
    let (start_line, line_count) = match validate_read_range(args) {
        Ok(range) => range,
        Err(error) => return Ok(error),
    };

    match tokio::fs::metadata(target_path).await {
        Ok(meta) => {
            if meta.is_dir() {
                return Ok(format!("Path {:?} is a directory", display_path));
            }
        }
        Err(error) => {
            return Ok(format!(
                "Fail to get metadata of {:?} due to {}",
                display_path, error
            ));
        }
    };

    let mut fp = match tokio::fs::File::open(target_path).await {
        Ok(fp) => fp,
        Err(error) => {
            return Ok(format!("Fail to open {:?} due to {}", display_path, error));
        }
    };

    let mut buf = vec![];
    fp.read_to_end(&mut buf).await?;

    match String::from_utf8(buf) {
        Ok(content) => Ok(slice_text_lines(
            &content,
            display_path,
            start_line,
            line_count,
        )),
        Err(error) => {
            let bytes = error.into_bytes();
            Ok(render_binary_preview(
                &bytes[..bytes.len().min(MAX_READ_BYTES)],
            ))
        }
    }
}

pub async fn list_directory_at_path<F>(
    target_path: &Path,
    display_path: &Path,
    list_files: F,
) -> Result<String, LLMYError>
where
    F: FnOnce(Vec<PathBuf>) -> Result<Vec<String>, LLMYError>,
{
    if !target_path.is_dir() {
        return Ok(format!("{:?} is not a directory", display_path));
    }

    let mut items = vec![];
    let mut entries = tokio::fs::read_dir(target_path).await?;
    while let Some(ent) = entries.next_entry().await? {
        items.push(ent.path());
    }
    let lns = list_files(items)?;
    Ok(format!(
        "The contents of folder {:?} is:\nname\ttype\tsize\n{}",
        display_path,
        lns.join("\n")
    ))
}

pub fn find_file_blocking_at_path<F>(
    target_path: &Path,
    display_path: &Path,
    pattern: &str,
    list_files: F,
) -> Result<String, LLMYError>
where
    F: FnOnce(Vec<PathBuf>) -> Result<Vec<String>, LLMYError>,
{
    let re = match glob::Pattern::new(pattern) {
        Ok(re) => re,
        Err(error) => return Ok(format!("Fail to compile the glob pattern due to {}", error)),
    };

    if !target_path.is_dir() {
        return Ok(format!("{:?} is not a directory", display_path));
    }

    let mut items = vec![];
    for ent in walkdir::WalkDir::new(target_path) {
        let ent = ent.map_err(std::io::Error::other)?;
        let Some(fname) = ent.file_name().to_str() else {
            continue;
        };
        if re.matches(fname) {
            items.push(ent.path().to_path_buf());
        }
    }
    let lns = list_files(items)?;
    Ok(format!(
        "The files found under directory {:?} with given pattern {} are:\n{}",
        display_path,
        pattern,
        lns.join("\n")
    ))
}

/// A single matching line collected during a directory grep.
struct GrepMatch {
    display: String,
    line_number: u64,
    content: String,
}

/// Truncates a single line to [`MAX_GREP_LINE_BYTES`] on a char boundary.
fn truncate_line(line: &str) -> String {
    if line.len() <= MAX_GREP_LINE_BYTES {
        return line.to_string();
    }

    let mut cutoff = MAX_GREP_LINE_BYTES;
    while cutoff > 0 && !line.is_char_boundary(cutoff) {
        cutoff -= 1;
    }
    format!("{}… [line truncated]", &line[..cutoff])
}

/// Builds a PCRE2 matcher mirroring the engine ripgrep uses for `--pcre2`.
fn build_pcre2_matcher(
    pattern: &str,
    case_insensitive: bool,
) -> Result<RegexMatcher, grep::pcre2::Error> {
    RegexMatcherBuilder::new()
        .caseless(case_insensitive)
        .build(pattern)
}

/// Builds the include/exclude override set, applying excludes as negated globs
/// just like ripgrep's `--glob` handling (which is itself backed by globset).
fn build_overrides(
    root: &Path,
    include: &[String],
    exclude: &[String],
) -> Result<Override, ignore::Error> {
    let mut builder = OverrideBuilder::new(root);
    for glob in include {
        builder.add(glob)?;
    }
    for glob in exclude {
        builder.add(&format!("!{}", glob))?;
    }
    builder.build()
}

/// Collects matching lines from a single file, honoring the invert pattern and
/// the global match limit.
struct GrepSink<'a> {
    matches: &'a mut Vec<GrepMatch>,
    display: &'a str,
    invert: Option<&'a RegexMatcher>,
    limit: usize,
}

impl Sink for GrepSink<'_> {
    type Error = io::Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, io::Error> {
        if let Some(invert) = self.invert {
            if invert.is_match(mat.bytes()).map_err(io::Error::other)? {
                // Matches the invert pattern, so skip it but keep searching.
                return Ok(true);
            }
        }

        let raw = String::from_utf8_lossy(mat.bytes());
        let trimmed = raw.trim_end_matches(['\n', '\r']);
        self.matches.push(GrepMatch {
            display: self.display.to_string(),
            line_number: mat.line_number().unwrap_or(0),
            content: truncate_line(trimmed),
        });

        // Stop searching this file once the global limit is reached.
        Ok(self.matches.len() < self.limit)
    }
}

/// Recursively greps `target_path` for lines matching `args.pattern`.
///
/// `display_match_path` maps each matched file's on-disk path to the path shown
/// in the output, letting the relative tool strip the sandbox root while the
/// absolute tool renders the path verbatim.
pub fn grep_directory_blocking_at_path<F>(
    target_path: &Path,
    display_path: &Path,
    args: &GrepDirectoryArgs,
    display_match_path: F,
) -> Result<String, LLMYError>
where
    F: Fn(&Path) -> PathBuf,
{
    if !target_path.is_dir() {
        return Ok(format!("{:?} is not a directory", display_path));
    }

    let limit = args.max_matches.unwrap_or(DEFAULT_GREP_MAX_MATCHES);
    if limit == 0 {
        return Ok("max_matches must be greater than or equal to 1".to_string());
    }

    let matcher = match build_pcre2_matcher(&args.pattern, args.case_insensitive) {
        Ok(matcher) => matcher,
        Err(error) => {
            return Ok(format!(
                "Fail to compile the match pattern due to {}",
                error
            ));
        }
    };
    let invert_matcher = match args.invert_pattern.as_deref() {
        Some(pattern) => match build_pcre2_matcher(pattern, args.case_insensitive) {
            Ok(matcher) => Some(matcher),
            Err(error) => {
                return Ok(format!(
                    "Fail to compile the invert pattern due to {}",
                    error
                ));
            }
        },
        None => None,
    };

    let overrides = match build_overrides(target_path, &args.include, &args.exclude) {
        Ok(overrides) => overrides,
        Err(error) => {
            return Ok(format!(
                "Fail to compile the include/exclude globs due to {}",
                error
            ));
        }
    };

    let walker = WalkBuilder::new(target_path).overrides(overrides).build();
    let mut searcher = SearcherBuilder::new()
        .line_number(true)
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .build();

    let mut matches: Vec<GrepMatch> = Vec::new();
    let mut truncated = false;

    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        let path = entry.path();
        let display = display_match_path(path).to_string_lossy().into_owned();
        let sink = GrepSink {
            matches: &mut matches,
            display: &display,
            invert: invert_matcher.as_ref(),
            limit,
        };

        if let Err(error) = searcher.search_path(&matcher, path, sink) {
            tracing::debug!(path = ?path, %error, "grep: failed to search file");
            continue;
        }

        if matches.len() >= limit {
            truncated = true;
            break;
        }
    }

    Ok(format_grep_results(
        display_path,
        args,
        &matches,
        limit,
        truncated,
    ))
}

/// Renders the collected matches into the textual tool result.
fn format_grep_results(
    display_path: &Path,
    args: &GrepDirectoryArgs,
    matches: &[GrepMatch],
    limit: usize,
    truncated: bool,
) -> String {
    if matches.is_empty() {
        return format!(
            "No matching lines for pattern {:?} under {:?}.",
            args.pattern, display_path
        );
    }

    let mut out = format!(
        "Found {} matching line(s) for pattern {:?} under {:?}:\n",
        matches.len(),
        args.pattern,
        display_path
    );
    for (index, grep_match) in matches.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        let _ = write!(
            out,
            "{}:{}:{}",
            grep_match.display, grep_match.line_number, grep_match.content
        );
    }
    if truncated {
        let _ = write!(
            out,
            "\n... reached the max_matches limit of {}; more matches may exist.",
            limit
        );
    }
    out
}

pub async fn write_file_at_path(
    target_path: &Path,
    display_path: &Path,
    args: &WriteFileArgs,
) -> Result<String, LLMYError> {
    if let Ok(meta) = tokio::fs::metadata(target_path).await {
        if meta.is_dir() {
            return Ok(format!(
                "Path {:?} is a directory, cannot write to it",
                display_path
            ));
        }
    }

    if let Some(parent) = target_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    tokio::fs::write(target_path, args.content.as_bytes()).await?;

    Ok(format!("Successfully wrote to file {:?}", display_path))
}

pub async fn delete_file_at_path(
    target_path: &Path,
    display_path: &Path,
) -> Result<String, LLMYError> {
    match tokio::fs::metadata(target_path).await {
        Ok(meta) => {
            if meta.is_dir() {
                return Ok(format!(
                    "Path {:?} is a directory, cannot delete it with delete_file",
                    display_path
                ));
            }
        }
        Err(error) => {
            return Ok(format!(
                "Fail to get metadata of {:?} due to {}",
                display_path, error
            ));
        }
    }

    tokio::fs::remove_file(target_path).await?;

    Ok(format!("Successfully deleted file {:?}", display_path))
}

const LEFT_SINGLE_CURLY_QUOTE: char = '\u{2018}';
const RIGHT_SINGLE_CURLY_QUOTE: char = '\u{2019}';
const LEFT_DOUBLE_CURLY_QUOTE: char = '\u{201c}';
const RIGHT_DOUBLE_CURLY_QUOTE: char = '\u{201d}';

fn normalize_quotes(text: &str) -> String {
    text.replace(LEFT_SINGLE_CURLY_QUOTE, "'")
        .replace(RIGHT_SINGLE_CURLY_QUOTE, "'")
        .replace(LEFT_DOUBLE_CURLY_QUOTE, "\"")
        .replace(RIGHT_DOUBLE_CURLY_QUOTE, "\"")
}

pub fn find_actual_string(file_content: &str, search_string: &str) -> Option<String> {
    if file_content.contains(search_string) {
        return Some(search_string.to_string());
    }
    if search_string.is_empty() {
        return Some(String::new());
    }

    let normalized_search = normalize_quotes(search_string);
    let normalized_file = normalize_quotes(file_content);
    let search_chars: Vec<char> = normalized_search.chars().collect();
    let file_chars: Vec<char> = normalized_file.chars().collect();
    let start_char = file_chars
        .windows(search_chars.len())
        .position(|window| window == search_chars.as_slice())?;

    Some(
        file_content
            .chars()
            .skip(start_char)
            .take(search_string.chars().count())
            .collect(),
    )
}

fn is_opening_quote_context(chars: &[char], index: usize) -> bool {
    if index == 0 {
        return true;
    }

    matches!(
        chars[index - 1],
        ' ' | '\t' | '\n' | '\r' | '(' | '[' | '{' | '\u{2014}' | '\u{2013}'
    )
}

fn apply_curly_double_quotes(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    for (index, ch) in chars.iter().enumerate() {
        if *ch == '"' {
            result.push(if is_opening_quote_context(&chars, index) {
                LEFT_DOUBLE_CURLY_QUOTE
            } else {
                RIGHT_DOUBLE_CURLY_QUOTE
            });
        } else {
            result.push(*ch);
        }
    }
    result
}

fn apply_curly_single_quotes(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    for (index, ch) in chars.iter().enumerate() {
        if *ch == '\'' {
            let prev = index.checked_sub(1).and_then(|i| chars.get(i));
            let next = chars.get(index + 1);
            let is_contraction = prev.is_some_and(|ch| ch.is_alphabetic())
                && next.is_some_and(|ch| ch.is_alphabetic());
            if is_contraction {
                result.push(RIGHT_SINGLE_CURLY_QUOTE);
            } else {
                result.push(if is_opening_quote_context(&chars, index) {
                    LEFT_SINGLE_CURLY_QUOTE
                } else {
                    RIGHT_SINGLE_CURLY_QUOTE
                });
            }
        } else {
            result.push(*ch);
        }
    }
    result
}

pub fn preserve_quote_style(old_string: &str, actual_old_string: &str, new_string: &str) -> String {
    if old_string == actual_old_string {
        return new_string.to_string();
    }

    let has_double_quotes = actual_old_string.contains(LEFT_DOUBLE_CURLY_QUOTE)
        || actual_old_string.contains(RIGHT_DOUBLE_CURLY_QUOTE);
    let has_single_quotes = actual_old_string.contains(LEFT_SINGLE_CURLY_QUOTE)
        || actual_old_string.contains(RIGHT_SINGLE_CURLY_QUOTE);

    let mut result = new_string.to_string();
    if has_double_quotes {
        result = apply_curly_double_quotes(&result);
    }
    if has_single_quotes {
        result = apply_curly_single_quotes(&result);
    }
    result
}

pub fn apply_text_edit(
    original_content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> String {
    let replace = |content: &str, search: &str, replacement: &str| {
        if replace_all {
            content.replace(search, replacement)
        } else {
            content.replacen(search, replacement, 1)
        }
    };

    if !new_string.is_empty() {
        return replace(original_content, old_string, new_string);
    }

    let strip_trailing_newline =
        !old_string.ends_with('\n') && original_content.contains(&format!("{}\n", old_string));

    if strip_trailing_newline {
        replace(original_content, &format!("{}\n", old_string), new_string)
    } else {
        replace(original_content, old_string, new_string)
    }
}

pub async fn edit_file_at_path(
    target_path: &Path,
    display_path: &Path,
    args: &EditFileArgs,
) -> Result<String, LLMYError> {
    if args.old_string == args.new_string {
        return Ok(
            "No changes to make: old_string and new_string are exactly the same.".to_string(),
        );
    }

    match tokio::fs::metadata(target_path).await {
        Ok(meta) => {
            if meta.is_dir() {
                return Ok(format!(
                    "Path {:?} is a directory, cannot edit it with edit_file",
                    display_path
                ));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if args.old_string.is_empty() {
                if let Some(parent) = target_path.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                tokio::fs::write(target_path, args.new_string.as_bytes()).await?;
                return Ok(format!("Successfully created file {:?}", display_path));
            }

            return Ok(format!("File {:?} does not exist", display_path));
        }
        Err(error) => {
            return Ok(format!(
                "Fail to get metadata of {:?} due to {}",
                display_path, error
            ));
        }
    }

    let content = match String::from_utf8(tokio::fs::read(target_path).await?) {
        Ok(content) => content,
        Err(_) => {
            return Ok(format!(
                "Path {:?} is not a UTF-8 text file and cannot be edited with edit_file",
                display_path
            ));
        }
    };

    if args.old_string.is_empty() {
        if !content.is_empty() {
            return Ok("Cannot create new file - file already exists.".to_string());
        }

        tokio::fs::write(target_path, args.new_string.as_bytes()).await?;
        return Ok(format!("Successfully edited file {:?}", display_path));
    }

    let Some(actual_old_string) = find_actual_string(&content, &args.old_string) else {
        return Ok(format!(
            "String to replace not found in file {:?}.\nString: {}",
            display_path, args.old_string
        ));
    };

    let matches = content.matches(&actual_old_string).count();
    if matches > 1 && !args.replace_all {
        return Ok(format!(
            "Found {} matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, provide more context to uniquely identify the instance.\nString: {}",
            matches, args.old_string
        ));
    }

    let replacement = preserve_quote_style(&args.old_string, &actual_old_string, &args.new_string);
    let updated_content =
        apply_text_edit(&content, &actual_old_string, &replacement, args.replace_all);

    if updated_content == content {
        return Ok("Original and edited file match exactly. Failed to apply edit.".to_string());
    }

    tokio::fs::write(target_path, updated_content.as_bytes()).await?;

    let replaced = if args.replace_all { matches } else { 1 };
    let suffix = if replaced == 1 { "" } else { "s" };
    Ok(format!(
        "Successfully edited file {:?}; replaced {} occurrence{}",
        display_path, replaced, suffix
    ))
}
