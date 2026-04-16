//! File-system tool implementations for `llmy-agent`.

use std::{
    fmt::Write,
    path::{Component, Path, PathBuf},
};

use llmy_agent::LLMYError;
use llmy_agent_derive::tool;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::AsyncReadExt;

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

fn render_binary_preview(bytes: &[u8]) -> String {
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

/// Arguments accepted by [`ReadFileTool`].
#[derive(Deserialize, JsonSchema, Default)]
pub struct ReadFileToolArgs {
    /// Path to the file to read, relative to the tool root.
    pub file_path: PathBuf,
}

/// Reads a file from a sandboxed working directory.
#[derive(Debug, Clone)]
#[tool(
    arguments = ReadFileToolArgs,
    invoke = read_file,
    name = "read_file",
    description = "Read file contents of the path `file_path`. The result will be hexdump if the file is a binary file.",
)]
pub struct ReadFileTool {
    /// Root directory that all file operations are constrained to.
    pub cwd: PathBuf,
}

impl ReadFileTool {
    /// Creates a file-reading tool rooted at `cwd`.
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }

    /// Reads up to 8 KiB from `file_path` and falls back to a hex preview for binary data.
    pub async fn read_file(&self, args: ReadFileToolArgs) -> Result<String, LLMYError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &args.file_path) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        match tokio::fs::metadata(&target_path).await {
            Ok(meta) => {
                if meta.is_dir() {
                    return Ok(format!("Path {:?} is a directory", &target_path));
                }
            }
            Err(e) => {
                return Ok(format!(
                    "Fail to get metadata of {:?} due to {}",
                    &target_path, e
                ));
            }
        };
        let mut fp = match tokio::fs::File::open(&target_path).await {
            Ok(fp) => fp,
            Err(e) => return Ok(format!("Fail to open {:?} due to {}", &target_path, e)),
        };

        let mut buf = vec![];
        fp.read_to_end(&mut buf).await?;

        let buf = if buf.len() >= 8192 {
            // too long and cutoff
            buf[0..8192].to_vec()
        } else {
            buf
        };

        match String::from_utf8(buf) {
            Ok(s) => Ok(s),
            Err(e) => Ok(render_binary_preview(&e.into_bytes())),
        }
    }
}

/// Formats a list of filesystem entries as tab-separated rows relative to `cwd`.
pub fn list_files(cwd: &Path, fpaths: Vec<PathBuf>) -> Result<Vec<String>, LLMYError> {
    let mut lns = vec![];
    let cwd = cwd.canonicalize()?;
    for fp in fpaths {
        let meta = fp.metadata()?;
        let canonical_path = fp.canonicalize()?;
        let relative_path = canonical_path.strip_prefix(&cwd).unwrap_or(&canonical_path);
        let ln = format!(
            "{:?}\t{}\t{}",
            relative_path,
            if meta.is_dir() {
                "directory"
            } else if meta.is_file() {
                "file"
            } else if meta.is_symlink() {
                "symlink"
            } else {
                ""
            },
            meta.len()
        );
        lns.push(ln);
    }
    Ok(lns)
}

/// Arguments accepted by [`ListDirectoryTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ListDirectoryToolArgs {
    /// Directory path to list, relative to the tool root.
    pub relative_path: PathBuf,
}

/// Lists the immediate contents of a directory under a sandboxed root.
#[derive(Debug, Clone)]
#[tool(
    arguments = ListDirectoryToolArgs,
    invoke = list_directory,
    name = "list_dir",
    description = "List a given directory entries. '.' is allowed to list entries of the root directory but '..' is not allowed to avoid path traversal. Absolute path is not allowed and you shall always use relative path to the root directory.",
)]
pub struct ListDirectoryTool {
    /// Root directory that all file operations are constrained to.
    pub cwd: PathBuf,
}

impl ListDirectoryTool {
    /// Creates a directory-listing tool rooted at `path`.
    pub fn new_root(path: PathBuf) -> Self {
        Self { cwd: path }
    }

    /// Lists direct child entries of `relative_path` as `name\ttype\tsize` rows.
    pub async fn list_directory(&self, args: ListDirectoryToolArgs) -> Result<String, LLMYError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &args.relative_path) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        if !target_path.is_dir() {
            return Ok(format!("{:?} is not a directory", &target_path));
        }

        let mut items = vec![];
        let mut entries = tokio::fs::read_dir(&target_path).await?;
        while let Some(ent) = entries.next_entry().await? {
            items.push(ent.path());
        }
        let lns = list_files(&self.cwd, items)?;
        Ok(format!(
            "The contents of folder {:?} is:\nname\ttype\tsize\n{}",
            &args.relative_path,
            lns.join("\n")
        ))
    }
}

/// Arguments accepted by [`FindFileTool`].
#[derive(Deserialize, JsonSchema)]
pub struct FindFileArgs {
    /// Directory to search recursively, relative to the tool root.
    pub directory: PathBuf,
    /// Glob pattern matched against each file name, for example `*.rs`.
    pub file_name_pattern: String,
}

/// Finds files by glob pattern under a sandboxed root directory.
#[derive(Debug, Clone)]
#[tool(
    arguments = FindFileArgs,
    invoke = find_file,
    name = "find_file",
    description = "Find files with names having the given glob pattern under the given directory. For example, use '*.c' to find all C source files. For directory, note '.' is allowed to list entries of the root directory but '..' is not allowed to avoid path traversal. Absolute path is not allowed and you shall always use relative path to the root directory.",
)]
pub struct FindFileTool {
    /// Root directory that all file operations are constrained to.
    pub cwd: PathBuf,
}

impl FindFileTool {
    /// Creates a recursive file-search tool rooted at `path`.
    pub fn new(path: PathBuf) -> Self {
        Self { cwd: path }
    }

    fn find_file_blocking(
        cwd: PathBuf,
        directory: PathBuf,
        pattern: String,
    ) -> Result<String, LLMYError> {
        let re = match glob::Pattern::new(&pattern) {
            Ok(re) => re,
            Err(e) => return Ok(format!("Fail to compile the glob pattern due to {}", e)),
        };

        let target_path = match sanitize_join_relative_path(&cwd, &directory) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };
        if !target_path.is_dir() {
            return Ok(format!("{:?} is not a directory", &target_path));
        }

        let mut items = vec![];
        for ent in walkdir::WalkDir::new(&target_path) {
            let ent = ent.map_err(std::io::Error::other)?;
            let Some(fname) = ent.file_name().to_str() else {
                continue;
            };
            if re.matches(&fname) {
                items.push(ent.path().to_path_buf());
            }
        }
        let lns = list_files(&cwd, items)?;
        Ok(format!(
            "The files found under directory {:?} with given pattern {} are:\n{}",
            &directory,
            &pattern,
            lns.join("\n")
        ))
    }

    /// Recursively searches `directory` for file names matching `file_name_pattern`.
    pub async fn find_file(&self, arguments: FindFileArgs) -> Result<String, LLMYError> {
        let cwd = self.cwd.clone();
        tokio::task::spawn_blocking(move || {
            Self::find_file_blocking(cwd, arguments.directory, arguments.file_name_pattern)
        })
        .await
        .expect("fail to join")
    }
}

/// Arguments accepted by [`WriteFileTool`].
#[derive(Deserialize, JsonSchema)]
pub struct WriteFileArgs {
    /// Destination file path, relative to the tool root.
    pub file_path: PathBuf,
    /// Full file contents to write.
    pub content: String,
}

/// Writes files beneath a sandboxed working directory.
#[derive(Debug, Clone)]
#[tool(
    arguments = WriteFileArgs,
    invoke = write_file,
    name = "write_file",
    description = "Write content to the file at the given path. The file will be created if it doesn't exist, or overwritten if it does. Parent directories will be created automatically. The path should be always relative path and '.' is allowed while '..' is not allowed.",
)]
pub struct WriteFileTool {
    /// Root directory that all file operations are constrained to.
    pub cwd: PathBuf,
}

impl WriteFileTool {
    /// Creates a file-writing tool rooted at `cwd`.
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }

    /// Writes `content` to `file_path`, creating missing parent directories when needed.
    pub async fn write_file(&self, args: WriteFileArgs) -> Result<String, LLMYError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &args.file_path) {
            Ok(p) => p,
            Err(e) => return Ok(e),
        };

        if let Ok(meta) = tokio::fs::metadata(&target_path).await {
            if meta.is_dir() {
                return Ok(format!(
                    "Path {:?} is a directory, cannot write to it",
                    &args.file_path
                ));
            }
        }

        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&target_path, args.content).await?;

        Ok(format!("Successfully wrote to file {:?}", &args.file_path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmy_agent::Tool;

    #[test]
    fn rejects_absolute_and_parent_paths() {
        let cwd = Path::new("workspace");

        assert!(sanitize_join_relative_path(cwd, Path::new("/tmp")).is_err());
        assert!(sanitize_join_relative_path(cwd, Path::new("../tmp")).is_err());
        assert_eq!(
            sanitize_join_relative_path(cwd, Path::new("nested/file.txt")).unwrap(),
            PathBuf::from("workspace/nested/file.txt")
        );
    }

    #[test]
    fn derive_preserves_tool_metadata() {
        assert_eq!(<ReadFileTool as Tool>::NAME, "read_file");
        assert_eq!(<ListDirectoryTool as Tool>::NAME, "list_dir");
        assert_eq!(<FindFileTool as Tool>::NAME, "find_file");
        assert_eq!(<WriteFileTool as Tool>::NAME, "write_file");
    }
}
