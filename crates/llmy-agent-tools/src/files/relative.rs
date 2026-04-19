use std::path::PathBuf;

use llmy_agent::LLMYError;
use llmy_agent_derive::tool;

use super::common::{
    DeleteFileArgs, EditFileArgs, FindFileArgs, ListDirectoryToolArgs, ReadFileToolArgs,
    WriteFileArgs, delete_file_at_path, edit_file_at_path, find_file_blocking_at_path,
    list_directory_at_path, list_files_relative, read_file_at_path, sanitize_join_relative_path,
    write_file_at_path,
};
use super::prompt::{
    RELATIVE_DELETE_FILE_TOOL_DESCRIPTION, RELATIVE_EDIT_FILE_TOOL_DESCRIPTION,
    RELATIVE_FIND_FILE_TOOL_DESCRIPTION, RELATIVE_LIST_DIRECTORY_TOOL_DESCRIPTION,
    RELATIVE_READ_FILE_TOOL_DESCRIPTION, RELATIVE_WRITE_FILE_TOOL_DESCRIPTION,
};

/// Reads a file from a sandboxed working directory.
#[derive(Debug, Clone)]
#[tool(
    arguments = ReadFileToolArgs,
    invoke = read_file,
    name = "read_file",
    description = RELATIVE_READ_FILE_TOOL_DESCRIPTION,
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

    /// Reads a whole file or a requested line range from `file_path`.
    pub async fn read_file(&self, args: ReadFileToolArgs) -> Result<String, LLMYError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &args.file_path) {
            Ok(path) => path,
            Err(error) => return Ok(error),
        };

        read_file_at_path(&target_path, &args.file_path, &args).await
    }
}

/// Lists the immediate contents of a directory under a sandboxed root.
#[derive(Debug, Clone)]
#[tool(
    arguments = ListDirectoryToolArgs,
    invoke = list_directory,
    name = "list_dir",
    description = RELATIVE_LIST_DIRECTORY_TOOL_DESCRIPTION,
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
            Ok(path) => path,
            Err(error) => return Ok(error),
        };

        list_directory_at_path(&target_path, &args.relative_path, |items| {
            list_files_relative(&self.cwd, items)
        })
        .await
    }
}

/// Finds files by glob pattern under a sandboxed root directory.
#[derive(Debug, Clone)]
#[tool(
    arguments = FindFileArgs,
    invoke = find_file,
    name = "find_file",
    description = RELATIVE_FIND_FILE_TOOL_DESCRIPTION,
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
        let target_path = match sanitize_join_relative_path(&cwd, &directory) {
            Ok(path) => path,
            Err(error) => return Ok(error),
        };

        find_file_blocking_at_path(&target_path, &directory, &pattern, |items| {
            list_files_relative(&cwd, items)
        })
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

/// Writes files beneath a sandboxed working directory.
#[derive(Debug, Clone)]
#[tool(
    arguments = WriteFileArgs,
    invoke = write_file,
    name = "write_file",
    description = RELATIVE_WRITE_FILE_TOOL_DESCRIPTION,
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
            Ok(path) => path,
            Err(error) => return Ok(error),
        };

        write_file_at_path(&target_path, &args.file_path, &args).await
    }
}

/// Deletes files beneath a sandboxed working directory.
#[derive(Debug, Clone)]
#[tool(
    arguments = DeleteFileArgs,
    invoke = delete_file,
    name = "delete_file",
    description = RELATIVE_DELETE_FILE_TOOL_DESCRIPTION,
)]
pub struct DeleteFileTool {
    /// Root directory that all file operations are constrained to.
    pub cwd: PathBuf,
}

impl DeleteFileTool {
    /// Creates a file-deletion tool rooted at `cwd`.
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }

    /// Deletes the file at `file_path`.
    pub async fn delete_file(&self, args: DeleteFileArgs) -> Result<String, LLMYError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &args.file_path) {
            Ok(path) => path,
            Err(error) => return Ok(error),
        };

        delete_file_at_path(&target_path, &args.file_path).await
    }
}

/// Edits files beneath a sandboxed working directory by replacing exact text.
#[derive(Debug, Clone)]
#[tool(
    arguments = EditFileArgs,
    invoke = edit_file,
    name = "edit_file",
    description = RELATIVE_EDIT_FILE_TOOL_DESCRIPTION,
)]
pub struct EditFileTool {
    /// Root directory that all file operations are constrained to.
    pub cwd: PathBuf,
}

impl EditFileTool {
    /// Creates a file-editing tool rooted at `cwd`.
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd }
    }

    /// Replaces exact text within `file_path`.
    pub async fn edit_file(&self, args: EditFileArgs) -> Result<String, LLMYError> {
        let target_path = match sanitize_join_relative_path(&self.cwd, &args.file_path) {
            Ok(path) => path,
            Err(error) => return Ok(error),
        };

        edit_file_at_path(&target_path, &args.file_path, &args).await
    }
}
