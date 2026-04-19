//! Dangerous absolute-path file-system tool implementations.
//! These tools intentionally bypass the sandbox root and can access any path
//! visible to the current process. Prefer the relative-path tools whenever the
//! caller can work within a sandboxed root.

use std::path::PathBuf;

use llmy_agent::LLMYError;
use llmy_agent_derive::tool;
use schemars::JsonSchema;
use serde::Deserialize;

use super::common::{
    DeleteFileArgs, EditFileArgs, FindFileArgs, ReadFileToolArgs, WriteFileArgs,
    delete_file_at_path, edit_file_at_path, find_file_blocking_at_path, list_directory_at_path,
    list_files_absolute, read_file_at_path, write_file_at_path,
};
use super::prompt::{
    ABSOLUTE_DELETE_FILE_TOOL_DESCRIPTION, ABSOLUTE_EDIT_FILE_TOOL_DESCRIPTION,
    ABSOLUTE_FIND_FILE_TOOL_DESCRIPTION, ABSOLUTE_LIST_DIRECTORY_TOOL_DESCRIPTION,
    ABSOLUTE_READ_FILE_TOOL_DESCRIPTION, ABSOLUTE_WRITE_FILE_TOOL_DESCRIPTION,
};

/// Arguments accepted by the dangerous direct-path list-directory tool.
#[derive(Deserialize, JsonSchema)]
pub struct AbsoluteListDirectoryToolArgs {
    /// Directory path to list. This is used as provided.
    pub directory_path: PathBuf,
}

/// Direct-path file-reading tool.
///
/// This bypasses the sandbox root and can read any file visible to the current
/// process, so prefer the relative-path variant whenever you can constrain the
/// agent to a workspace root.
#[derive(Debug, Clone, Default)]
#[tool(
    arguments = ReadFileToolArgs,
    invoke = read_file,
    name = "read_file",
    description = ABSOLUTE_READ_FILE_TOOL_DESCRIPTION,
)]
pub struct AbsoluteReadFileTool {}

impl AbsoluteReadFileTool {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn read_file(&self, args: ReadFileToolArgs) -> Result<String, LLMYError> {
        let target_path = args.file_path.clone();
        read_file_at_path(&target_path, &args.file_path, &args).await
    }
}

/// Direct-path directory-listing tool.
///
/// This bypasses the sandbox root and can inspect any directory visible to the
/// current process, so prefer the relative-path variant whenever you can
/// constrain the agent to a workspace root.
#[derive(Debug, Clone, Default)]
#[tool(
    arguments = AbsoluteListDirectoryToolArgs,
    invoke = list_directory,
    name = "list_dir",
    description = ABSOLUTE_LIST_DIRECTORY_TOOL_DESCRIPTION,
)]
pub struct AbsoluteListDirectoryTool {}

impl AbsoluteListDirectoryTool {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn list_directory(
        &self,
        args: AbsoluteListDirectoryToolArgs,
    ) -> Result<String, LLMYError> {
        let target_path = args.directory_path.clone();
        list_directory_at_path(&target_path, &args.directory_path, list_files_absolute).await
    }
}

/// Direct-path recursive file-search tool.
///
/// This bypasses the sandbox root and can search any directory visible to the
/// current process, so prefer the relative-path variant whenever you can
/// constrain the agent to a workspace root.
#[derive(Debug, Clone, Default)]
#[tool(
    arguments = FindFileArgs,
    invoke = find_file,
    name = "find_file",
    description = ABSOLUTE_FIND_FILE_TOOL_DESCRIPTION,
)]
pub struct AbsoluteFindFileTool {}

impl AbsoluteFindFileTool {
    pub fn new() -> Self {
        Self {}
    }

    fn find_file_blocking(directory: PathBuf, pattern: String) -> Result<String, LLMYError> {
        let target_path = directory.clone();
        find_file_blocking_at_path(&target_path, &directory, &pattern, list_files_absolute)
    }

    pub async fn find_file(&self, arguments: FindFileArgs) -> Result<String, LLMYError> {
        tokio::task::spawn_blocking(move || {
            Self::find_file_blocking(arguments.directory, arguments.file_name_pattern)
        })
        .await
        .expect("fail to join")
    }
}

/// Direct-path file-writing tool.
///
/// This bypasses the sandbox root and can overwrite any writable file visible
/// to the current process, so prefer the relative-path variant whenever you can
/// constrain the agent to a workspace root.
#[derive(Debug, Clone, Default)]
#[tool(
    arguments = WriteFileArgs,
    invoke = write_file,
    name = "write_file",
    description = ABSOLUTE_WRITE_FILE_TOOL_DESCRIPTION,
)]
pub struct AbsoluteWriteFileTool {}

impl AbsoluteWriteFileTool {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn write_file(&self, args: WriteFileArgs) -> Result<String, LLMYError> {
        let target_path = args.file_path.clone();
        write_file_at_path(&target_path, &args.file_path, &args).await
    }
}

/// Direct-path file-deletion tool.
///
/// This bypasses the sandbox root and can delete any writable file visible to
/// the current process, so prefer the relative-path variant whenever you can
/// constrain the agent to a workspace root.
#[derive(Debug, Clone, Default)]
#[tool(
    arguments = DeleteFileArgs,
    invoke = delete_file,
    name = "delete_file",
    description = ABSOLUTE_DELETE_FILE_TOOL_DESCRIPTION,
)]
pub struct AbsoluteDeleteFileTool {}

impl AbsoluteDeleteFileTool {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn delete_file(&self, args: DeleteFileArgs) -> Result<String, LLMYError> {
        let target_path = args.file_path.clone();
        delete_file_at_path(&target_path, &args.file_path).await
    }
}

/// Direct-path file-editing tool.
///
/// This bypasses the sandbox root and can modify any writable text file visible
/// to the current process, so prefer the relative-path variant whenever you can
/// constrain the agent to a workspace root.
#[derive(Debug, Clone, Default)]
#[tool(
    arguments = EditFileArgs,
    invoke = edit_file,
    name = "edit_file",
    description = ABSOLUTE_EDIT_FILE_TOOL_DESCRIPTION,
)]
pub struct AbsoluteEditFileTool {}

impl AbsoluteEditFileTool {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn edit_file(&self, args: EditFileArgs) -> Result<String, LLMYError> {
        let target_path = args.file_path.clone();
        edit_file_at_path(&target_path, &args.file_path, &args).await
    }
}
