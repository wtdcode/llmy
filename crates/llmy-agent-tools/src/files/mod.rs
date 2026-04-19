//! File-system tool implementations for `llmy-agent`.

mod absolute;
mod common;
mod prompt;
mod relative;

pub use absolute::{
    AbsoluteDeleteFileTool, AbsoluteEditFileTool, AbsoluteFindFileTool, AbsoluteListDirectoryTool,
    AbsoluteListDirectoryToolArgs, AbsoluteReadFileTool, AbsoluteWriteFileTool,
};
pub use common::{
    DeleteFileArgs, EditFileArgs, FindFileArgs, ListDirectoryToolArgs, ReadFileToolArgs,
    WriteFileArgs, sanitize_join_relative_path,
};
pub use relative::{
    DeleteFileTool, EditFileTool, FindFileTool, ListDirectoryTool, ReadFileTool, WriteFileTool,
};

#[cfg(test)]
mod tests {
    use super::*;
    use llmy_agent::Tool;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

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
        assert_eq!(<DeleteFileTool as Tool>::NAME, "delete_file");
        assert_eq!(<EditFileTool as Tool>::NAME, "edit_file");
        assert_eq!(<AbsoluteReadFileTool as Tool>::NAME, "read_file");
        assert_eq!(<AbsoluteListDirectoryTool as Tool>::NAME, "list_dir");
        assert_eq!(<AbsoluteFindFileTool as Tool>::NAME, "find_file");
        assert_eq!(<AbsoluteWriteFileTool as Tool>::NAME, "write_file");
        assert_eq!(<AbsoluteDeleteFileTool as Tool>::NAME, "delete_file");
        assert_eq!(<AbsoluteEditFileTool as Tool>::NAME, "edit_file");
    }

    #[tokio::test]
    async fn read_file_honors_requested_line_range() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("sample.txt");
        tokio::fs::write(&file_path, "one\ntwo\nthree\nfour\n")
            .await
            .unwrap();

        let tool = ReadFileTool::new(dir.path().to_path_buf());
        let result = tool
            .read_file(ReadFileToolArgs {
                file_path: PathBuf::from("sample.txt"),
                start_line: Some(2),
                line_count: Some(2),
            })
            .await
            .unwrap();

        assert_eq!(result, "two\nthree\n");
    }

    #[tokio::test]
    async fn edit_file_requires_unique_match_without_replace_all() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("sample.txt");
        tokio::fs::write(&file_path, "beta\nalpha\nbeta\n")
            .await
            .unwrap();

        let tool = EditFileTool::new(dir.path().to_path_buf());
        let result = tool
            .edit_file(EditFileArgs {
                file_path: PathBuf::from("sample.txt"),
                old_string: "beta".to_string(),
                new_string: "BETA".to_string(),
                replace_all: false,
            })
            .await
            .unwrap();

        assert!(result.contains("Found 2 matches"));
    }
}
