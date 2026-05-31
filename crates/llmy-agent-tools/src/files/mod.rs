//! File-system tool implementations for `llmy-agent`.

mod absolute;
mod common;
mod prompt;
mod relative;

pub use absolute::{
    AbsoluteDeleteFileTool, AbsoluteEditFileTool, AbsoluteFindFileTool, AbsoluteGrepDirectoryTool,
    AbsoluteListDirectoryTool, AbsoluteListDirectoryToolArgs, AbsoluteReadFileTool,
    AbsoluteWriteFileTool,
};
pub use common::{
    DeleteFileArgs, EditFileArgs, FindFileArgs, GrepDirectoryArgs, ListDirectoryToolArgs,
    ReadFileToolArgs, WriteFileArgs, sanitize_join_relative_path,
};
pub use relative::{
    DeleteFileTool, EditFileTool, FindFileTool, GrepDirectoryTool, ListDirectoryTool, ReadFileTool,
    WriteFileTool,
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
        assert_eq!(<GrepDirectoryTool as Tool>::NAME, "grep");
        assert_eq!(<WriteFileTool as Tool>::NAME, "write_file");
        assert_eq!(<DeleteFileTool as Tool>::NAME, "delete_file");
        assert_eq!(<EditFileTool as Tool>::NAME, "edit_file");
        assert_eq!(<AbsoluteReadFileTool as Tool>::NAME, "read_file");
        assert_eq!(<AbsoluteListDirectoryTool as Tool>::NAME, "list_dir");
        assert_eq!(<AbsoluteFindFileTool as Tool>::NAME, "find_file");
        assert_eq!(<AbsoluteGrepDirectoryTool as Tool>::NAME, "grep");
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

    fn grep_args(directory: &str, pattern: &str) -> GrepDirectoryArgs {
        GrepDirectoryArgs {
            directory: PathBuf::from(directory),
            pattern: pattern.to_string(),
            invert_pattern: None,
            include: vec![],
            exclude: vec![],
            case_insensitive: false,
            max_matches: None,
        }
    }

    async fn seed_grep_workspace(dir: &Path) {
        tokio::fs::write(dir.join("alpha.rs"), "fn alpha() {}\nlet TODO = 1;\n")
            .await
            .unwrap();
        tokio::fs::write(dir.join("beta.txt"), "TODO: write beta\nplain line\n")
            .await
            .unwrap();
        tokio::fs::create_dir_all(dir.join("nested")).await.unwrap();
        tokio::fs::write(
            dir.join("nested/gamma.rs"),
            "// TODO nested\nfn gamma() {}\n",
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn grep_matches_pattern_recursively() {
        let dir = tempdir().unwrap();
        seed_grep_workspace(dir.path()).await;

        let tool = GrepDirectoryTool::new(dir.path().to_path_buf());
        let result = tool.grep_directory(grep_args(".", "TODO")).await.unwrap();

        assert!(result.contains("alpha.rs:2:let TODO = 1;"), "{result}");
        assert!(result.contains("beta.txt:1:TODO: write beta"), "{result}");
        assert!(
            result.contains("nested/gamma.rs:1:// TODO nested"),
            "{result}"
        );
    }

    #[tokio::test]
    async fn grep_respects_include_and_invert_pattern() {
        let dir = tempdir().unwrap();
        seed_grep_workspace(dir.path()).await;

        let tool = GrepDirectoryTool::new(dir.path().to_path_buf());
        let mut args = grep_args(".", "TODO");
        args.include = vec!["*.rs".to_string()];
        args.invert_pattern = Some("nested".to_string());
        let result = tool.grep_directory(args).await.unwrap();

        assert!(result.contains("alpha.rs:2:let TODO = 1;"), "{result}");
        // Excluded by the `*.rs` include filter.
        assert!(!result.contains("beta.txt"), "{result}");
        // Dropped by the invert pattern even though the file matches the include.
        assert!(!result.contains("gamma.rs"), "{result}");
    }

    #[tokio::test]
    async fn grep_honors_max_matches_limit() {
        let dir = tempdir().unwrap();
        seed_grep_workspace(dir.path()).await;

        let tool = GrepDirectoryTool::new(dir.path().to_path_buf());
        let mut args = grep_args(".", "TODO");
        args.max_matches = Some(1);
        let result = tool.grep_directory(args).await.unwrap();

        assert!(result.contains("Found 1 matching line(s)"), "{result}");
        assert!(
            result.contains("reached the max_matches limit of 1"),
            "{result}"
        );
    }

    #[tokio::test]
    async fn grep_rejects_parent_traversal() {
        let dir = tempdir().unwrap();
        seed_grep_workspace(dir.path()).await;

        let tool = GrepDirectoryTool::new(dir.path().to_path_buf());
        let result = tool.grep_directory(grep_args("../", "TODO")).await.unwrap();

        assert!(result.contains("contains '..'"), "{result}");
    }

    #[tokio::test]
    async fn grep_supports_pcre2_and_case_insensitivity() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("repeat.txt"), "value FOOfoo here\n")
            .await
            .unwrap();

        let tool = GrepDirectoryTool::new(dir.path().to_path_buf());
        // Backreferences are a PCRE2-only construct unsupported by the default
        // Rust regex engine, so a successful match proves the pcre2 matcher and
        // case-insensitivity are both in effect.
        let mut args = grep_args(".", r"(foo)\1");
        args.case_insensitive = true;
        let result = tool.grep_directory(args).await.unwrap();

        assert!(
            result.contains("repeat.txt:1:value FOOfoo here"),
            "{result}"
        );
    }
}
