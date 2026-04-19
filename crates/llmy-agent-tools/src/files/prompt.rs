use const_format::concatcp;

pub const SANDBOXED_PATH_USAGE_DESCRIPTION: &str = "All paths must stay within the tool root. Use relative paths only; '.' is allowed, while '..' and absolute paths are rejected.";

pub const DIRECT_PATH_USAGE_DESCRIPTION: &str =
    "Paths are used exactly as provided without sandboxing.";

pub const READ_FILE_USAGE_DESCRIPTION: &str = "Reads a file from the local filesystem.\n\nUsage:\n- Use this tool for files, not directories.\n- By default it reads from the beginning of the file.\n- You can optionally provide `start_line` and `line_count` to read only part of a file, which is preferable when you already know the relevant section.\n- Binary files return a hexdump preview instead of decoded text.";

pub const LIST_DIRECTORY_USAGE_DESCRIPTION: &str = "Lists the immediate entries of a directory.\n\nUsage:\n- This tool is not recursive; it only returns direct children of the directory.\n- Use this tool for directories, not files.\n- Use `find_file` when you need a recursive file search by name pattern.";

pub const FIND_FILE_USAGE_DESCRIPTION: &str = "Finds files by glob pattern.\n\nUsage:\n- Supports glob patterns such as `*.rs` or `src/**/*.ts`.\n- Searches recursively under the given directory.\n- Use this tool when you need to find files by name pattern rather than by file contents.";

pub const WRITE_FILE_USAGE_DESCRIPTION: &str = "Writes a file to the local filesystem.\n\nUsage:\n- This tool overwrites the existing file if there is one at the target path.\n- Prefer `edit_file` for localized modifications; use `write_file` for new files or complete rewrites.\n- If you are replacing existing content, read the file first so you do not accidentally discard unrelated changes.\n- Parent directories are created automatically.";

pub const DELETE_FILE_USAGE_DESCRIPTION: &str = "Deletes a file from the local filesystem.\n\nUsage:\n- This tool only deletes files, not directories.\n- Deletion is permanent from the tool's perspective, so verify the path carefully before using it.\n- If you are not certain about the target, inspect the directory or read the file first.";

pub const EDIT_FILE_USAGE_DESCRIPTION: &str = "Performs exact string replacements in files.\n\nUsage:\n- Read the file first so `old_string` matches the exact current contents before editing.\n- When editing text copied from `read_file` output, preserve the exact indentation and whitespace from the file. Do not normalize tabs, spaces, or line breaks in the matched text.\n- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.\n- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful when you want to rename a variable, function, or repeated literal.";

pub const RELATIVE_READ_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Read a file from the sandboxed local filesystem. ",
    SANDBOXED_PATH_USAGE_DESCRIPTION,
    " ",
    READ_FILE_USAGE_DESCRIPTION,
);

pub const ABSOLUTE_READ_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Read a file from the local filesystem using the provided path without sandboxing. ",
    DIRECT_PATH_USAGE_DESCRIPTION,
    " ",
    READ_FILE_USAGE_DESCRIPTION,
);

pub const RELATIVE_LIST_DIRECTORY_TOOL_DESCRIPTION: &str = concatcp!(
    "List a directory from the sandboxed local filesystem. ",
    SANDBOXED_PATH_USAGE_DESCRIPTION,
    " ",
    LIST_DIRECTORY_USAGE_DESCRIPTION,
);

pub const ABSOLUTE_LIST_DIRECTORY_TOOL_DESCRIPTION: &str = concatcp!(
    "List a directory from the local filesystem using the provided path without sandboxing. ",
    DIRECT_PATH_USAGE_DESCRIPTION,
    " ",
    LIST_DIRECTORY_USAGE_DESCRIPTION,
);

pub const RELATIVE_FIND_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Find files from the sandboxed local filesystem. ",
    SANDBOXED_PATH_USAGE_DESCRIPTION,
    " ",
    FIND_FILE_USAGE_DESCRIPTION,
);

pub const ABSOLUTE_FIND_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Find files from the local filesystem using the provided path without sandboxing. ",
    DIRECT_PATH_USAGE_DESCRIPTION,
    " ",
    FIND_FILE_USAGE_DESCRIPTION,
);

pub const RELATIVE_WRITE_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Write a file to the sandboxed local filesystem. ",
    SANDBOXED_PATH_USAGE_DESCRIPTION,
    " ",
    WRITE_FILE_USAGE_DESCRIPTION,
);

pub const ABSOLUTE_WRITE_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Write a file to the local filesystem using the provided path without sandboxing. ",
    DIRECT_PATH_USAGE_DESCRIPTION,
    " ",
    WRITE_FILE_USAGE_DESCRIPTION,
);

pub const RELATIVE_DELETE_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Delete a file from the sandboxed local filesystem. ",
    SANDBOXED_PATH_USAGE_DESCRIPTION,
    " ",
    DELETE_FILE_USAGE_DESCRIPTION,
);

pub const ABSOLUTE_DELETE_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Delete a file from the local filesystem using the provided path without sandboxing. ",
    DIRECT_PATH_USAGE_DESCRIPTION,
    " ",
    DELETE_FILE_USAGE_DESCRIPTION,
);

pub const RELATIVE_EDIT_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Edit a file in the sandboxed local filesystem by replacing `old_string` with `new_string`. ",
    SANDBOXED_PATH_USAGE_DESCRIPTION,
    " ",
    EDIT_FILE_USAGE_DESCRIPTION,
);

pub const ABSOLUTE_EDIT_FILE_TOOL_DESCRIPTION: &str = concatcp!(
    "Edit a file in the local filesystem using the provided path without sandboxing by replacing `old_string` with `new_string`. ",
    DIRECT_PATH_USAGE_DESCRIPTION,
    " ",
    EDIT_FILE_USAGE_DESCRIPTION,
);
