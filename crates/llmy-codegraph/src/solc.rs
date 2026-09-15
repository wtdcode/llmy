//! Solidity extraction over the compiler's own AST (the `--ast` output of
//! `forge build`, one `SourceUnit` per file). Every declaration carries the
//! compiler's node id and every reference its `referencedDeclaration`, so
//! calls, inheritance and state accesses resolve exactly instead of by name.
//! Call options (`f{value: x}(...)`), low-level calls (`to.call(...)`),
//! inline assembly (`call(...)`, `sstore(...)`) and writes through storage
//! pointers are all visible here and invisible to the tree-sitter extractor.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use color_eyre::eyre::eyre;
use foundry_compilers_artifacts::ast::yul::{
    YulBlock, YulCase, YulExpression, YulFunctionCall, YulIdentifier, YulStatement,
};
use foundry_compilers_artifacts::ast::{
    AssignmentOperator, Block, BlockOrStatement, ContractDefinition, ContractDefinitionPart,
    ContractKind, Expression, ExpressionOrVariableDeclarationStatement, FunctionCall,
    FunctionCallKind, FunctionDefinition, FunctionKind, IdentifierOrIdentifierPath, InlineAssembly,
    ModifierDefinition, ModifierInvocation, SourceLocation, SourceUnit, SourceUnitPart, Statement,
    StorageLocation, UnaryOperator, UserDefinedTypeNameOrIdentifierPath, VariableDeclaration,
};
use llmy_types::error::LLMYError;
use serde::Deserialize;

use crate::extract::{
    FileExtraction, RawCallSite, RawCallable, RawModule, RawParent, RawState, RawStateRef,
};
use crate::model::{CallableKind, Language, LineSpan, ModuleKind, StateKind};

/// Names of the Yul builtins that leave the contract: value or code moves,
/// or another account's code runs.
const YUL_EXTERNAL_CALLS: [&str; 7] = [
    "call",
    "callcode",
    "delegatecall",
    "staticcall",
    "create",
    "create2",
    "selfdestruct",
];

/// One compiled source: its text and its AST.
#[derive(Debug, Clone)]
pub struct SolcUnit {
    pub source: String,
    pub ast: SourceUnit,
}

/// The compiler output of a project: one AST per source file under the
/// project root, keyed by the path relative to that root.
#[derive(Debug, Clone)]
pub struct SolcBuildInfo {
    pub root: PathBuf,
    pub units: BTreeMap<PathBuf, SolcUnit>,
    /// Sorted `path:length` entries plus the compiler version, so a cached
    /// graph built from these ASTs is told apart from a tree-sitter one.
    pub fingerprint: String,
}

/// How `forge build` is run to produce the ASTs.
#[derive(Debug, Clone)]
pub struct ForgeAstOptions {
    pub forge_bin: PathBuf,
    /// `--skip` globs (forge's `test` and `script` aliases included).
    pub skip: Vec<String>,
    /// `FOUNDRY_PROFILE` for the build; the project default when unset.
    pub profile: Option<String>,
}

/// The parts of a forge build-info file the extraction needs.
#[derive(Debug, Deserialize)]
struct BuildInfoFile {
    input: BuildInfoInput,
    output: BuildInfoOutput,
    #[serde(rename = "solcLongVersion", default)]
    solc_long_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BuildInfoInput {
    sources: BTreeMap<String, BuildInfoSource>,
}

#[derive(Debug, Deserialize)]
struct BuildInfoSource {
    content: String,
}

#[derive(Debug, Deserialize)]
struct BuildInfoOutput {
    sources: BTreeMap<String, BuildInfoAst>,
}

#[derive(Debug, Deserialize)]
struct BuildInfoAst {
    ast: Option<SourceUnit>,
}

impl SolcBuildInfo {
    /// Reads every build-info JSON under `dir` (forge writes one per compiler
    /// job) and keeps the sources that live under `root`. A source listed in
    /// several jobs is taken from the first.
    pub async fn load(root: PathBuf, dir: &Path) -> Result<Self, LLMYError> {
        let root = root
            .canonicalize()
            .map_err(|e| eyre!("cannot canonicalize {}: {}", root.display(), e))?;
        let mut units: BTreeMap<PathBuf, SolcUnit> = BTreeMap::new();
        let mut versions: BTreeSet<String> = BTreeSet::new();
        let mut entries = tokio::fs::read_dir(dir)
            .await
            .map_err(|e| eyre!("cannot read build-info dir {}: {}", dir.display(), e))?;
        let mut files: Vec<PathBuf> = vec![];
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| eyre!("cannot list {}: {}", dir.display(), e))?
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                files.push(path);
            }
        }
        files.sort();
        for path in files {
            let bytes = tokio::fs::read(&path)
                .await
                .map_err(|e| eyre!("cannot read {}: {}", path.display(), e))?;
            let file: BuildInfoFile = serde_json::from_slice(&bytes)
                .map_err(|e| eyre!("cannot parse build-info {}: {}", path.display(), e))?;
            if let Some(version) = file.solc_long_version {
                versions.insert(version);
            }
            for (name, output) in file.output.sources {
                let Some(ast) = output.ast else {
                    tracing::warn!("build-info {} has no AST for {name}", path.display());
                    continue;
                };
                let Some(relative) = Self::relative_to(&root, &name) else {
                    continue;
                };
                if units.contains_key(&relative) {
                    continue;
                }
                let source = match file.input.sources.get(&name) {
                    Some(source) => source.content.clone(),
                    None => tokio::fs::read_to_string(root.join(&relative))
                        .await
                        .map_err(|e| eyre!("cannot read source {}: {}", relative.display(), e))?,
                };
                units.insert(relative, SolcUnit { source, ast });
            }
        }
        if units.is_empty() {
            return Err(eyre!(
                "no compiled source under {} found in {}",
                root.display(),
                dir.display()
            )
            .into());
        }
        let mut entries: Vec<String> = units
            .iter()
            .map(|(path, unit)| format!("{}:{}", path.display(), unit.source.len()))
            .collect();
        entries.sort();
        entries.extend(versions);
        Ok(Self {
            root,
            units,
            fingerprint: entries.join("\n"),
        })
    }

    /// Runs `forge build --ast --build-info` for the project at `root` with
    /// every output redirected under `dir`, then loads the result. The
    /// project's own configuration (remappings, profile, solc version)
    /// applies exactly as it does for a plain `forge build`.
    pub async fn from_forge(
        root: PathBuf,
        dir: &Path,
        options: &ForgeAstOptions,
    ) -> Result<Self, LLMYError> {
        let build_info = dir.join("build-info");
        tokio::fs::create_dir_all(&build_info)
            .await
            .map_err(|e| eyre!("cannot create {}: {}", build_info.display(), e))?;
        let mut command = tokio::process::Command::new(&options.forge_bin);
        command
            .current_dir(&root)
            .arg("build")
            .arg("--force")
            .arg("--ast")
            .arg("--build-info")
            .arg("--build-info-path")
            .arg(&build_info)
            .arg("-o")
            .arg(dir.join("out"))
            .arg("--cache-path")
            .arg(dir.join("cache"));
        for skip in &options.skip {
            command.arg("--skip").arg(skip);
        }
        if let Some(profile) = &options.profile {
            command.env("FOUNDRY_PROFILE", profile);
        }
        let output = command.output().await.map_err(|e| {
            eyre!(
                "cannot run {} in {}: {}",
                options.forge_bin.display(),
                root.display(),
                e
            )
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: String = stderr
                .chars()
                .rev()
                .take(4000)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Err(eyre!(
                "forge build --ast failed ({}) in {}:\n{}",
                output.status,
                root.display(),
                tail
            )
            .into());
        }
        Self::load(root, &build_info).await
    }

    /// The path of a compiled source relative to `root`: forge names sources
    /// by project-relative path, solc by absolute path; anything outside the
    /// root (a remapped dependency) is not part of the project.
    fn relative_to(root: &Path, name: &str) -> Option<PathBuf> {
        let path = Path::new(name);
        if path.is_absolute() {
            let canonical = path.canonicalize().ok()?;
            return canonical.strip_prefix(root).ok().map(Path::to_path_buf);
        }
        let joined = root.join(path);
        let canonical = joined.canonicalize().unwrap_or(joined);
        canonical.strip_prefix(root).ok().map(Path::to_path_buf)
    }
}

/// Byte offset to line number for one source text.
struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    fn new(source: &str) -> Self {
        let mut starts = vec![0];
        for (offset, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                starts.push(offset + 1);
            }
        }
        Self { starts }
    }

    /// 1-based line holding the byte offset.
    fn line_of(&self, offset: usize) -> usize {
        match self.starts.binary_search(&offset) {
            Ok(index) => index + 1,
            Err(index) => index,
        }
    }

    fn span_of(&self, src: &SourceLocation) -> LineSpan {
        let start = src.start.unwrap_or(0);
        let end = start + src.length.unwrap_or(0);
        LineSpan {
            start_line: self.line_of(start),
            end_line: self.line_of(end.saturating_sub(1).max(start)),
        }
    }
}

/// Extracts one compiled source unit.
pub struct SolcExtractor;

impl SolcExtractor {
    pub fn extract(relative: &Path, unit: &SolcUnit) -> Result<FileExtraction, LLMYError> {
        let file = UnitReader {
            source: &unit.source,
            lines: LineIndex::new(&unit.source),
        };
        let mut modules = vec![];
        let mut free_functions = vec![];
        for part in &unit.ast.nodes {
            match part {
                SourceUnitPart::ContractDefinition(contract) => {
                    modules.push(file.module(contract));
                }
                SourceUnitPart::FunctionDefinition(function) => {
                    free_functions.push(file.function(function));
                }
                _ => {}
            }
        }
        if !free_functions.is_empty() {
            let stem = relative
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "<file>".to_string());
            modules.push(RawModule {
                name: stem,
                kind: ModuleKind::Library,
                span: LineSpan {
                    start_line: 1,
                    end_line: file.lines.starts.len(),
                },
                parents: vec![],
                callables: free_functions,
                states: vec![],
                node_id: Some(unit.ast.id as i64),
            });
        }
        Ok(FileExtraction {
            file: relative.to_path_buf(),
            language: Language::Solidity,
            modules,
            parse_errors: 0,
        })
    }
}

/// The source text and line index of the unit being extracted.
struct UnitReader<'a> {
    source: &'a str,
    lines: LineIndex,
}

impl UnitReader<'_> {
    fn text_of(&self, src: &SourceLocation) -> String {
        let start = src.start.unwrap_or(0).min(self.source.len());
        let end = (start + src.length.unwrap_or(0)).min(self.source.len());
        let mut lo = start;
        while lo < end && !self.source.is_char_boundary(lo) {
            lo += 1;
        }
        let mut hi = end;
        while hi > lo && !self.source.is_char_boundary(hi) {
            hi -= 1;
        }
        self.source[lo..hi].to_string()
    }

    fn module(&self, contract: &ContractDefinition) -> RawModule {
        let kind = match contract.kind {
            ContractKind::Interface => ModuleKind::Interface,
            ContractKind::Library => ModuleKind::Library,
            ContractKind::Contract => ModuleKind::Contract,
        };
        let parents = contract
            .base_contracts
            .iter()
            .map(|spec| match &spec.base_name {
                UserDefinedTypeNameOrIdentifierPath::UserDefinedTypeName(udt) => RawParent {
                    name: udt.name.clone().unwrap_or_else(|| self.text_of(&udt.src)),
                    declaration: Some(udt.referenced_declaration as i64),
                },
                UserDefinedTypeNameOrIdentifierPath::IdentifierPath(path) => RawParent {
                    name: path.name.clone(),
                    declaration: Some(path.referenced_declaration as i64),
                },
            })
            .collect();
        let mut states = vec![];
        let mut callables = vec![];
        for part in &contract.nodes {
            match part {
                ContractDefinitionPart::VariableDeclaration(variable)
                    if variable.state_variable =>
                {
                    states.push(self.state(variable));
                }
                ContractDefinitionPart::FunctionDefinition(function) => {
                    callables.push(self.function(function));
                }
                ContractDefinitionPart::ModifierDefinition(modifier) => {
                    callables.push(self.modifier(modifier));
                }
                _ => {}
            }
        }
        RawModule {
            name: contract.name.clone(),
            kind,
            span: self.lines.span_of(&contract.src),
            parents,
            callables,
            states,
            node_id: Some(contract.id as i64),
        }
    }

    fn state(&self, variable: &VariableDeclaration) -> RawState {
        let type_text = variable
            .type_descriptions
            .type_string
            .clone()
            .or_else(|| {
                variable
                    .type_name
                    .as_ref()
                    .map(|_| self.text_of(&variable.src))
            })
            .unwrap_or_default();
        RawState {
            name: variable.name.clone(),
            kind: StateKind::StateVariable,
            type_text,
            span: self.lines.span_of(&variable.src),
            node_id: Some(variable.id as i64),
        }
    }

    /// The declaration text before the body: `function f(uint a) public
    /// returns (bool)`; the whole declaration for a body-less one.
    fn signature(&self, src: &SourceLocation, body: Option<&Block>) -> String {
        let start = src.start.unwrap_or(0);
        let end = match body.and_then(|b| b.src.start) {
            Some(body_start) if body_start > start => body_start,
            _ => start + src.length.unwrap_or(0),
        };
        let head = self.text_of(&SourceLocation {
            start: Some(start),
            length: Some(end - start),
            index: src.index,
        });
        head.trim().trim_end_matches(';').trim().to_string()
    }

    fn function(&self, function: &FunctionDefinition) -> RawCallable {
        let (name, kind) = match function.kind() {
            FunctionKind::Constructor => ("constructor".to_string(), CallableKind::Constructor),
            FunctionKind::Receive => ("receive".to_string(), CallableKind::Receive),
            FunctionKind::Fallback => ("fallback".to_string(), CallableKind::Fallback),
            FunctionKind::Function | FunctionKind::FreeFunction => {
                (function.name.clone(), CallableKind::Function)
            }
        };
        let mut walker = BodyWalker::new(self);
        for invocation in &function.modifiers {
            walker.modifier_invocation(invocation);
        }
        if let Some(body) = &function.body {
            walker.block(body);
        }
        RawCallable {
            name,
            kind,
            signature: self.signature(&function.src, function.body.as_ref()),
            span: self.lines.span_of(&function.src),
            calls: walker.calls,
            state_refs: walker.refs,
            node_id: Some(function.id as i64),
        }
    }

    fn modifier(&self, modifier: &ModifierDefinition) -> RawCallable {
        let mut walker = BodyWalker::new(self);
        if let Some(body) = &modifier.body {
            walker.block(body);
        }
        RawCallable {
            name: modifier.name.clone(),
            kind: CallableKind::Modifier,
            signature: self.signature(&modifier.src, modifier.body.as_ref()),
            span: self.lines.span_of(&modifier.src),
            calls: walker.calls,
            state_refs: walker.refs,
            node_id: Some(modifier.id as i64),
        }
    }
}

/// Walks one callable body in source order, collecting the call sites and the
/// state references with their lines. Storage pointers (`Position storage p
/// = positions[id]`) alias the state they point at, so a write through `p`
/// is recorded against `positions`.
struct BodyWalker<'a> {
    reader: &'a UnitReader<'a>,
    /// Local declaration id -> the declarations it aliases.
    aliases: BTreeMap<i64, BTreeSet<i64>>,
    calls: Vec<RawCallSite>,
    refs: Vec<RawStateRef>,
}

impl<'a> BodyWalker<'a> {
    fn new(reader: &'a UnitReader<'a>) -> Self {
        Self {
            reader,
            aliases: BTreeMap::new(),
            calls: vec![],
            refs: vec![],
        }
    }

    fn line(&self, src: &SourceLocation) -> usize {
        self.reader.lines.line_of(src.start.unwrap_or(0))
    }

    fn modifier_invocation(&mut self, invocation: &ModifierInvocation) {
        let (name, declaration) = match &invocation.modifier_name {
            IdentifierOrIdentifierPath::Identifier(identifier) => (
                identifier.name.clone(),
                identifier.referenced_declaration.map(|id| id as i64),
            ),
            IdentifierOrIdentifierPath::IdentifierPath(path) => {
                (path.name.clone(), Some(path.referenced_declaration as i64))
            }
        };
        self.calls.push(RawCallSite {
            text: name.clone(),
            name,
            qualifier: None,
            line: self.line(&invocation.src),
            declaration,
        });
        for argument in &invocation.arguments {
            self.expression(argument, false, true);
        }
    }

    fn block(&mut self, block: &Block) {
        for statement in &block.statements {
            self.statement(statement);
        }
    }

    fn block_or_statement(&mut self, body: &BlockOrStatement) {
        match body {
            BlockOrStatement::Block(block) => self.block(block),
            BlockOrStatement::Statement(statement) => self.statement(statement),
        }
    }

    fn statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Block(block) => self.block(block),
            Statement::UncheckedBlock(block) => {
                for statement in &block.statements {
                    self.statement(statement);
                }
            }
            Statement::ExpressionStatement(expression) => {
                self.expression(&expression.expression, false, true);
            }
            Statement::VariableDeclarationStatement(declaration) => {
                if let Some(initial) = &declaration.initial_value {
                    let aliased = self.storage_base(initial);
                    if !aliased.is_empty() {
                        for local in declaration.declarations.iter().flatten() {
                            if matches!(local.storage_location, StorageLocation::Storage) {
                                self.aliases
                                    .entry(local.id as i64)
                                    .or_default()
                                    .extend(aliased.iter().copied());
                            }
                        }
                    }
                    self.expression(initial, false, true);
                }
            }
            Statement::IfStatement(statement) => {
                self.expression(&statement.condition, false, true);
                self.block_or_statement(&statement.true_body);
                if let Some(false_body) = &statement.false_body {
                    self.block_or_statement(false_body);
                }
            }
            Statement::WhileStatement(statement) => {
                self.expression(&statement.condition, false, true);
                self.block_or_statement(&statement.body);
            }
            Statement::DoWhileStatement(statement) => {
                self.block(&statement.body);
                self.expression(&statement.condition, false, true);
            }
            Statement::ForStatement(statement) => {
                match &statement.initialization_expression {
                    Some(ExpressionOrVariableDeclarationStatement::ExpressionStatement(
                        expression,
                    )) => self.expression(&expression.expression, false, true),
                    Some(
                        ExpressionOrVariableDeclarationStatement::VariableDeclarationStatement(
                            declaration,
                        ),
                    ) => self.statement(&Statement::VariableDeclarationStatement(
                        declaration.clone(),
                    )),
                    None => {}
                }
                if let Some(condition) = &statement.condition {
                    self.expression(condition, false, true);
                }
                if let Some(step) = &statement.loop_expression {
                    self.expression(&step.expression, false, true);
                }
                self.block_or_statement(&statement.body);
            }
            Statement::Return(statement) => {
                if let Some(expression) = &statement.expression {
                    self.expression(expression, false, true);
                }
            }
            Statement::EmitStatement(statement) => {
                for argument in &statement.event_call.arguments {
                    self.expression(argument, false, true);
                }
            }
            Statement::RevertStatement(statement) => {
                for argument in &statement.error_call.arguments {
                    self.expression(argument, false, true);
                }
            }
            Statement::TryStatement(statement) => {
                self.call(&statement.external_call);
                for clause in &statement.clauses {
                    self.block(&clause.block);
                }
            }
            Statement::InlineAssembly(assembly) => self.assembly(assembly),
            Statement::PlaceholderStatement(_) | Statement::Break(_) | Statement::Continue(_) => {}
        }
    }

    fn expression(&mut self, expression: &Expression, write: bool, read: bool) {
        match expression {
            Expression::Identifier(identifier) => {
                if let Some(declaration) = identifier.referenced_declaration {
                    self.touch(
                        declaration as i64,
                        &identifier.name,
                        write,
                        read,
                        self.line(&identifier.src),
                    );
                }
            }
            Expression::MemberAccess(access) => {
                self.expression(&access.expression, write, read);
                if let Some(declaration) = access.referenced_declaration {
                    self.touch(
                        declaration as i64,
                        &access.member_name,
                        write,
                        read,
                        self.line(&access.src),
                    );
                }
            }
            Expression::IndexAccess(access) => {
                self.expression(&access.base_expression, write, read);
                if let Some(index) = &access.index_expression {
                    self.expression(index, false, true);
                }
            }
            Expression::IndexRangeAccess(access) => {
                self.expression(&access.base_expression, write, read);
                if let Some(start) = &access.start_expression {
                    self.expression(start, false, true);
                }
                if let Some(end) = &access.end_expression {
                    self.expression(end, false, true);
                }
            }
            Expression::Assignment(assignment) => {
                let compound = !matches!(assignment.operator, AssignmentOperator::Assign);
                self.expression(&assignment.lhs, true, compound);
                self.expression(&assignment.rhs, false, true);
            }
            Expression::UnaryOperation(operation) => match operation.operator {
                UnaryOperator::Increment | UnaryOperator::Decrement => {
                    self.expression(&operation.sub_expression, true, true);
                }
                UnaryOperator::Delete => self.expression(&operation.sub_expression, true, false),
                _ => self.expression(&operation.sub_expression, false, true),
            },
            Expression::FunctionCall(call) => self.call(call),
            Expression::FunctionCallOptions(options) => {
                self.expression(&options.expression, false, true);
                for option in &options.options {
                    self.expression(option, false, true);
                }
            }
            Expression::TupleExpression(tuple) => {
                for component in tuple.components.iter().flatten() {
                    self.expression(component, write, read);
                }
            }
            Expression::BinaryOperation(operation) => {
                self.expression(&operation.lhs, false, true);
                self.expression(&operation.rhs, false, true);
            }
            Expression::Conditional(conditional) => {
                self.expression(&conditional.condition, false, true);
                self.expression(&conditional.true_expression, write, read);
                self.expression(&conditional.false_expression, write, read);
            }
            Expression::NewExpression(_)
            | Expression::Literal(_)
            | Expression::ElementaryTypeNameExpression(_) => {}
        }
    }

    /// A call expression: the site it records (with the call options
    /// stripped), then the callee expression and arguments for the state
    /// they read. `arr.push(x)` / `arr.pop()` write the array.
    fn call(&mut self, call: &FunctionCall) {
        let target = match &call.expression {
            Expression::FunctionCallOptions(options) => {
                for option in &options.options {
                    self.expression(option, false, true);
                }
                &options.expression
            }
            other => other,
        };
        if matches!(call.kind, FunctionCallKind::FunctionCall) {
            match target {
                Expression::Identifier(identifier) => {
                    // Builtins (`require`, `keccak256`) carry negative ids.
                    if identifier.referenced_declaration.is_some_and(|id| id >= 0) {
                        self.calls.push(RawCallSite {
                            text: identifier.name.clone(),
                            name: identifier.name.clone(),
                            qualifier: None,
                            line: self.line(&call.src),
                            declaration: identifier.referenced_declaration.map(|id| id as i64),
                        });
                    }
                }
                Expression::MemberAccess(access) => {
                    if access.member_name == "push" || access.member_name == "pop" {
                        self.expression(&access.expression, true, true);
                        for argument in &call.arguments {
                            self.expression(argument, false, true);
                        }
                        return;
                    }
                    let qualifier = match &access.expression {
                        Expression::Identifier(identifier) => identifier.name.clone(),
                        Expression::MemberAccess(inner) => self.reader.text_of(&inner.src),
                        other => self.reader.text_of(&other.src()),
                    };
                    self.calls.push(RawCallSite {
                        text: self.reader.text_of(&access.src),
                        name: access.member_name.clone(),
                        qualifier: Some(qualifier),
                        line: self.line(&call.src),
                        declaration: access
                            .referenced_declaration
                            .filter(|id| *id >= 0)
                            .map(|id| id as i64),
                    });
                }
                _ => {}
            }
        }
        self.expression(target, false, true);
        for argument in &call.arguments {
            self.expression(argument, false, true);
        }
    }

    /// The declarations an expression aliases when it is bound to a storage
    /// pointer: the base of a member or index chain, through earlier aliases.
    fn storage_base(&self, expression: &Expression) -> BTreeSet<i64> {
        match expression {
            Expression::Identifier(identifier) => match identifier.referenced_declaration {
                Some(declaration) => match self.aliases.get(&(declaration as i64)) {
                    Some(targets) => targets.clone(),
                    None => BTreeSet::from([declaration as i64]),
                },
                None => BTreeSet::new(),
            },
            Expression::MemberAccess(access) => self.storage_base(&access.expression),
            Expression::IndexAccess(access) => self.storage_base(&access.base_expression),
            Expression::IndexRangeAccess(access) => self.storage_base(&access.base_expression),
            Expression::Conditional(conditional) => {
                let mut out = self.storage_base(&conditional.true_expression);
                out.extend(self.storage_base(&conditional.false_expression));
                out
            }
            Expression::TupleExpression(tuple) => tuple
                .components
                .iter()
                .flatten()
                .flat_map(|component| self.storage_base(component))
                .collect(),
            _ => BTreeSet::new(),
        }
    }

    /// Records a reference to `declaration`, or to the declarations a storage
    /// pointer local aliases. Locals and parameters end up as references the
    /// assembler drops, since no state item carries their id.
    fn touch(&mut self, declaration: i64, name: &str, write: bool, read: bool, line: usize) {
        let targets: Vec<i64> = match self.aliases.get(&declaration) {
            Some(aliased) => aliased.iter().copied().collect(),
            None => vec![declaration],
        };
        for target in targets {
            if write {
                self.refs.push(RawStateRef {
                    name: name.to_string(),
                    write: true,
                    line,
                    declaration: Some(target),
                });
            }
            if read {
                self.refs.push(RawStateRef {
                    name: name.to_string(),
                    write: false,
                    line,
                    declaration: Some(target),
                });
            }
        }
    }

    /// Inline assembly: Yul calls that leave the contract become call sites,
    /// `sstore` / `sload` and slot references become state accesses through
    /// the block's external references.
    fn assembly(&mut self, assembly: &InlineAssembly) {
        let Some(block) = &assembly.ast else {
            return;
        };
        let references: BTreeMap<usize, i64> = assembly
            .external_references
            .iter()
            .filter_map(|reference| {
                reference
                    .src
                    .start
                    .map(|start| (start, reference.declaration as i64))
            })
            .collect();
        self.yul_block(block, &references);
    }

    fn yul_block(&mut self, block: &YulBlock, references: &BTreeMap<usize, i64>) {
        for statement in &block.statements {
            self.yul_statement(statement, references);
        }
    }

    fn yul_statement(&mut self, statement: &YulStatement, references: &BTreeMap<usize, i64>) {
        match statement {
            YulStatement::YulAssignment(assignment) => {
                self.yul_expression(&assignment.value, references);
                for variable in &assignment.variable_names {
                    self.yul_identifier(variable, references, true);
                }
            }
            YulStatement::YulBlock(block) => self.yul_block(block, references),
            YulStatement::YulExpressionStatement(statement) => {
                self.yul_expression(&statement.expression, references);
            }
            YulStatement::YulForLoop(statement) => {
                self.yul_block(&statement.pre, references);
                self.yul_expression(&statement.condition, references);
                self.yul_block(&statement.post, references);
                self.yul_block(&statement.body, references);
            }
            YulStatement::YulFunctionDefinition(definition) => {
                self.yul_block(&definition.body, references);
            }
            YulStatement::YulIf(statement) => {
                self.yul_expression(&statement.condition, references);
                self.yul_block(&statement.body, references);
            }
            YulStatement::YulSwitch(statement) => {
                self.yul_expression(&statement.expression, references);
                for case in &statement.cases {
                    self.yul_case(case, references);
                }
            }
            YulStatement::YulVariableDeclaration(declaration) => {
                if let Some(value) = &declaration.value {
                    self.yul_expression(value, references);
                }
            }
            YulStatement::YulBreak(_)
            | YulStatement::YulContinue(_)
            | YulStatement::YulLeave(_) => {}
        }
    }

    fn yul_case(&mut self, case: &YulCase, references: &BTreeMap<usize, i64>) {
        self.yul_block(&case.body, references);
    }

    fn yul_expression(&mut self, expression: &YulExpression, references: &BTreeMap<usize, i64>) {
        match expression {
            YulExpression::YulFunctionCall(call) => self.yul_call(call, references),
            YulExpression::YulIdentifier(identifier) => {
                self.yul_identifier(identifier, references, false);
            }
            YulExpression::YulLiteral(_) => {}
        }
    }

    fn yul_call(&mut self, call: &YulFunctionCall, references: &BTreeMap<usize, i64>) {
        let name = call.function_name.name.as_str();
        if YUL_EXTERNAL_CALLS.contains(&name) {
            self.calls.push(RawCallSite {
                text: format!("assembly.{name}"),
                name: name.to_string(),
                qualifier: Some("assembly".to_string()),
                line: self.line(&call.src),
                declaration: None,
            });
        }
        let stores_slot = name == "sstore";
        for (position, argument) in call.arguments.iter().enumerate() {
            match argument {
                YulExpression::YulIdentifier(identifier) if stores_slot && position == 0 => {
                    self.yul_identifier(identifier, references, true);
                }
                other => self.yul_expression(other, references),
            }
        }
    }

    fn yul_identifier(
        &mut self,
        identifier: &YulIdentifier,
        references: &BTreeMap<usize, i64>,
        write: bool,
    ) {
        let Some(start) = identifier.src.start else {
            return;
        };
        if let Some(declaration) = references.get(&start).copied() {
            let name = identifier
                .name
                .split('.')
                .next()
                .unwrap_or(identifier.name.as_str())
                .to_string();
            self.touch(
                declaration,
                &name,
                write,
                !write,
                self.line(&identifier.src),
            );
        }
    }
}

/// The source location of any expression variant.
trait ExpressionSrc {
    fn src(&self) -> SourceLocation;
}

impl ExpressionSrc for Expression {
    fn src(&self) -> SourceLocation {
        match self {
            Expression::Assignment(node) => node.src.clone(),
            Expression::BinaryOperation(node) => node.src.clone(),
            Expression::Conditional(node) => node.src.clone(),
            Expression::ElementaryTypeNameExpression(node) => node.src.clone(),
            Expression::FunctionCall(node) => node.src.clone(),
            Expression::FunctionCallOptions(node) => node.src.clone(),
            Expression::Identifier(node) => node.src.clone(),
            Expression::IndexAccess(node) => node.src.clone(),
            Expression::IndexRangeAccess(node) => node.src.clone(),
            Expression::Literal(node) => node.src.clone(),
            Expression::MemberAccess(node) => node.src.clone(),
            Expression::NewExpression(node) => node.src.clone(),
            Expression::TupleExpression(node) => node.src.clone(),
            Expression::UnaryOperation(node) => node.src.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::SolcBuildInfo;
    use crate::builder::CodeGraphBuilder;
    use crate::model::{AccessKind, CalleeRef, CodeGraph, ModuleKind, ParentRef};

    fn fixture_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/solc")
    }

    async fn graph() -> CodeGraph {
        let root = fixture_root();
        let info = SolcBuildInfo::load(root.clone(), &root.join("build-info"))
            .await
            .expect("build-info loads");
        assert_eq!(info.units.len(), 1);
        CodeGraphBuilder::new(root)
            .with_solc(info)
            .build()
            .await
            .expect("graph builds")
            .graph
    }

    fn callable(graph: &CodeGraph, module: &str, name: &str) -> i64 {
        let module_id = graph
            .modules
            .values()
            .find(|m| m.name == module)
            .unwrap_or_else(|| panic!("module {module}"))
            .id;
        graph
            .callables
            .values()
            .find(|c| c.module_id == module_id && c.name == name)
            .unwrap_or_else(|| panic!("callable {module}.{name}"))
            .id
    }

    fn state(graph: &CodeGraph, module: &str, name: &str) -> i64 {
        let module_id = graph
            .modules
            .values()
            .find(|m| m.name == module)
            .unwrap_or_else(|| panic!("module {module}"))
            .id;
        graph
            .states
            .values()
            .find(|s| s.module_id == module_id && s.name == name)
            .unwrap_or_else(|| panic!("state {module}.{name}"))
            .id
    }

    #[tokio::test]
    async fn contracts_interfaces_and_inheritance_resolve_by_declaration_id() {
        let graph = graph().await;
        let kinds: Vec<(String, ModuleKind)> = graph
            .modules
            .values()
            .map(|m| (m.name.clone(), m.kind))
            .collect();
        assert!(kinds.contains(&("IToken".to_string(), ModuleKind::Interface)));
        assert!(kinds.contains(&("Impl".to_string(), ModuleKind::Contract)));
        let base = graph
            .modules
            .values()
            .find(|m| m.name == "Base")
            .expect("Base")
            .id;
        let implementation = graph
            .modules
            .values()
            .find(|m| m.name == "Impl")
            .expect("Impl")
            .id;
        assert!(
            graph.inherit_edges.iter().any(|e| {
                e.module_id == implementation && e.parent == ParentRef::Resolved(base)
            })
        );
    }

    #[tokio::test]
    async fn calls_with_options_low_level_calls_and_assembly_are_edges() {
        let graph = graph().await;
        let deposit = callable(&graph, "Vault", "deposit");
        let settle = callable(&graph, "IPool", "settle");
        let transfer = callable(&graph, "IToken", "transfer");
        let edges: Vec<_> = graph
            .call_edges
            .iter()
            .filter(|e| e.caller_id == deposit)
            .collect();
        assert!(edges.iter().any(|e| e.callee_text == "pool.settle"
            && e.callee == CalleeRef::Resolved(settle)
            && e.line == 60));
        assert!(
            edges
                .iter()
                .any(|e| e.callee_text == "IToken(token).transfer"
                    && e.callee == CalleeRef::Resolved(transfer))
        );

        let impl_transfer = callable(&graph, "Impl", "_transfer");
        assert!(graph.call_edges.iter().any(|e| e.caller_id == impl_transfer
            && e.callee_text == "to.call"
            && matches!(e.callee, CalleeRef::External(_))));

        let sweep = callable(&graph, "Vault", "sweep");
        let only_owner = callable(&graph, "Vault", "onlyOwner");
        assert!(graph.call_edges.iter().any(|e| e.caller_id == sweep
            && e.callee_text == "assembly.call"
            && matches!(e.callee, CalleeRef::External(_))));
        assert!(
            graph
                .call_edges
                .iter()
                .any(|e| e.caller_id == sweep && e.callee == CalleeRef::Resolved(only_owner))
        );
    }

    #[tokio::test]
    async fn base_declarations_resolve_and_state_writes_keep_their_lines() {
        let graph = graph().await;
        let withdraw = callable(&graph, "Base", "_withdraw");
        let base_transfer = callable(&graph, "Base", "_transfer");
        let call_line = graph
            .call_edges
            .iter()
            .find(|e| e.caller_id == withdraw && e.callee == CalleeRef::Resolved(base_transfer))
            .expect("_withdraw calls _transfer")
            .line;
        let balance = state(&graph, "Base", "balance");
        let write_line = graph
            .state_edges
            .iter()
            .find(|e| {
                e.callable_id == withdraw && e.state_id == balance && e.access == AccessKind::Write
            })
            .expect("_withdraw writes balance")
            .line;
        assert!(
            write_line > call_line,
            "write {write_line} after call {call_line}"
        );

        let deposit = callable(&graph, "Vault", "deposit");
        let positions = state(&graph, "Vault", "positions");
        let total = state(&graph, "Vault", "total");
        assert!(graph.state_edges.iter().any(|e| e.callable_id == deposit
            && e.state_id == positions
            && e.access == AccessKind::Write));
        assert!(graph.state_edges.iter().any(|e| e.callable_id == deposit
            && e.state_id == total
            && e.access == AccessKind::Write));
        let held = callable(&graph, "Vault", "held");
        let token = state(&graph, "Vault", "token");
        assert!(
            graph.state_edges.iter().any(|e| e.callable_id == held
                && e.state_id == token
                && e.access == AccessKind::Read)
        );
    }
}
