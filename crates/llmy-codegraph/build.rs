fn main() {
    // Vendored Move grammars (generated tree-sitter parsers).
    // grammars/move-aptos: aptos-labs/tree-sitter-move-on-aptos (Apache-2.0)
    // grammars/move-sui:   tzakian/tree-sitter-move (MIT)
    cc::Build::new()
        .include("grammars/move-aptos")
        .file("grammars/move-aptos/parser.c")
        .file("grammars/move-aptos/scanner.c")
        .flag_if_supported("-w")
        .compile("tree-sitter-move-aptos");

    cc::Build::new()
        .include("grammars/move-sui")
        .file("grammars/move-sui/parser.c")
        .flag_if_supported("-w")
        .compile("tree-sitter-move-sui");

    println!("cargo:rerun-if-changed=grammars");
}
