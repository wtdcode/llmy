use heck::ToSnakeCase;
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use quote::{ToTokens, format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Error, Expr, ExprLit, Ident, ItemStruct, Lit, LitStr, MetaNameValue, Result, Token, Type,
};

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(item as ItemStruct);
    let args = syn::parse_macro_input!(attr as ToolArgs);

    expand_tool(args, item_struct).into()
}

struct ToolArgs {
    description: Option<LitStr>,
    arguments: Type,
    name: Option<LitStr>,
    invoke: Ident,
}

impl Parse for ToolArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let args = Punctuated::<MetaNameValue, Token![,]>::parse_terminated(input)?;

        let mut description = None;
        let mut arguments = None;
        let mut name = None;
        let mut invoke = None;

        for arg in args {
            let key = arg
                .path
                .get_ident()
                .ok_or_else(|| Error::new_spanned(&arg.path, "expected a simple identifier"))?
                .to_string();

            match key.as_str() {
                "description" => {
                    assign_once(
                        &mut description,
                        parse_string_literal(&arg.value, "description")?,
                        "description",
                        &arg,
                    )?;
                }
                "arguments" => {
                    assign_once(
                        &mut arguments,
                        syn::parse2(arg.value.to_token_stream())?,
                        "arguments",
                        &arg,
                    )?;
                }
                "name" => {
                    assign_once(
                        &mut name,
                        parse_string_literal(&arg.value, "name")?,
                        "name",
                        &arg,
                    )?;
                }
                "invoke" => {
                    assign_once(&mut invoke, parse_method_ident(&arg.value)?, "invoke", &arg)?;
                }
                _ => {
                    return Err(Error::new_spanned(
                        &arg.path,
                        format!("unknown tool argument `{key}`"),
                    ));
                }
            }
        }

        Ok(Self {
            description,
            arguments: arguments.ok_or_else(|| {
                Error::new(Span::call_site(), "missing required `arguments` argument")
            })?,
            name,
            invoke: invoke.ok_or_else(|| {
                Error::new(Span::call_site(), "missing required `invoke` argument")
            })?,
        })
    }
}

fn assign_once<T>(slot: &mut Option<T>, value: T, key: &str, meta: &MetaNameValue) -> Result<()> {
    if slot.is_some() {
        return Err(Error::new_spanned(
            meta,
            format!("duplicate `{key}` argument"),
        ));
    }

    *slot = Some(value);
    Ok(())
}

fn parse_string_literal(expr: &Expr, key: &str) -> Result<LitStr> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(value),
            ..
        }) => Ok(value.clone()),
        _ => Err(Error::new_spanned(
            expr,
            format!("`{key}` must be a string literal"),
        )),
    }
}

fn parse_method_ident(expr: &Expr) -> Result<Ident> {
    let path: syn::Path = syn::parse2(expr.to_token_stream())?;

    path.get_ident()
        .cloned()
        .ok_or_else(|| Error::new_spanned(expr, "`invoke` must be a single method name"))
}

fn default_tool_name(ident: &Ident) -> LitStr {
    LitStr::new(&ident.to_string().to_snake_case(), ident.span())
}

fn expand_tool(args: ToolArgs, item_struct: ItemStruct) -> proc_macro2::TokenStream {
    let (tool_path, error_path) = resolve_support_paths();
    let struct_ident = &item_struct.ident;
    let (impl_generics, ty_generics, where_clause) = item_struct.generics.split_for_impl();
    let name = args.name.unwrap_or_else(|| default_tool_name(struct_ident));
    let description = match args.description {
        Some(description) => quote!(Some(#description)),
        None => quote!(None),
    };
    let arguments = args.arguments;
    let invoke = args.invoke;

    quote! {
        #item_struct

        impl #impl_generics #tool_path for #struct_ident #ty_generics #where_clause {
            type ARGUMENTS = #arguments;
            const NAME: &str = #name;
            const DESCRIPTION: Option<&str> = #description;

            fn invoke(
                &self,
                arguments: Self::ARGUMENTS,
            ) -> impl ::core::future::Future<Output = ::core::result::Result<::std::string::String, #error_path>> + Send {
                self.#invoke(arguments)
            }
        }
    }
}

fn resolve_support_paths() -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    resolve_support_paths_from(crate_name("llmy-agent").ok(), crate_name("llmy").ok())
}

fn resolve_support_paths_from(
    llmy_agent: Option<FoundCrate>,
    llmy: Option<FoundCrate>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    if let Some(paths) = resolve_llmy_agent_paths(llmy_agent) {
        return paths;
    }

    if let Some(paths) = resolve_llmy_paths(llmy) {
        return paths;
    }

    (quote!(::llmy_agent::Tool), quote!(::llmy_agent::LLMYError))
}

fn resolve_llmy_agent_paths(
    found: Option<FoundCrate>,
) -> Option<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    match found? {
        FoundCrate::Itself => Some((quote!(crate::Tool), quote!(crate::LLMYError))),
        FoundCrate::Name(name) => {
            let ident = format_ident!("{}", name);
            Some((quote!(::#ident::Tool), quote!(::#ident::LLMYError)))
        }
    }
}

fn resolve_llmy_paths(
    found: Option<FoundCrate>,
) -> Option<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    match found? {
        FoundCrate::Itself => Some((quote!(crate::agent::Tool), quote!(crate::agent::LLMYError))),
        FoundCrate::Name(name) => {
            let ident = format_ident!("{}", name);
            Some((
                quote!(::#ident::agent::Tool),
                quote!(::#ident::agent::LLMYError),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse2;

    #[test]
    fn parses_tool_arguments() {
        let args: ToolArgs = parse2(quote! {
            description = "Read a file",
            arguments = ReadFileToolArgs,
            name = "read_file",
            invoke = read_file
        })
        .expect("tool args should parse");

        assert_eq!(
            args.description.as_ref().map(LitStr::value).as_deref(),
            Some("Read a file")
        );
        assert_eq!(
            args.name.as_ref().map(LitStr::value).as_deref(),
            Some("read_file")
        );
        assert_eq!(args.invoke.to_string(), "read_file");
        assert!(matches!(args.arguments, Type::Path(_)));
    }

    #[test]
    fn rejects_missing_required_arguments() {
        let err = match parse2::<ToolArgs>(quote! {
            description = "Read a file",
            invoke = read_file
        }) {
            Ok(_) => panic!("missing arguments should fail"),
            Err(err) => err,
        };

        assert_eq!(err.to_string(), "missing required `arguments` argument");
    }

    #[test]
    fn rejects_duplicate_arguments() {
        let err = match parse2::<ToolArgs>(quote! {
            arguments = FirstArgs,
            arguments = SecondArgs,
            invoke = read_file
        }) {
            Ok(_) => panic!("duplicate arguments should fail"),
            Err(err) => err,
        };

        assert_eq!(err.to_string(), "duplicate `arguments` argument");
    }

    #[test]
    fn derives_default_name_from_struct_name() {
        let ident = Ident::new("ReadFileTool", Span::call_site());

        assert_eq!(default_tool_name(&ident).value(), "read_file_tool");
    }

    #[test]
    fn prefers_llmy_agent_when_both_crates_exist() {
        let (tool_path, error_path) = resolve_support_paths_from(
            Some(FoundCrate::Name("llmy_agent".into())),
            Some(FoundCrate::Name("llmy".into())),
        );

        assert_eq!(tool_path.to_string(), ":: llmy_agent :: Tool");
        assert_eq!(error_path.to_string(), ":: llmy_agent :: LLMYError");
    }

    #[test]
    fn falls_back_to_llmy_agent_module_when_available() {
        let (tool_path, error_path) =
            resolve_support_paths_from(None, Some(FoundCrate::Name("llmy".into())));

        assert_eq!(tool_path.to_string(), ":: llmy :: agent :: Tool");
        assert_eq!(error_path.to_string(), ":: llmy :: agent :: LLMYError");
    }

    #[test]
    fn resolves_self_paths_for_llmy_agent() {
        let (tool_path, error_path) = resolve_support_paths_from(Some(FoundCrate::Itself), None);

        assert_eq!(tool_path.to_string(), "crate :: Tool");
        assert_eq!(error_path.to_string(), "crate :: LLMYError");
    }

    #[test]
    fn resolves_self_paths_for_llmy() {
        let (tool_path, error_path) = resolve_support_paths_from(None, Some(FoundCrate::Itself));

        assert_eq!(tool_path.to_string(), "crate :: agent :: Tool");
        assert_eq!(error_path.to_string(), "crate :: agent :: LLMYError");
    }
}
