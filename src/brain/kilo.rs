//! The Kilo Intermediary Representation & Tokenizer (1024 total tokens).
//!
//! Kilo is a compact, canonical intermediate language for Rust code.
//! It addresses vocabulary explosion in small models by enforcing:
//! - Exactly 1024 total tokens in the vocabulary (IDs 0..1023).
//! - Prenamed canonical variables (`v0`..`v115`, `arg0`..`arg15`, `fld0`..`fld47`, `T0`..`T39`, `fn0`..`fn39`).
//! - Placeholders for literals (`"<str>"`, `"<char>"`, `"<bytes>"`, `0`, `0.0`).
//! - Dedicated token budget for non-standard crates (Bevy, GPUI, Serde, Tokio, Tracing, WGPU, Syn, etc.).
//! - Fully translatable back into syntactically valid Rust.

use std::collections::HashMap;
use std::sync::OnceLock;
use anyhow::{Context, Result, ensure};
use syn::visit_mut::{self, VisitMut};
use syn::{
    Item, ItemFn, ItemStruct, ItemImpl, ItemTrait, ImplItem, ImplItemFn,
    Expr, Stmt, Pat, Lit, Ident, FnArg, Member, GenericParam, TypeParamBound,
    WherePredicate, parse_quote
};
use quote::ToTokens;
use proc_macro2::{TokenTree, Group, Literal, TokenStream};

pub const VOCAB_SIZE: usize = 1024;
pub const PAD_ID: usize = 0;
pub const BOS_ID: usize = 1;
pub const EOS_ID: usize = 2;
pub const UNK_ID: usize = 3;

/// Canonical 1024-token vocabulary.
pub static VOCAB: [&str; VOCAB_SIZE] = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "as",
    "async",
    "await",
    "break",
    "const",
    "continue",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
    "yield",
    "default",
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    ";",
    ":",
    "::",
    ",",
    ".",
    "->",
    "=>",
    "?",
    "!",
    "&",
    "&&",
    "|",
    "||",
    "^",
    "~",
    "+",
    "-",
    "*",
    "/",
    "%",
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "<<",
    ">>",
    "..",
    "..=",
    "_",
    "#",
    "@",
    "$",
    "bool",
    "char",
    "str",
    "String",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
    "Option",
    "Some",
    "None",
    "Result",
    "Ok",
    "Err",
    "Vec",
    "Box",
    "Rc",
    "Arc",
    "Cell",
    "RefCell",
    "Mutex",
    "RwLock",
    "HashMap",
    "HashSet",
    "BTreeMap",
    "BTreeSet",
    "VecDeque",
    "Duration",
    "Instant",
    "Path",
    "PathBuf",
    "File",
    "Read",
    "Write",
    "PhantomData",
    "Pin",
    "Poll",
    "Future",
    "Clone",
    "Copy",
    "Debug",
    "Default",
    "Display",
    "Error",
    "Eq",
    "PartialEq",
    "Ord",
    "PartialOrd",
    "Hash",
    "Send",
    "Sync",
    "Sized",
    "Unpin",
    "Iterator",
    "IntoIterator",
    "Extend",
    "From",
    "Into",
    "AsRef",
    "AsMut",
    "Borrow",
    "BorrowMut",
    "Drop",
    "Deref",
    "DerefMut",
    "Fn",
    "FnMut",
    "FnOnce",
    "new",
    "clone",
    "to_string",
    "as_str",
    "as_bytes",
    "as_ref",
    "as_mut",
    "len",
    "is_empty",
    "push",
    "pop",
    "insert",
    "remove",
    "get",
    "get_mut",
    "contains",
    "clear",
    "retain",
    "iter",
    "iter_mut",
    "into_iter",
    "map",
    "filter",
    "filter_map",
    "find",
    "any",
    "all",
    "collect",
    "take",
    "skip",
    "enumerate",
    "zip",
    "fold",
    "for_each",
    "next",
    "count",
    "unwrap",
    "expect",
    "unwrap_or",
    "unwrap_or_else",
    "unwrap_or_default",
    "is_some",
    "is_none",
    "is_ok",
    "is_err",
    "and_then",
    "or_else",
    "ok",
    "err",
    "transpose",
    "flatten",
    "replace",
    "abs",
    "min",
    "max",
    "clamp",
    "round",
    "floor",
    "ceil",
    "trim",
    "split",
    "lines",
    "starts_with",
    "ends_with",
    "println",
    "eprintln",
    "format",
    "print",
    "eprint",
    "write",
    "writeln",
    "vec",
    "panic",
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "todo",
    "unimplemented",
    "unreachable",
    "matches",
    "derive",
    "inline",
    "test",
    "cfg",
    "cfg_attr",
    "allow",
    "warn",
    "deny",
    "repr",
    "doc",
    "feature",
    r#""<str>""#,
    r#""<char>""#,
    r#""<bytes>""#,
    r#""<num>""#,
    "true",
    "false",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "12",
    "16",
    "24",
    "32",
    "48",
    "64",
    "100",
    "128",
    "255",
    "256",
    "512",
    "1024",
    "-1",
    "-2",
    "0.0",
    "0.5",
    "1.0",
    "2.0",
    "0.1",
    "f32_inf",
    "f32_nan",
    r#""""#,
    r#"" ""#,
    r#""
""#,
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",
    "v29",
    "v30",
    "v31",
    "v32",
    "v33",
    "v34",
    "v35",
    "v36",
    "v37",
    "v38",
    "v39",
    "v40",
    "v41",
    "v42",
    "v43",
    "v44",
    "v45",
    "v46",
    "v47",
    "v48",
    "v49",
    "v50",
    "v51",
    "v52",
    "v53",
    "v54",
    "v55",
    "v56",
    "v57",
    "v58",
    "v59",
    "v60",
    "v61",
    "v62",
    "v63",
    "v64",
    "v65",
    "v66",
    "v67",
    "v68",
    "v69",
    "v70",
    "v71",
    "v72",
    "v73",
    "v74",
    "v75",
    "v76",
    "v77",
    "v78",
    "v79",
    "v80",
    "v81",
    "v82",
    "v83",
    "v84",
    "v85",
    "v86",
    "v87",
    "v88",
    "v89",
    "v90",
    "v91",
    "v92",
    "v93",
    "v94",
    "v95",
    "v96",
    "v97",
    "v98",
    "v99",
    "v100",
    "v101",
    "v102",
    "v103",
    "v104",
    "v105",
    "v106",
    "v107",
    "v108",
    "v109",
    "v110",
    "v111",
    "v112",
    "v113",
    "v114",
    "v115",
    "arg0",
    "arg1",
    "arg2",
    "arg3",
    "arg4",
    "arg5",
    "arg6",
    "arg7",
    "arg8",
    "arg9",
    "arg10",
    "arg11",
    "arg12",
    "arg13",
    "arg14",
    "arg15",
    "cx",
    "ctx",
    "window",
    "state",
    "event",
    "req",
    "res",
    "app",
    "i",
    "j",
    "k",
    "x",
    "y",
    "z",
    "w",
    "idx",
    "pos",
    "offset",
    "item",
    "elem",
    "val",
    "key",
    "entry",
    "opt",
    "res_val",
    "acc",
    "msg",
    "chunk",
    "byte",
    "line",
    "node",
    "T0",
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "T13",
    "T14",
    "T15",
    "T16",
    "T17",
    "T18",
    "T19",
    "T20",
    "T21",
    "T22",
    "T23",
    "T24",
    "T25",
    "T26",
    "T27",
    "T28",
    "T29",
    "T30",
    "T31",
    "T32",
    "T33",
    "T34",
    "T35",
    "T36",
    "T37",
    "T38",
    "T39",
    "Variant0",
    "Variant1",
    "Variant2",
    "Variant3",
    "Variant4",
    "Variant5",
    "Variant6",
    "Variant7",
    "Variant8",
    "Variant9",
    "Variant10",
    "Variant11",
    "Variant12",
    "Variant13",
    "Variant14",
    "Variant15",
    "Trait0",
    "Trait1",
    "Trait2",
    "Trait3",
    "Trait4",
    "Trait5",
    "Trait6",
    "Trait7",
    "fn0",
    "fn1",
    "fn2",
    "fn3",
    "fn4",
    "fn5",
    "fn6",
    "fn7",
    "fn8",
    "fn9",
    "fn10",
    "fn11",
    "fn12",
    "fn13",
    "fn14",
    "fn15",
    "fn16",
    "fn17",
    "fn18",
    "fn19",
    "fn20",
    "fn21",
    "fn22",
    "fn23",
    "fn24",
    "fn25",
    "fn26",
    "fn27",
    "fn28",
    "fn29",
    "fn30",
    "fn31",
    "fn32",
    "fn33",
    "fn34",
    "fn35",
    "fn36",
    "fn37",
    "fn38",
    "fn39",
    "build",
    "update",
    "init",
    "run",
    "handle",
    "process",
    "parse",
    "render",
    "check",
    "send",
    "recv",
    "start",
    "stop",
    "reset",
    "flush",
    "close",
    "step",
    "apply",
    "dispatch",
    "execute",
    "setup",
    "teardown",
    "validate",
    "compute",
    "draw",
    "poll",
    "emit",
    "listen",
    "fld0",
    "fld1",
    "fld2",
    "fld3",
    "fld4",
    "fld5",
    "fld6",
    "fld7",
    "fld8",
    "fld9",
    "fld10",
    "fld11",
    "fld12",
    "fld13",
    "fld14",
    "fld15",
    "fld16",
    "fld17",
    "fld18",
    "fld19",
    "fld20",
    "fld21",
    "fld22",
    "fld23",
    "fld24",
    "fld25",
    "fld26",
    "fld27",
    "fld28",
    "fld29",
    "fld30",
    "fld31",
    "fld32",
    "fld33",
    "fld34",
    "fld35",
    "fld36",
    "fld37",
    "fld38",
    "fld39",
    "fld40",
    "fld41",
    "fld42",
    "fld43",
    "fld44",
    "fld45",
    "fld46",
    "fld47",
    "id",
    "name",
    "data",
    "width",
    "height",
    "value",
    "index",
    "size",
    "inner",
    "parent",
    "child",
    "kind",
    "error",
    "result",
    "status",
    "target",
    "source",
    "path",
    "buffer",
    "time",
    "tag",
    "config",
    "ptr",
    "T",
    "U",
    "V",
    "W",
    "A",
    "B",
    "C",
    "D",
    "E",
    "R",
    "Item",
    "Output",
    "Key",
    "Val",
    "Context",
    "State",
    "'a",
    "'b",
    "'c",
    "'static",
    "'life0",
    "'life1",
    "'life2",
    "'life3",
    "Gen0",
    "Gen1",
    "Gen2",
    "Gen3",
    "Gen4",
    "Gen5",
    "Gen6",
    "Gen7",
    "std",
    "core",
    "alloc",
    "io",
    "fmt",
    "sync",
    "fs",
    "collections",
    "ops",
    "mem",
    "slice",
    "mod0",
    "mod1",
    "mod2",
    "mod3",
    "mod4",
    "mod5",
    "mod6",
    "mod7",
    "mod8",
    "mod9",
    "mod10",
    "mod11",
    "mod12",
    "mod13",
    "mod14",
    "mod15",
    "Entity",
    "Component",
    "Resource",
    "Query",
    "Commands",
    "App",
    "Plugin",
    "Res",
    "ResMut",
    "Event",
    "EventReader",
    "EventWriter",
    "Update",
    "Startup",
    "PreUpdate",
    "PostUpdate",
    "FixedUpdate",
    "Vec2",
    "Vec3",
    "Vec4",
    "Quat",
    "Mat4",
    "Transform",
    "GlobalTransform",
    "Handle",
    "Asset",
    "Assets",
    "Timer",
    "Parent",
    "Children",
    "With",
    "Without",
    "Added",
    "Changed",
    "Or",
    "despawn",
    "world",
    "system",
    "bundle",
    "Reflect",
    "reflect",
    "Window",
    "View",
    "Model",
    "Element",
    "Bounds",
    "Point",
    "Pixels",
    "Render",
    "EventEmitter",
    "TestAppContext",
    "VisualTestContext",
    "Color",
    "Rgba",
    "Hsla",
    "rgba",
    "hsla",
    "rgb",
    "TextStyle",
    "FontId",
    "div",
    "v_flex",
    "h_flex",
    "InteractiveElement",
    "StatefulInteractiveElement",
    "SharedString",
    "AnyView",
    "paint",
    "layout",
    "style",
    "on_click",
    "cursor",
    "Serialize",
    "Deserialize",
    "serde",
    "serde_json",
    "to_string_pretty",
    "from_slice",
    "from_reader",
    "from_str",
    "to_value",
    "Value",
    "Serializer",
    "Deserializer",
    "Visitor",
    "rename",
    "rename_all",
    "skip_serializing_if",
    "borrow",
    "Object",
    "Array",
    "Number",
    "Bool",
    "Null",
    "to_vec",
    "from_value",
    "tokio",
    "spawn",
    "sleep",
    "channel",
    "Sender",
    "Receiver",
    "oneshot",
    "mpsc",
    "broadcast",
    "watch",
    "AsyncRead",
    "AsyncWrite",
    "AsyncSeek",
    "AsyncReadExt",
    "AsyncWriteExt",
    "select",
    "join",
    "block_on",
    "task",
    "timeout",
    "Interval",
    "JoinHandle",
    "Notify",
    "Barrier",
    "Semaphore",
    "yield_now",
    "ready",
    "pending",
    "tracing",
    "log",
    "info",
    "debug",
    "trace",
    "instrument",
    "span",
    "Level",
    "record",
    "enabled",
    "subscriber",
    "anyhow",
    "thiserror",
    "bail",
    "ensure",
    "anyhow_error",
    "with_context",
    "context",
    "custom",
    "root_cause",
    "downcast",
    "downcast_ref",
    "downcast_mut",
    "backtrace",
    "wgpu",
    "Device",
    "Queue",
    "Buffer",
    "Texture",
    "TextureView",
    "Sampler",
    "ShaderStages",
    "BindGroup",
    "BindGroupLayout",
    "PipelineLayout",
    "RenderPipeline",
    "ComputePipeline",
    "CommandEncoder",
    "RenderPass",
    "ComputePass",
    "BufferDescriptor",
    "TextureDescriptor",
    "ShaderModule",
    "ColorTargetState",
    "VertexState",
    "FragmentState",
    "PrimitiveState",
    "BlendState",
    "DepthStencilState",
    "MultisampleState",
    "Extent3d",
    "BufferUsages",
    "TextureUsages",
    "ShaderLocation",
    "VertexBufferLayout",
    "VertexAttribute",
    "syn",
    "quote",
    "proc_macro",
    "proc_macro2",
    "TokenStream",
    "Ident",
    "Span",
    "parse_macro_input",
    "parse_file",
    "DeriveInput",
    "Data",
    "Fields",
    "Expr",
    "Stmt",
    "Pat",
    "Block",
    "Lit",
    "Generics",
    "Attribute",
    "Meta",
    "crate0",
    "crate1",
    "crate2",
    "crate3",
    "crate4",
    "crate5",
    "crate6",
    "crate7",
    "ext_type0",
    "ext_type1",
    "ext_type2",
    "ext_type3",
    "ext_type4",
    "ext_type5",
    "ext_type6",
    "ext_type7",
    "ext_type8",
    "ext_type9",
    "ext_type10",
    "ext_type11",
    "ext_type12",
    "ext_type13",
    "ext_type14",
    "ext_type15",
    "ext_type16",
    "ext_type17",
    "ext_type18",
    "ext_type19",
    "ext_type20",
    "ext_type21",
    "ext_type22",
    "ext_type23",
    "ext_fn0",
    "ext_fn1",
    "ext_fn2",
    "ext_fn3",
    "ext_fn4",
    "ext_fn5",
    "ext_fn6",
    "ext_fn7",
    "ext_fn8",
    "ext_fn9",
    "ext_fn10",
    "ext_fn11",
    "ext_fn12",
    "ext_fn13",
    "ext_fn14",
    "ext_fn15",
    "ext_trait0",
    "ext_trait1",
    "ext_trait2",
    "ext_trait3",
    "ext_trait4",
    "ext_trait5",
    "ext_trait6",
    "ext_trait7",
    "raw",
    "box",
    "typeof",
    "abstract",
    "become",
    "do",
    "final",
    "override",
    "priv",
    "unsized",
    "virtual",
    "NonZeroUsize",
    "NonZeroU32",
    "AtomicBool",
    "AtomicUsize",
    "AtomicI32",
    "Ordering",
    "Relaxed",
    "SeqCst",
    "Acquire",
    "Release",
    "Cow",
    "Cursor",
    "Seek",
    "SeekFrom",
    "ErrorKind",
    "ExitStatus",
    "PI",
    "TAU",
    "INFINITY",
    "NEG_INFINITY",
    "NAN",
    "sin",
    "cos",
    "tan",
    "sqrt",
    "powf",
    "powi"
];

static TOKEN_MAP: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();

pub fn token_map() -> &'static HashMap<&'static str, usize> {
    TOKEN_MAP.get_or_init(|| {
        let mut m = HashMap::with_capacity(VOCAB_SIZE);
        for (i, &t) in VOCAB.iter().enumerate() {
            m.insert(t, i);
        }
        m
    })
}

pub fn token_to_id(token: &str) -> Option<usize> {
    token_map().get(token).copied()
}

pub fn id_to_token(id: usize) -> Option<&'static str> {
    VOCAB.get(id).copied()
}

/// Tokenize Kilo source code into discrete token IDs in `0..1024`.
pub fn tokenize_kilo(code: &str) -> Vec<usize> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = code.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // 1. Skip whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        // 2. Check for placeholder strings like "<str>", "<char>", etc.
        if chars[i] == '"' {
            let start = i;
            i += 1;
            while i < len && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < len && chars[i] == '"' {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            let id = token_to_id(&s).unwrap_or_else(|| {
                token_to_id(r#""<str>""#).unwrap_or(UNK_ID)
            });
            tokens.push(id);
            continue;
        }

        // 3. Check for byte string literals b"..."
        if i + 1 < len && chars[i] == 'b' && chars[i + 1] == '"' {
            let start = i;
            i += 2;
            while i < len && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < len {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i < len && chars[i] == '"' {
                i += 1;
            }
            tokens.push(token_to_id(r#""<bytes>""#).unwrap_or(UNK_ID));
            continue;
        }

        // 4. Check for lifetime like 'a, 'static
        if chars[i] == '\'' && i + 1 < len && (chars[i + 1].is_alphabetic() || chars[i + 1] == '_') {
            let start = i;
            i += 1;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push(token_to_id(&s).unwrap_or_else(|| {
                token_to_id("'a").unwrap_or(UNK_ID)
            }));
            continue;
        }

        // 5. Identifiers, keywords, numbers, prenamed registers
        if chars[i].is_alphanumeric() || chars[i] == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || (chars[i] == '.' && i + 1 < len && chars[i + 1].is_ascii_digit())) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            if let Some(id) = token_to_id(&word) {
                tokens.push(id);
            } else {
                tokens.push(UNK_ID);
            }
            continue;
        }

        // 6. Multi-character punctuation
        if i + 3 <= len {
            let s: String = chars[i..i + 3].iter().collect();
            if let Some(id) = token_to_id(&s) {
                tokens.push(id);
                i += 3;
                continue;
            }
        }

        if i + 2 <= len {
            let s: String = chars[i..i + 2].iter().collect();
            if let Some(id) = token_to_id(&s) {
                tokens.push(id);
                i += 2;
                continue;
            }
        }

        // 7. Single-character punctuation
        let single = chars[i].to_string();
        if let Some(id) = token_to_id(&single) {
            tokens.push(id);
            i += 1;
            continue;
        }

        tokens.push(UNK_ID);
        i += 1;
    }

    tokens
}

/// Detokenize Kilo token IDs into readable, syntactically valid Rust source code.
pub fn detokenize_kilo(ids: &[usize]) -> String {
    let mut out = String::new();
    let mut indent: usize = 0;
    let mut newline_needed = false;

    let append_indent = |s: &mut String, lvl: usize| {
        for _ in 0..lvl {
            s.push_str("    ");
        }
    };

    for &id in ids.iter() {
        if id == PAD_ID || id == BOS_ID || id == EOS_ID {
            continue;
        }
        let token = id_to_token(id).unwrap_or("<UNK>");

        if token == "}" {
            indent = indent.saturating_sub(1);
            if !out.ends_with('\n') {
                out.push('\n');
            }
            append_indent(&mut out, indent);
            out.push_str("}");
            out.push('\n');
            newline_needed = true;
            continue;
        }

        if newline_needed {
            append_indent(&mut out, indent);
            newline_needed = false;
        }

        if token == "{" {
            out.push(' ');
            out.push_str("{");
            out.push('\n');
            indent += 1;
            newline_needed = true;
            continue;
        }

        if token == ";" {
            out.push_str(";\n");
            newline_needed = true;
            continue;
        }

        if token == "," {
            out.push_str(", ");
            continue;
        }

        if token == ":" {
            out.push_str(": ");
            continue;
        }

        if token == "::" {
            out.push_str("::");
            continue;
        }

        if token == "." {
            out.push('.');
            continue;
        }

        if token == "(" || token == "[" {
            out.push_str(token);
            continue;
        }

        if token == ")" || token == "]" {
            out.push_str(token);
            continue;
        }

        // Leading space before operators or identifiers
        if !out.is_empty() && !out.ends_with('\n') && !out.ends_with(' ') && !out.ends_with('(') && !out.ends_with('[') && !out.ends_with('.') && !out.ends_with("::") {
            out.push(' ');
        }

        out.push_str(token);
    }

    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SymbolKind {
    Local,
    Arg,
    Field,
    Type,
    Fn,
    General,
}

struct KiloNormalizer {
    local_map: HashMap<String, String>,
    arg_map: HashMap<String, String>,
    field_map: HashMap<String, String>,
    type_map: HashMap<String, String>,
    fn_map: HashMap<String, String>,
    local_idx: usize,
    arg_idx: usize,
    field_idx: usize,
    type_idx: usize,
    fn_idx: usize,
}

impl KiloNormalizer {
    fn new() -> Self {
        Self {
            local_map: HashMap::new(),
            arg_map: HashMap::new(),
            field_map: HashMap::new(),
            type_map: HashMap::new(),
            fn_map: HashMap::new(),
            local_idx: 0,
            arg_idx: 0,
            field_idx: 0,
            type_idx: 0,
            fn_idx: 0,
        }
    }

    fn normalize_ident(&mut self, ident: &mut Ident, kind: SymbolKind) {
        let s = ident.to_string();
        // If ident is already a known token in Kilo VOCAB, preserve it!
        if token_to_id(&s).is_some() {
            return;
        }

        let mapped = match kind {
            SymbolKind::Local => {
                self.local_map.entry(s).or_insert_with(|| {
                    let name = format!("v{}", self.local_idx % 116);
                    self.local_idx += 1;
                    name
                }).clone()
            }
            SymbolKind::Arg => {
                self.arg_map.entry(s).or_insert_with(|| {
                    let name = format!("arg{}", self.arg_idx % 16);
                    self.arg_idx += 1;
                    name
                }).clone()
            }
            SymbolKind::Field => {
                self.field_map.entry(s).or_insert_with(|| {
                    let name = format!("fld{}", self.field_idx % 48);
                    self.field_idx += 1;
                    name
                }).clone()
            }
            SymbolKind::Type => {
                self.type_map.entry(s).or_insert_with(|| {
                    let name = format!("T{}", self.type_idx % 40);
                    self.type_idx += 1;
                    name
                }).clone()
            }
            SymbolKind::Fn => {
                self.fn_map.entry(s).or_insert_with(|| {
                    let name = format!("fn{}", self.fn_idx % 40);
                    self.fn_idx += 1;
                    name
                }).clone()
            }
            SymbolKind::General => {
                if s.chars().next().map_or(false, |c| c.is_uppercase()) {
                    self.type_map.entry(s).or_insert_with(|| {
                        let name = format!("T{}", self.type_idx % 40);
                        self.type_idx += 1;
                        name
                    }).clone()
                } else if let Some(m) = self.local_map.get(&s) {
                    m.clone()
                } else if let Some(m) = self.arg_map.get(&s) {
                    m.clone()
                } else if let Some(m) = self.field_map.get(&s) {
                    m.clone()
                } else if let Some(m) = self.fn_map.get(&s) {
                    m.clone()
                } else {
                    self.local_map.entry(s).or_insert_with(|| {
                        let name = format!("v{}", self.local_idx % 116);
                        self.local_idx += 1;
                        name
                    }).clone()
                }
            }
        };

        *ident = Ident::new(&mapped, ident.span());
    }

    fn filter_attrs(attrs: &mut Vec<syn::Attribute>) {
        attrs.retain(|a| {
            if a.path().is_ident("derive") || a.path().is_ident("inline") || a.path().is_ident("test") {
                true
            } else {
                false
            }
        });
    }

    fn normalize_token_stream(&mut self, stream: TokenStream) -> TokenStream {
        let mut out = Vec::new();
        for tt in stream {
            match tt {
                TokenTree::Group(group) => {
                    let inner = self.normalize_token_stream(group.stream());
                    out.push(TokenTree::Group(Group::new(group.delimiter(), inner)));
                }
                TokenTree::Ident(mut ident) => {
                    self.normalize_ident(&mut ident, SymbolKind::General);
                    out.push(TokenTree::Ident(ident));
                }
                TokenTree::Literal(lit) => {
                    let s = lit.to_string();
                    if s.starts_with('"') {
                        out.push(TokenTree::Literal(Literal::string("<str>")));
                    } else if s.starts_with('b') {
                        out.push(TokenTree::Literal(Literal::byte_string(b"<bytes>")));
                    } else if s.starts_with('\'') {
                        out.push(TokenTree::Literal(Literal::character('c')));
                    } else if token_to_id(&s).is_some() {
                        out.push(TokenTree::Literal(lit));
                    } else if s.contains('.') {
                        out.push(TokenTree::Literal(Literal::f64_unsuffixed(0.0)));
                    } else {
                        out.push(TokenTree::Literal(Literal::i64_unsuffixed(0)));
                    }
                }
                TokenTree::Punct(p) => {
                    out.push(TokenTree::Punct(p));
                }
            }
        }
        out.into_iter().collect()
    }
}

impl VisitMut for KiloNormalizer {
    fn visit_item_fn_mut(&mut self, item_fn: &mut ItemFn) {
        self.local_map.clear();
        self.arg_map.clear();
        self.local_idx = 0;
        self.arg_idx = 0;

        self.normalize_ident(&mut item_fn.sig.ident, SymbolKind::Fn);
        Self::filter_attrs(&mut item_fn.attrs);

        self.visit_generics_mut(&mut item_fn.sig.generics);

        for input in &mut item_fn.sig.inputs {
            if let FnArg::Typed(pat_type) = input {
                if let Pat::Ident(pat_ident) = &mut *pat_type.pat {
                    self.normalize_ident(&mut pat_ident.ident, SymbolKind::Arg);
                }
                self.visit_type_mut(&mut pat_type.ty);
            }
        }

        if let syn::ReturnType::Type(_, ref mut ret_ty) = item_fn.sig.output {
            self.visit_type_mut(ret_ty);
        }

        self.visit_block_mut(&mut item_fn.block);
    }

    fn visit_item_struct_mut(&mut self, item_struct: &mut ItemStruct) {
        self.normalize_ident(&mut item_struct.ident, SymbolKind::Type);
        Self::filter_attrs(&mut item_struct.attrs);
        self.visit_generics_mut(&mut item_struct.generics);

        for field in &mut item_struct.fields {
            Self::filter_attrs(&mut field.attrs);
            if let Some(ident) = &mut field.ident {
                self.normalize_ident(ident, SymbolKind::Field);
            }
            self.visit_type_mut(&mut field.ty);
        }
    }

    fn visit_item_impl_mut(&mut self, item_impl: &mut ItemImpl) {
        Self::filter_attrs(&mut item_impl.attrs);
        self.visit_generics_mut(&mut item_impl.generics);

        if let Some((_, ref mut trait_path, _)) = item_impl.trait_ {
            for seg in &mut trait_path.segments {
                self.normalize_ident(&mut seg.ident, SymbolKind::Type);
            }
        }
        self.visit_type_mut(&mut item_impl.self_ty);

        for item in &mut item_impl.items {
            if let ImplItem::Fn(impl_fn) = item {
                self.visit_impl_item_fn_mut(impl_fn);
            }
        }
    }

    fn visit_impl_item_fn_mut(&mut self, impl_fn: &mut ImplItemFn) {
        self.local_map.clear();
        self.arg_map.clear();
        self.local_idx = 0;
        self.arg_idx = 0;

        self.normalize_ident(&mut impl_fn.sig.ident, SymbolKind::Fn);
        Self::filter_attrs(&mut impl_fn.attrs);
        self.visit_generics_mut(&mut impl_fn.sig.generics);

        for input in &mut impl_fn.sig.inputs {
            if let FnArg::Typed(pat_type) = input {
                if let Pat::Ident(pat_ident) = &mut *pat_type.pat {
                    self.normalize_ident(&mut pat_ident.ident, SymbolKind::Arg);
                }
                self.visit_type_mut(&mut pat_type.ty);
            }
        }

        if let syn::ReturnType::Type(_, ref mut ret_ty) = impl_fn.sig.output {
            self.visit_type_mut(ret_ty);
        }

        self.visit_block_mut(&mut impl_fn.block);
    }

    fn visit_item_trait_mut(&mut self, item_trait: &mut ItemTrait) {
        self.normalize_ident(&mut item_trait.ident, SymbolKind::Type);
        Self::filter_attrs(&mut item_trait.attrs);
        self.visit_generics_mut(&mut item_trait.generics);
        for item in &mut item_trait.items {
            if let syn::TraitItem::Fn(trait_fn) = item {
                self.normalize_ident(&mut trait_fn.sig.ident, SymbolKind::Fn);
                Self::filter_attrs(&mut trait_fn.attrs);
                if let Some(ref mut default_block) = trait_fn.default {
                    self.visit_block_mut(default_block);
                }
            }
        }
    }

    fn visit_generics_mut(&mut self, generics: &mut syn::Generics) {
        for param in &mut generics.params {
            match param {
                GenericParam::Type(type_param) => {
                    self.normalize_ident(&mut type_param.ident, SymbolKind::Type);
                    for bound in &mut type_param.bounds {
                        if let TypeParamBound::Trait(trait_bound) = bound {
                            for seg in &mut trait_bound.path.segments {
                                self.normalize_ident(&mut seg.ident, SymbolKind::Type);
                            }
                        }
                    }
                }
                GenericParam::Lifetime(lifetime_def) => {
                    let s = lifetime_def.lifetime.to_string();
                    if token_to_id(&s).is_none() {
                        lifetime_def.lifetime = parse_quote!('a);
                    }
                }
                _ => {}
            }
        }
        if let Some(ref mut where_clause) = generics.where_clause {
            for predicate in &mut where_clause.predicates {
                if let WherePredicate::Type(pred_type) = predicate {
                    self.visit_type_mut(&mut pred_type.bounded_ty);
                    for bound in &mut pred_type.bounds {
                        if let TypeParamBound::Trait(trait_bound) = bound {
                            for seg in &mut trait_bound.path.segments {
                                self.normalize_ident(&mut seg.ident, SymbolKind::Type);
                            }
                        }
                    }
                }
            }
        }
    }

    fn visit_type_path_mut(&mut self, type_path: &mut syn::TypePath) {
        for seg in &mut type_path.path.segments {
            self.normalize_ident(&mut seg.ident, SymbolKind::Type);
            self.visit_path_arguments_mut(&mut seg.arguments);
        }
    }

    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        if let Stmt::Local(local) = stmt {
            self.visit_pat_mut(&mut local.pat);
            if let Some(ref mut init) = local.init {
                self.visit_expr_mut(&mut init.expr);
            }
        } else {
            visit_mut::visit_stmt_mut(self, stmt);
        }
    }

    fn visit_pat_ident_mut(&mut self, pat_ident: &mut syn::PatIdent) {
        self.normalize_ident(&mut pat_ident.ident, SymbolKind::Local);
        visit_mut::visit_pat_ident_mut(self, pat_ident);
    }

    fn visit_expr_path_mut(&mut self, expr_path: &mut syn::ExprPath) {
        if expr_path.path.segments.len() == 1 {
            self.normalize_ident(&mut expr_path.path.segments[0].ident, SymbolKind::General);
        } else {
            for seg in &mut expr_path.path.segments {
                let s = seg.ident.to_string();
                if token_to_id(&s).is_none() {
                    self.normalize_ident(&mut seg.ident, SymbolKind::General);
                }
            }
        }
        visit_mut::visit_expr_path_mut(self, expr_path);
    }

    fn visit_expr_struct_mut(&mut self, expr_struct: &mut syn::ExprStruct) {
        for field in &mut expr_struct.fields {
            if let Member::Named(ident) = &mut field.member {
                self.normalize_ident(ident, SymbolKind::Field);
            }
        }
        visit_mut::visit_expr_struct_mut(self, expr_struct);
    }

    fn visit_expr_field_mut(&mut self, expr_field: &mut syn::ExprField) {
        if let Member::Named(ident) = &mut expr_field.member {
            self.normalize_ident(ident, SymbolKind::Field);
        }
        visit_mut::visit_expr_field_mut(self, expr_field);
    }

    fn visit_expr_method_call_mut(&mut self, method_call: &mut syn::ExprMethodCall) {
        self.normalize_ident(&mut method_call.method, SymbolKind::Fn);
        visit_mut::visit_expr_method_call_mut(self, method_call);
    }

    fn visit_macro_mut(&mut self, mac: &mut syn::Macro) {
        mac.tokens = self.normalize_token_stream(mac.tokens.clone());
        visit_mut::visit_macro_mut(self, mac);
    }

    fn visit_ident_mut(&mut self, ident: &mut Ident) {
        let s = ident.to_string();
        if token_to_id(&s).is_none() {
            self.normalize_ident(ident, SymbolKind::General);
        }
    }

    fn visit_lit_mut(&mut self, lit: &mut Lit) {
        match lit {
            Lit::Str(_) => {
                *lit = parse_quote!("<str>");
            }
            Lit::ByteStr(_) => {
                *lit = parse_quote!(b"<bytes>");
            }
            Lit::Char(_) => {
                *lit = parse_quote!('c');
            }
            Lit::Int(int_lit) => {
                let s = int_lit.to_string();
                if token_to_id(&s).is_none() {
                    *lit = parse_quote!(0);
                }
            }
            Lit::Float(float_lit) => {
                let s = float_lit.to_string();
                if token_to_id(&s).is_none() {
                    *lit = parse_quote!(0.0);
                }
            }
            _ => {}
        }
    }
}

/// Find all unrecognized tokens in a code snippet.
pub fn find_unknown_tokens(code: &str) -> Vec<String> {
    let mut unks = Vec::new();
    let chars: Vec<char> = code.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        if chars[i] == '"' {
            let start = i;
            i += 1;
            while i < len && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < len { i += 2; } else { i += 1; }
            }
            if i < len && chars[i] == '"' { i += 1; }
            let s: String = chars[start..i].iter().collect();
            if token_to_id(&s).is_none() && token_to_id(r#""<str>""#).is_none() {
                unks.push(s);
            }
            continue;
        }
        if i + 1 < len && chars[i] == 'b' && chars[i + 1] == '"' {
            let start = i;
            i += 2;
            while i < len && chars[i] != '"' {
                if chars[i] == '\\' && i + 1 < len { i += 2; } else { i += 1; }
            }
            if i < len && chars[i] == '"' { i += 1; }
            continue;
        }
        if chars[i] == '\'' && i + 1 < len && (chars[i + 1].is_alphabetic() || chars[i + 1] == '_') {
            let start = i;
            i += 1;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') { i += 1; }
            let s: String = chars[start..i].iter().collect();
            if token_to_id(&s).is_none() { unks.push(s); }
            continue;
        }
        if chars[i].is_alphanumeric() || chars[i] == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_' || (chars[i] == '.' && i + 1 < len && chars[i + 1].is_ascii_digit())) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            if token_to_id(&word).is_none() {
                unks.push(word);
            }
            continue;
        }
        if i + 3 <= len && token_to_id(&chars[i..i + 3].iter().collect::<String>()).is_some() {
            i += 3;
            continue;
        }
        if i + 2 <= len && token_to_id(&chars[i..i + 2].iter().collect::<String>()).is_some() {
            i += 2;
            continue;
        }
        let single = chars[i].to_string();
        if token_to_id(&single).is_some() {
            i += 1;
            continue;
        }
        unks.push(single);
        i += 1;
    }
    unks
}

/// Convert Rust source code chunk into normalized Kilo representation.
pub fn rust_to_kilo(rust_code: &str) -> Result<String> {
    let text = rust_code.strip_prefix('\u{feff}').unwrap_or(rust_code);
    let mut item: Item = syn::parse_str(text)
        .or_else(|_| {
            let file = syn::parse_file(text)?;
            file.items.into_iter().next().context("no items in Rust source")
        })?;

    let mut normalizer = KiloNormalizer::new();
    normalizer.visit_item_mut(&mut item);

    let kilo_str = item.to_token_stream().to_string();
    let unks = find_unknown_tokens(&kilo_str);
    ensure!(unks.is_empty(), "Kilo conversion produced UNK tokens: {:?}", unks);

    let tokens = tokenize_kilo(&kilo_str);
    ensure!(!tokens.is_empty(), "empty Kilo token sequence");

    Ok(detokenize_kilo(&tokens))
}

/// Translate Kilo code back into valid Rust.
pub fn kilo_to_rust(kilo_code: &str) -> Result<String> {
    let tokens = tokenize_kilo(kilo_code);
    ensure!(!tokens.is_empty(), "empty Kilo tokens");
    let rust_code = detokenize_kilo(&tokens);
    // Verify syntactic validity
    syn::parse_file(&rust_code).context("parsing reconstructed Rust code")?;
    Ok(rust_code)
}

/// Verify that a Rust code chunk round-trips through Kilo to valid Rust.
pub fn verify_kilo_roundtrip(rust_code: &str) -> Result<bool> {
    let kilo = match rust_to_kilo(rust_code) {
        Ok(k) => k,
        Err(_) => return Ok(false),
    };
    match kilo_to_rust(&kilo) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size_is_1024() {
        assert_eq!(VOCAB.len(), 1024);
        assert_eq!(VOCAB_SIZE, 1024);
        let mut unique = std::collections::HashSet::new();
        for &t in VOCAB.iter() {
            assert!(unique.insert(t), "Duplicate token: {t}");
        }
    }

    #[test]
    fn test_round_trip_function() {
        let src = r#"
            pub fn compute_sum(a: i32, b: i32) -> i32 {
                let total = a + b;
                total
            }
        "#;
        let kilo = rust_to_kilo(src).unwrap();
        assert!(kilo.contains("fn0") || kilo.contains("compute"));
        assert!(kilo.contains("arg0"));
        assert!(kilo.contains("v0"));
        let rust = kilo_to_rust(&kilo).unwrap();
        assert!(syn::parse_file(&rust).is_ok());
    }

    #[test]
    fn test_round_trip_struct() {
        let src = r#"
            #[derive(Debug, Clone, Default)]
            pub struct WindowConfig {
                pub title: String,
                pub width: f32,
                pub height: f32,
                pub visible: bool,
            }
        "#;
        let kilo = rust_to_kilo(src).unwrap();
        assert!(kilo.contains("struct T0"));
        assert!(kilo.contains("width: f32") || kilo.contains("fld"));
        let rust = kilo_to_rust(&kilo).unwrap();
        assert!(syn::parse_file(&rust).is_ok());
    }

    #[test]
    fn test_round_trip_impl() {
        let src = r#"
            impl Default for Sphere {
                fn default() -> Self {
                    Self { radius: 0.5 }
                }
            }
        "#;
        let kilo = rust_to_kilo(src).unwrap();
        assert!(kilo.contains("impl Default for T0"));
        assert!(kilo.contains("0.5"));
        let rust = kilo_to_rust(&kilo).unwrap();
        assert!(syn::parse_file(&rust).is_ok());
    }

    #[test]
    fn test_string_placeholder() {
        let src = r#"
            fn log_msg() {
                println!("<str>");
            }
        "#;
        let kilo = rust_to_kilo(src).unwrap();
        assert!(kilo.contains(r#""<str>""#));
        let rust = kilo_to_rust(&kilo).unwrap();
        assert!(syn::parse_file(&rust).is_ok());
    }
}
