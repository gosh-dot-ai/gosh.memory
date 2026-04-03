#!/usr/bin/env python3
"""gosh-memory CLI.

Commands:
  start           — start MCP server
  import          — import history from file, directory, or git repo
  migrate-sqlite  — migrate one legacy key from JSON/NPZ to SQLite
  status          — check if server is running
  setup           — configure providers and API keys
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# ── start ──

def cmd_start(args):
    """Start the MCP server."""
    import uvicorn

    from .config import MemoryConfig
    from .mcp_server import SERVER_TOKEN, _save_token, create_app

    if args.server_token:
        import src.mcp_server as _mcp_mod
        _mcp_mod.SERVER_TOKEN = args.server_token
        SERVER_TOKEN = args.server_token

    cfg = MemoryConfig.from_args(args)
    app = create_app(app_data_dir=args.data_dir, app_cfg=cfg)

    token_path = _save_token()

    # Resolve embedding config for startup display
    try:
        from .setup_store import get_config as _get_cfg
        _scfg = _get_cfg()
        _embed_prov = _scfg.get("embed_provider", "openai")
        _embed_mod = _scfg.get("embed_model", "text-embedding-3-large")
    except Exception:
        _embed_prov, _embed_mod = "openai", "text-embedding-3-large"

    # TLS setup
    use_tls = args.tls or args.tls_certfile or args.tls_keyfile
    ssl_kwargs = {}
    join_token_str = None

    if use_tls:
        from .tls import ensure_cert, make_join_token
        if bool(args.tls_certfile) != bool(args.tls_keyfile):
            print("Error: --tls-certfile and --tls-keyfile must be provided together",
                  file=sys.stderr)
            sys.exit(1)
        if args.tls_certfile and args.tls_keyfile:
            certfile, keyfile = args.tls_certfile, args.tls_keyfile
        else:
            certfile, keyfile = ensure_cert(args.data_dir)
        ssl_kwargs["ssl_certfile"] = certfile
        ssl_kwargs["ssl_keyfile"] = keyfile
        scheme = "https"
        # Use advertise_host for join token so remote agents get a routable address
        advertise = getattr(args, "advertise_host", None) or args.host
        if advertise in ("0.0.0.0", "::"):
            import socket
            advertise = socket.gethostname()
        url = f"https://{advertise}:{args.port}"
        join_token_str = make_join_token(url, SERVER_TOKEN, certfile)
    else:
        scheme = "http"

    print(f"gosh.memory MCP Server")
    print(f"  Listening: {scheme}://{args.host}:{args.port}")
    print(f"  Data dir:  {Path(args.data_dir).resolve()}")
    print(f"  Embeddings:{_embed_prov} / {_embed_mod}")
    print(f"  TLS:       {'ON' if use_tls else 'off'}")
    print(f"  POST /mcp     — tool calls")
    print(f"  GET  /mcp/sse — Courier SSE")
    print(f"  Token:    {SERVER_TOKEN}")
    print(f"  Saved to: {token_path}")
    if join_token_str:
        print(f"\n  Join token (pass to agents on remote machines):")
        print(f"  gosh-agent --join {join_token_str}\n")
    uvicorn.run(app, host=args.host, port=args.port, **ssl_kwargs)


# ── status ──

def cmd_status(args):
    """Check if gosh.memory server is responding."""
    import ssl as _ssl
    import urllib.request

    scheme = "https" if args.tls else "http"
    base = f"{scheme}://{args.host}:{args.port}"

    ssl_ctx = None
    if args.tls:
        ssl_ctx = _ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = _ssl.CERT_NONE

    url = f"{base}/health"
    try:
        with urllib.request.urlopen(url, timeout=3, context=ssl_ctx) as r:
            print(f"gosh.memory running at {base}")
    except Exception:
        print(f"gosh.memory not reachable at {base}")
        sys.exit(1)

    # MCP transport check
    # 401 = Unauthorized (auth enabled — server IS running)
    # 405 = Method Not Allowed (rejects GET, expects POST)
    # 406 = Not Acceptable (content negotiation — server is running)
    mcp_url = f"{base}/mcp"
    VALID_MCP_CODES = (200, 401, 405, 406)
    try:
        req = urllib.request.Request(mcp_url, method="GET")
        with urllib.request.urlopen(req, timeout=2, context=ssl_ctx) as r:
            mcp_ok = r.status in VALID_MCP_CODES
            print(f"  MCP transport: {'ok' if mcp_ok else 'unreachable'}")
    except urllib.error.HTTPError as e:
        mcp_ok = e.code in VALID_MCP_CODES
        auth_note = " (auth enabled)" if e.code == 401 else ""
        print(f"  MCP transport: {'ok' if mcp_ok else 'unreachable'}{auth_note}")
    except Exception:
        print(f"  MCP transport: unreachable")


# ── import ──

def cmd_import(args):
    """Import history into gosh.memory."""
    from .git_importer import import_git
    from .importers import SUPPORTED_FORMATS, parse_history
    from .memory import MemoryServer

    fmt = args.format
    options = json.loads(args.options) if args.options else {}

    # Class B: source-based formats
    if fmt == "git":
        if not args.source_uri:
            print("Error: --source-uri required for git format", file=sys.stderr)
            sys.exit(1)
        # Inject token from --token arg (takes precedence over options["token"])
        if hasattr(args, "token") and args.token:
            options = {**options, "token": args.token}
        sessions = import_git(args.source_uri, options)
        source_label = args.source_uri
    # Class A: content-based formats
    else:
        if fmt not in SUPPORTED_FORMATS:
            print(f"Error: unknown format {fmt!r}. "
                  f"Supported: {sorted(SUPPORTED_FORMATS | {'git'})}", file=sys.stderr)
            sys.exit(1)
        if fmt == "directory":
            if not args.dir:
                print("Error: --dir required for directory format", file=sys.stderr)
                sys.exit(1)
            dir_path = Path(args.dir)
            # H9 fix: use rglob for recursive traversal of nested directories
            IMPORTABLE_SUFFIXES = {'.txt', '.md', '.json', '.py', '.rst',
                                   '.yaml', '.yml', '.csv', '.xml', '.html',
                                   '.log', '.cfg', '.ini', '.toml'}
            parts = []
            for f in sorted(dir_path.rglob("*")):
                if f.is_file() and f.suffix in IMPORTABLE_SUFFIXES:
                    try:
                        rel = f.relative_to(dir_path)
                    except ValueError:
                        rel = f.name
                    parts.append(f"---FILE: {rel}---")
                    parts.append(f.read_text(encoding="utf-8", errors="replace"))
            content = "\n".join(parts)
        else:
            if not args.file:
                print("Error: --file required", file=sys.stderr)
                sys.exit(1)
            content = Path(args.file).read_text(encoding="utf-8", errors="replace")
        sessions = parse_history(fmt, content)
        source_label = args.file or args.dir

    if not sessions:
        print("No sessions parsed. Check input file.")
        sys.exit(1)

    print(f"Parsed {len(sessions)} sessions from {source_label}")

    from .config import MemoryConfig
    from .setup_store import get_config as _get_config
    _cfg = _get_config()
    _extract_model = _cfg.get("extraction_model", MemoryConfig().extraction_model)

    server = MemoryServer(
        data_dir=args.data_dir,
        key=args.key,
        agent_id=args.agent_id,
        swarm_id=args.swarm_id,
        extract_model=_extract_model,
    )
    print(f"Using extraction model: {_extract_model}")

    total_facts = 0
    errors = []

    async def run():
        nonlocal total_facts
        for s in sessions:
            try:
                result = await server.store(
                    content=s["content"],
                    session_num=s["session_num"],
                    session_date=s["session_date"],
                    speakers=s.get("speakers", "User and Assistant"),
                    agent_id=args.agent_id,
                    swarm_id=args.swarm_id,
                    scope=args.scope,
                    content_type=args.content_type,
                )
                n = result.get("facts_extracted", 0)
                total_facts += n
                print(f"  [{s['session_num']:3d}] {s['session_date']}  {n} facts")
            except Exception as e:
                errors.append({"session": s["session_num"], "error": str(e)})
                print(f"  [{s['session_num']:3d}] ERROR: {e}", file=sys.stderr)

    asyncio.run(run())
    print(f"\nDone: {len(sessions) - len(errors)} sessions, {total_facts} facts.")

    # C3: notify running server to reload data from disk
    port = args.port
    try:
        import urllib.request
        # Read token from saved file
        token = ""
        token_path = Path.home() / ".gosh-memory" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["x-server-token"] = token
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/admin/reload",
            data=json.dumps({"key": args.key}).encode(),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
        print("Server notified — data reloaded.")
    except Exception:
        print("Note: could not notify running server — restart may be needed.")

    if errors:
        print(f"{len(errors)} errors.", file=sys.stderr)
        sys.exit(1)


# ── migrate-sqlite ──

def cmd_migrate_sqlite(args):
    """Migrate one memory key from legacy JSON/NPZ storage to SQLite."""
    from .storage import migrate_jsonnpz_to_sqlite

    data_dir = Path(args.data_dir)
    sqlite_path = data_dir / f"{args.key}.sqlite3"
    legacy_path = data_dir / f"{args.key}.json"

    if sqlite_path.exists() and not legacy_path.exists():
        print(f"Key '{args.key}' already uses SQLite: {sqlite_path}")
        return

    try:
        result = migrate_jsonnpz_to_sqlite(str(data_dir), args.key)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"SQLite migration failed for key '{args.key}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Migrated key '{args.key}' to SQLite: {result['sqlite_path']}")
    print("Legacy files were backed up with .bak suffixes.")


# ── setup ──

def cmd_setup(args):
    """Configure providers, API keys, and models."""
    from .setup_store import (
        get_api_key,
        is_configured,
        list_configured_providers,
        load_config,
        save_config,
        store_api_key,
    )

    # --show: display current config
    if args.show:
        import os

        config = load_config()
        if not config:
            print("Not configured. Run: gosh-memory setup")
            return
        print(f"Provider:   {config.get('provider', '(not set)')}")
        models = config.get("models", {})
        print(f"Extraction: {models.get('extraction', '(default)')}")
        print(f"Inference:  {models.get('inference', '(default)')}")
        print(f"Judge:      {models.get('judge', '(default)')}")

        # Show API key source for the configured provider
        provider = config.get("provider", "")
        key = None
        key_source = "(not set)"

        # Check env var first (inline map to avoid heavy providers import)
        _PROVIDER_ENV = {
            "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY",
            "inception": "INCEPTION_API_KEY",
        }
        env_var = _PROVIDER_ENV.get(provider)
        if env_var and os.environ.get(env_var):
            key = os.environ[env_var]
            key_source = f"env var ${env_var}"

        if not key:
            # Check config file
            api_keys = config.get("api_keys", {})
            if provider in api_keys:
                key = api_keys[provider]
                key_source = "~/.gosh-memory/config.json (plaintext, chmod 600)"

        print(f"API key:    {'***' if key else '(not set)'} ({key_source})")

        from .config import MemoryConfig
        emb = config.get("embedding", {})
        print(f"Embedding:  {emb.get('provider', 'openai')} / {emb.get('model', MemoryConfig().embed_model)}")
        return

    # Non-interactive: --provider + --api-key
    if args.provider and args.api_key:
        store_api_key(args.provider, args.api_key)
        # Also save provider to config
        cfg = load_config()
        cfg["provider"] = args.provider
        save_config(cfg)
        print(f"Provider: {args.provider}, API key stored.")
        return

    if args.provider and not args.api_key:
        print("Error: --api-key required with --provider", file=sys.stderr)
        sys.exit(1)

    # ── Interactive wizard ──
    print()
    print("  gosh.memory — Setup Wizard")
    print("  " + "=" * 36)
    print()

    config = load_config()

    # Step 1: LLM provider
    providers = [
        ("groq",      "Groq      — qwen, llama (cheapest)"),
        ("inception", "Inception — mercury-2 (fast extraction)"),
        ("openai",    "OpenAI    — gpt-4.1-mini, gpt-4o, o3"),
        ("anthropic", "Anthropic — claude-sonnet, claude-opus"),
        ("google",    "Google    — gemini-2.5-pro, gemini-2.0-flash"),
    ]

    print("  LLM Provider:")
    for i, (pid, desc) in enumerate(providers, 1):
        existing = get_api_key(pid)
        marker = " [configured]" if existing else ""
        print(f"    {i}) {desc}{marker}")

    choice = input(f"\n  Select [1-{len(providers)}, default=1]: ").strip() or "1"
    try:
        idx = int(choice) - 1
        if not 0 <= idx < len(providers):
            idx = 0
        provider_id = providers[idx][0]
    except ValueError:
        provider_id = "groq"

    print()
    print("  API keys stored in ~/.gosh-memory/config.json (plaintext, chmod 600).")
    print("  For production, prefer environment variables instead.")
    print()

    # Step 2: API key
    existing_key = get_api_key(provider_id)
    if existing_key:
        masked = existing_key[:8] + "..." + existing_key[-4:]
        print(f"\n  Current key: {masked}")
        change = input("  Update key? [y/N]: ").strip().lower()
        if change == "y":
            key = input(f"  {provider_id} API key: ").strip()
            if key:
                store_api_key(provider_id, key)
                print("  Key stored in config file.")
        else:
            print("  Keeping existing key.")
    else:
        key = input(f"\n  {provider_id} API key: ").strip()
        if key:
            store_api_key(provider_id, key)
            print("  Key stored in config file.")
        else:
            print("  Skipped. Set later or use env var.")

    # Step 3: Model selection
    default_models = {
        "groq":      {"extraction": "qwen/qwen3-32b",
                      "inference":  "qwen/qwen3-32b",
                      "judge":      "qwen/qwen3-32b"},
        "inception": {"extraction": "inception/mercury-2",
                      "inference":  "qwen/qwen3-32b",
                      "judge":      "qwen/qwen3-32b"},
        "openai":    {"extraction": "gpt-4.1-mini",
                      "inference":  "gpt-4.1-mini",
                      "judge":      "gpt-4.1-mini"},
        "anthropic": {"extraction": "anthropic/claude-sonnet-4-6",
                      "inference":  "anthropic/claude-sonnet-4-6",
                      "judge":      "anthropic/claude-sonnet-4-6"},
        "google":    {"extraction": "google/gemini-2.0-flash",
                      "inference":  "google/gemini-2.5-pro",
                      "judge":      "google/gemini-2.0-flash"},
    }

    defaults = default_models.get(provider_id, default_models["groq"])

    print(f"\n  Models (press Enter for defaults):")
    extraction = input(f"    Extraction [{defaults['extraction']}]: ").strip() or defaults["extraction"]
    inference  = input(f"    Inference  [{defaults['inference']}]: ").strip() or defaults["inference"]
    judge      = input(f"    Judge      [{defaults['judge']}]: ").strip() or defaults["judge"]

    # Step 4: Embedding
    print(f"\n  Embedding:")
    print(f"    1) OpenAI API — text-embedding-3-large (best quality)")
    print(f"    2) Local     — sentence-transformers (free, no API key)")

    embed_choice = input("  Select [1-2, default=1]: ").strip() or "1"
    if embed_choice == "2":
        embed_provider = "local"
        LOCAL_EMBED_MODELS = [
            ("BAAI/bge-large-en-v1.5",         "MTEB retrieval ~54%, MIT, 512 token ctx, ~1.3GB"),
            ("nomic-ai/nomic-embed-text-v1.5",  "2048 token ctx, MIT, ~550MB"),
        ]
        print("\n  Local embedding model:")
        for i, (name, desc) in enumerate(LOCAL_EMBED_MODELS, 1):
            print(f"    {i}) {name}")
            print(f"       {desc}")
        while True:
            mchoice = input(f"  Model [1-{len(LOCAL_EMBED_MODELS)}, default=1]: ").strip() or "1"
            try:
                midx = int(mchoice) - 1
                if 0 <= midx < len(LOCAL_EMBED_MODELS):
                    embed_model = LOCAL_EMBED_MODELS[midx][0]
                    break
            except ValueError:
                pass
    else:
        embed_provider = "openai"
        embed_model = "text-embedding-3-large"
        if provider_id != "openai" and not get_api_key("openai"):
            print("\n  OpenAI embeddings require an OpenAI API key.")
            oai_key = input("  OpenAI API key: ").strip()
            if oai_key:
                store_api_key("openai", oai_key)
                print("  Key stored.")
            else:
                print("  Warning: No OpenAI key. Embeddings will fail unless")
                print("  OPENAI_API_KEY env var is set at runtime.")

    # Reload config from disk — includes api_keys written by store_api_key() above
    config = load_config()
    config["provider"] = provider_id
    config["models"] = {
        "extraction": extraction,
        "inference":  inference,
        "judge":      judge,
    }
    config["embedding"] = {
        "provider": embed_provider,
        "model":    embed_model,
    }
    save_config(config)

    print(f"\n  Config saved to ~/.gosh-memory/config.json")
    print(f"  API keys stored in plaintext config (chmod 600)")
    print(f"\n  Ready! Start the server:")
    print(f"    gosh-memory start")


# ── main ──

def main():
    parser = argparse.ArgumentParser(
        prog="gosh-memory",
        description="gosh.memory — semantic memory for AI agents.",
        epilog="Run 'gosh-memory <command> --help' for command-specific options."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # start
    p_start = sub.add_parser("start", help="Start MCP server",
                              description="Start the gosh.memory MCP server. "
                                          "Exposes tools over HTTP + SSE on /mcp.")
    p_start.add_argument("--host",     default="127.0.0.1",
                         help="Host to bind (default: 127.0.0.1)")
    p_start.add_argument("--port",     type=int, default=8765,
                         help="Port to listen on (default: 8765)")
    p_start.add_argument("--data-dir", default="./data", dest="data_dir",
                         help="Directory for data storage (default: ./data)")
    p_start.add_argument("--model",    default=None,
                         help="Model shortcut — sets all stages at once")
    p_start.add_argument("--extraction-model", default=None, dest="extraction_model",
                         help="Extraction model override")
    p_start.add_argument("--inference-model",  default=None, dest="inference_model",
                         help="Inference model override")
    p_start.add_argument("--judge-model",      default=None, dest="judge_model",
                         help="Judge model override")
    p_start.add_argument("--embed-model",      default=None, dest="embed_model",
                         help="Embedding model override")
    p_start.add_argument("--server-token",     default=None, dest="server_token",
                         help="Server auth token (auto-generated if not set)")
    p_start.add_argument("--tls",             action="store_true", default=False,
                         help="Enable HTTPS with auto-generated self-signed certificate")
    p_start.add_argument("--tls-certfile",    default=None, dest="tls_certfile",
                         help="Custom TLS certificate file (implies --tls)")
    p_start.add_argument("--tls-keyfile",     default=None, dest="tls_keyfile",
                         help="Custom TLS key file (implies --tls)")
    p_start.add_argument("--advertise-host", default=None, dest="advertise_host",
                         help="Routable hostname/IP for join token (default: --host, "
                              "auto-detected if 0.0.0.0)")

    # status
    p_status = sub.add_parser("status", help="Check if server is running",
                               description="Ping the server health endpoint. "
                                           "Exits with code 1 if not reachable.")
    p_status.add_argument("--host", default="127.0.0.1",
                          help="Server host (default: 127.0.0.1)")
    p_status.add_argument("--port", type=int, default=8765,
                          help="Server port (default: 8765)")
    p_status.add_argument("--tls", action="store_true", default=False,
                          help="Use HTTPS for health check")

    # import
    p_import = sub.add_parser("import", help="Import conversation history",
                              description="Import history from a file, directory, or git repo "
                                          "into gosh.memory for semantic retrieval.")
    p_import.add_argument("--format",       required=True,
                          choices=["conversation_json", "text", "directory", "git"],
                          help="Input format")
    p_import.add_argument("--file",         help="Input file (Class A formats)")
    p_import.add_argument("--dir",          help="Input directory (directory format)")
    p_import.add_argument("--source-uri",   dest="source_uri",
                          help="Source URI (Class B formats: git)")
    p_import.add_argument("--options",      default=None,
                          help='Format-specific JSON options, e.g. \'{"branch":"main"}\'')
    p_import.add_argument("--key",          default="default",
                          help="Memory server key — namespace for this import (default: default)")
    p_import.add_argument("--agent-id",     default="default", dest="agent_id",
                          help="Agent identity tag (default: default)")
    p_import.add_argument("--swarm-id",     default="default", dest="swarm_id",
                          help="Swarm identity tag (default: default)")
    p_import.add_argument("--scope",        default="agent-private",
                          choices=["agent-private", "swarm-shared", "system-wide"],
                          help="Scope for imported facts (default: agent-private)")
    p_import.add_argument("--content-type", default="default", dest="content_type",
                          help="Librarian prompt type: default|technical|financial|...")
    p_import.add_argument("--token",         default=None,
                          help="Personal access token for private repos "
                               "(injected into clone URL, not logged)")
    p_import.add_argument("--port",          type=int, default=8765,
                          help="Server port for reload notification (default: 8765)")
    p_import.add_argument("--data-dir",     default="./data", dest="data_dir",
                          help="Directory for data storage (default: ./data)")

    # migrate-sqlite
    p_migrate = sub.add_parser("migrate-sqlite", help="Migrate one key from legacy JSON/NPZ storage to SQLite",
                               description="Create {key}.sqlite3 from legacy JSON/NPZ storage, verify a lossless "
                                           "round-trip, and back up legacy files with .bak suffixes.")
    p_migrate.add_argument("--key", required=True,
                           help="Memory server key to migrate")
    p_migrate.add_argument("--data-dir", default="./data", dest="data_dir",
                           help="Directory containing storage files (default: ./data)")

    # setup
    p_setup = sub.add_parser("setup", help="Configure providers and API keys",
                              description="Interactive wizard to configure LLM provider, "
                                          "API keys, and model selection. Keys stored in "
                                          "plaintext config file (~/.gosh-memory/config.json, "
                                          "chmod 600). For production, prefer env vars.")
    p_setup.add_argument("--provider",  default=None,
                         choices=["openai", "groq", "anthropic", "google", "inception"],
                         help="Provider (non-interactive mode)")
    p_setup.add_argument("--api-key",   default=None, dest="api_key",
                         help="API key (non-interactive mode)")
    p_setup.add_argument("--embed-provider", default=None, dest="embed_provider",
                         choices=["openai", "local"],
                         help="Embedding provider (openai or local)")
    p_setup.add_argument("--show",      action="store_true",
                         help="Show current configuration")

    args = parser.parse_args()
    {
        "start":          cmd_start,
        "status":         cmd_status,
        "import":         cmd_import,
        "migrate-sqlite": cmd_migrate_sqlite,
        "setup":          cmd_setup,
    }[args.command](args)


if __name__ == "__main__":
    main()
