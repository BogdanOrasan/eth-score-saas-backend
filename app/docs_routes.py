from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path
import markdown

router = APIRouter()

# backend/
#   app/
#     main.py
#     docs_routes.py
#   docs/
#     manual.md
#     admin.md
#     user-guide.md
#     disclaimer.md
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


def render_md(file_path: Path, title: str = "Docs") -> str:
    if not file_path.exists():
        return f"""
        <html><body style="font-family:system-ui;margin:24px">
          <h1>404</h1>
          <p>File not found: <code>{file_path}</code></p>
          <p><a href="/docs">Back to Docs</a></p>
        </body></html>
        """

    text = file_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "toc", "codehilite"],
    )

    return f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>{title}</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
            margin: 24px;
            max-width: 980px;
            line-height: 1.55;
          }}
          nav a {{ text-decoration: none; }}
          hr {{ margin: 20px 0; }}
          pre {{
            overflow-x: auto;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 10px;
          }}
          code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas;
          }}
        </style>
      </head>
      <body>
        <nav>
          <a href="/docs">Index</a>
          · <a href="/docs/manual">Manual</a>
          · <a href="/docs/admin">Admin</a>
          · <a href="/docs/user-guide">User Guide</a>
          · <a href="/docs/disclaimer">Disclaimer</a>
        </nav>
        <hr/>
        {html_body}
      </body>
    </html>
    """


@router.get("/docs", response_class=HTMLResponse)
def docs_index():
    return """
    <html><body style="font-family:system-ui;margin:24px">
      <h1>Docs</h1>
      <ul>
        <li><a href="/docs/manual">Manual SaaS</a></li>
        <li><a href="/docs/admin">Admin & Config</a></li>
        <li><a href="/docs/user-guide">User Guide</a></li>
        <li><a href="/docs/disclaimer">Legal & Risk Disclaimer</a></li>
      </ul>
    </body></html>
    """


@router.get("/docs/manual", response_class=HTMLResponse)
def docs_manual():
    return render_md(DOCS_DIR / "manual.md", "Manual SaaS")


@router.get("/docs/admin", response_class=HTMLResponse)
def docs_admin():
    return render_md(DOCS_DIR / "admin.md", "Admin & Config")


@router.get("/docs/user-guide", response_class=HTMLResponse)
def docs_user_guide():
    return render_md(DOCS_DIR / "user-guide.md", "User Guide")


@router.get("/docs/disclaimer", response_class=HTMLResponse)
def docs_disclaimer():
    return render_md(DOCS_DIR / "disclaimer.md", "Disclaimer")
