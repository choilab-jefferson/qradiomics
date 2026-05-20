"""qr pattern search — case-insensitive keyword search over pattern templates."""

import click

from qradiomics import PatternLoader


@click.command("search")
@click.argument("query")
def search_patterns(query):
    """Search pattern templates by keyword (matches id, name, and tags)."""
    rows = PatternLoader().list_patterns()
    q = query.lower()

    def _match(r):
        haystack = " ".join(
            [
                r.get("pattern_id", ""),
                r.get("name", ""),
                " ".join(r.get("tags") or []),
            ]
        ).lower()
        return q in haystack

    hits = [r for r in rows if _match(r)]
    if not hits:
        click.echo(f"No patterns matched '{query}'.")
        return

    click.echo(f"Found {len(hits)} pattern(s) matching '{query}':\n")
    for i, r in enumerate(hits, 1):
        click.echo(f"{i}. {r['pattern_id']}  ({r.get('name', '')})")
        if r.get("tags"):
            click.echo(f"   Tags: {', '.join(r['tags'])}")
        click.echo()


if __name__ == "__main__":
    search_patterns()
