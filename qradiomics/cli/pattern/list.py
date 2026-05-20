"""qr pattern list — show all bundled pattern templates."""

import click

from qradiomics import PatternLoader


@click.command("list")
def list_patterns():
    """List all available pattern templates."""
    rows = PatternLoader().list_patterns()

    if not rows:
        click.echo("No pattern templates found.")
        return

    click.echo(f"Found {len(rows)} pattern template(s):\n")
    for i, r in enumerate(rows, 1):
        click.echo(f"{i}. {r['pattern_id']}  ({r.get('name', '')})")
        if r.get("version"):
            click.echo(f"   Version: {r['version']}")
        if r.get("tags"):
            click.echo(f"   Tags:    {', '.join(r['tags'])}")
        click.echo()


if __name__ == "__main__":
    list_patterns()
