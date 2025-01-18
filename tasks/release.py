from invoke import task


@task
def build(c):
    """Build the package."""
    c.run("poetry build")


@task
def publish(c):
    """Publish to PyPI."""
    c.run("poetry publish")
