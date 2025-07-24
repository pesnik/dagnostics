from invoke.tasks import task


@task
def build(c):
    """Build the package."""
    c.run("uv build")


@task
def publish(c):
    """Publish to PyPI."""
    c.run("uv publish")


@task
def publish_test(c):
    """Publish to TestPyPI."""
    c.run("uv publish --repository testpypi")
