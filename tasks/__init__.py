from invoke import Collection

from tasks import dev, lint, release

ns = Collection()
ns.add_collection(Collection.from_module(dev))
ns.add_collection(Collection.from_module(lint))
ns.add_collection(Collection.from_module(release))
