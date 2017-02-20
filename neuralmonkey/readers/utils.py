from typing import List, Any, Callable
import magic


# pylint: disable=invalid-name
Reader = Callable[[List[str]], Any]

FILETYPER = magic.Magic(mime=True)
FILETYPER.flags |= magic.MAGIC_SYMLINK
FILETYPER.cookie = magic.magic_open(FILETYPER.flags)
magic.magic_load(FILETYPER.cookie, None)
