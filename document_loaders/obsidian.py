from pathlib import Path
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.obsidian import ObsidianLoader


class MyObsidianLoader(ObsidianLoader):
    def lazy_load(self) -> Iterator[Document]:
        paths = list(Path(self.file_path).glob("**/*.md"))
        for path in paths:
            with open(path, encoding=self.encoding) as f:
                text = f.read()

            try:
                front_matter = self._parse_front_matter(text)
                tags = self._parse_document_tags(text)
                dataview_fields = self._parse_dataview_fields(text)
                text = self._remove_front_matter(text)
                metadata = {
                    "source": str(path.name),
                    "path": str(path),
                    "created": path.stat().st_ctime,
                    "last_modified": path.stat().st_mtime,
                    "last_accessed": path.stat().st_atime,
                    **self._to_langchain_compatible_metadata(front_matter),
                    **dataview_fields,
                }

                if tags or front_matter.get("tags"):
                    metadata["tags"] = ",".join(
                        tags | set(front_matter.get("tags", []) or [])
                    )
            except:
                metadata = {
                    "source": str(path.name),
                    "path": str(path),
                    "created": path.stat().st_ctime,
                    "last_modified": path.stat().st_mtime,
                    "last_accessed": path.stat().st_atime,
                }

            yield Document(page_content=text, metadata=metadata)
