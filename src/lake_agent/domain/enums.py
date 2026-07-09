from enum import Enum


class Modality(str, Enum):
    TABULAR = "tabular"
    DOCUMENT = "document"
    IMAGE = "image"
    TEXT = "text"
    SEMI_STRUCTURE = "semi_structure"
    AUDIO = "audio"
    VIDEO = "video"
    EPUB = "epub"
    SLIDE_SHOW = "slide_show"
    DATABASE = "database"
    SQL_SCRIPT = "sql_script"
    ARCHIVE = "archive"
    WEB = "web"
    UNKNOWN = "unknown"


class FileStatus(str, Enum):
    IDENTIFIED = "identified"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    ERROR = "error"
