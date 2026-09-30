"""Reject Windows release archives containing Debug CRT dependencies."""

import argparse
import struct
import sys
import zipfile


DEBUG_CRT = {
    "ucrtbased.dll",
    "msvcp140d.dll",
    "vcruntime140d.dll",
    "vcruntime140_1d.dll",
    "concrt140d.dll",
}
REQUIRED = {"ninfer.exe", "ninfer-serve.exe"}


def imports(data: bytes) -> set[str]:
    """Read the PE import directory directly from packaged executable bytes."""
    if data[:2] != b"MZ":
        raise ValueError("missing DOS signature")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise ValueError("missing PE signature")
    section_count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic not in (0x10B, 0x20B):
        raise ValueError("unsupported PE optional header")
    directory = optional + (96 if magic == 0x10B else 112)
    import_rva, _ = struct.unpack_from("<II", data, directory + 8)
    if not import_rva:
        return set()

    sections = []
    for index in range(section_count):
        offset = optional + optional_size + 40 * index
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, offset + 8
        )
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def file_offset(rva: int) -> int:
        for address, size, raw in sections:
            if address <= rva < address + size:
                return raw + rva - address
        raise ValueError(f"unmapped import RVA {rva:#x}")

    result = set()
    offset = file_offset(import_rva)
    while data[offset : offset + 20] != bytes(20):
        name_rva = struct.unpack_from("<I", data, offset + 12)[0]
        name_offset = file_offset(name_rva)
        end = data.index(b"\0", name_offset)
        result.add(data[name_offset:end].decode("ascii").lower())
        offset += 20
    return result


def check(archive: str) -> None:
    with zipfile.ZipFile(archive) as package:
        names = {item.filename.lower() for item in package.infolist()}
        missing = REQUIRED - names
        if missing:
            raise ValueError(f"missing executables: {', '.join(sorted(missing))}")
        for item in package.infolist():
            if not item.filename.lower().endswith((".exe", ".dll")):
                continue
            debug_imports = imports(package.read(item)) & DEBUG_CRT
            if debug_imports:
                raise ValueError(
                    f"{item.filename} imports Debug CRT: {', '.join(sorted(debug_imports))}"
                )
    print(f"Validated Windows Release CRT imports: {archive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Windows release ZIP")
    arguments = parser.parse_args()
    try:
        check(arguments.archive)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Invalid Windows release: {error}", file=sys.stderr)
        sys.exit(1)
