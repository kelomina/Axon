import json
import sys
from pathlib import Path

import pefile

path = Path(sys.argv[1])
pe = pefile.PE(str(path), fast_load=True)
before = {}
for index, name in enumerate(("DEBUG", "BASERELOC", "TLS", "EXCEPTION", "SECURITY")):
    before[name] = [int(pe.OPTIONAL_HEADER.DATA_DIRECTORY[index].VirtualAddress), int(pe.OPTIONAL_HEADER.DATA_DIRECTORY[index].Size)]
try:
    pe.parse_data_directories(directories=[
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DEBUG"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_BASERELOC"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_TLS"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXCEPTION"],
        pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"],
    ])
except Exception as exc:
    before["parse_error"] = repr(exc)
result = {
    "machine": int(pe.FILE_HEADER.Machine),
    "directories": before,
    "attrs": {name: hasattr(pe, "DIRECTORY_ENTRY_" + name) for name in ("DEBUG", "BASERELOC", "TLS", "EXCEPTION", "SECURITY")},
    "sections": [{"raw": int(s.SizeOfRawData), "virtual": int(s.Misc_VirtualSize), "entropy": float(s.get_entropy()), "name": bytes(s.Name).decode("utf-8", "ignore").rstrip("\x00")} for s in pe.sections],
}
imports = []
for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
    names = []
    for imp in entry.imports:
        names.append({"name": None if imp.name is None else bytes(imp.name).decode("utf-8", "ignore"), "ordinal": getattr(imp, "ordinal", None), "address": int(getattr(imp, "address", 0) or 0)})
    imports.append({"dll": bytes(entry.dll).decode("utf-8", "ignore"), "names": names})
result["imports"] = imports
for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
    if bytes(entry.dll).decode("utf-8", "ignore").upper() == "WS2_32.DLL":
        result["ws2_struct"] = {
            "oft": int(entry.struct.OriginalFirstThunk),
            "ft": int(entry.struct.FirstThunk),
            "name": int(entry.struct.Name),
            "raw_oft": [int(x) for x in pe.get_data(int(entry.struct.OriginalFirstThunk), 16)],
            "raw_ft": [int(x) for x in pe.get_data(int(entry.struct.FirstThunk), 16)],
        }
print(json.dumps(result, separators=(",", ":")))
