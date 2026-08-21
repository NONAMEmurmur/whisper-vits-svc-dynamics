"""Explorer-style folder picker (IFileOpenDialog + FOS_PICKFOLDERS)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _guid(guid_str: str):
    import uuid
    from ctypes import c_byte

    return (c_byte * 16).from_buffer_copy(uuid.UUID(guid_str).bytes_le)


def pick_folder(initial: str | None = None) -> str | None:
    if not sys.platform.startswith("win"):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=initial or None, title="作業フォルダを選ぶ")
        root.destroy()
        return chosen or None

    import ctypes
    from ctypes import HRESULT, POINTER, byref, c_int, c_void_p, c_wchar_p
    from ctypes.wintypes import DWORD, HWND, LPCWSTR

    ole32 = ctypes.OleDLL("ole32")
    shell32 = ctypes.WinDLL("shell32")

    CLSCTX_INPROC_SERVER = 1
    COINIT_APARTMENTTHREADED = 0x2
    FOS_PICKFOLDERS = 0x20
    FOS_FORCEFILESYSTEM = 0x40
    SIGDN_FILESYSPATH = 0x80058000
    S_OK = 0

    clsid = _guid("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")
    iid_file_open = _guid("{D57C7288-D4AD-4768-BE02-9D969532D960}")
    iid_shell_item = _guid("{43826D1E-E718-42EE-BC55-A1E261C37BFE}")

    def vtbl(obj, index, restype, *argtypes):
        lpvtbl = ctypes.cast(obj, POINTER(c_void_p)).contents.value
        fn_ptr = ctypes.cast(lpvtbl + index * ctypes.sizeof(c_void_p), POINTER(c_void_p)).contents.value
        proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
        return proto(fn_ptr)

    ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    dialog = c_void_p()
    hr = ole32.CoCreateInstance(clsid, None, CLSCTX_INPROC_SERVER, iid_file_open, byref(dialog))
    if hr != S_OK or not dialog.value:
        raise OSError(f"CoCreateInstance failed: 0x{hr & 0xFFFFFFFF:08X}")

    try:
        set_options = vtbl(dialog, 9, HRESULT, DWORD)
        get_options = vtbl(dialog, 10, HRESULT, POINTER(DWORD))
        set_folder = vtbl(dialog, 12, HRESULT, c_void_p)
        set_title = vtbl(dialog, 17, HRESULT, LPCWSTR)
        show = vtbl(dialog, 3, HRESULT, HWND)
        get_result = vtbl(dialog, 20, HRESULT, POINTER(c_void_p))
        release = vtbl(dialog, 2, c_int)

        opts = DWORD(0)
        get_options(dialog, byref(opts))
        set_options(dialog, opts.value | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM)
        set_title(dialog, "作業フォルダを選ぶ")

        start_item = c_void_p()
        if initial and Path(initial).exists():
            shell32.SHCreateItemFromParsingName.argtypes = [LPCWSTR, c_void_p, c_void_p, POINTER(c_void_p)]
            shell32.SHCreateItemFromParsingName.restype = HRESULT
            if shell32.SHCreateItemFromParsingName(initial, None, iid_shell_item, byref(start_item)) == S_OK:
                set_folder(dialog, start_item)

        if show(dialog, None) != S_OK:
            return None

        result_item = c_void_p()
        if get_result(dialog, byref(result_item)) != S_OK or not result_item.value:
            return None

        get_display_name = vtbl(result_item, 5, HRESULT, c_int, POINTER(c_wchar_p))
        release_item = vtbl(result_item, 2, c_int)
        path_ptr = c_wchar_p()
        try:
            if get_display_name(result_item, SIGDN_FILESYSPATH, byref(path_ptr)) != S_OK or not path_ptr.value:
                return None
            return path_ptr.value
        finally:
            if path_ptr:
                ole32.CoTaskMemFree(path_ptr)
            release_item(result_item)
    finally:
        release(dialog)
        ole32.CoUninitialize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", default="")
    args = parser.parse_args()
    path = pick_folder(args.initial or None)
    if path:
        sys.stdout.write(path)
